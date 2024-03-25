import os
from functools import partial
from glob import glob

import h5py
import jax
import jax.profiler
import numpy as np
import ray
import requests
from tqdm import tqdm

from bio_clip.data.alphafold_scripts.protein import from_pdb_string
from bio_clip.data.downstream.shared import (
    alphafold_to_legacy_bioclip,
    filter_out_sample,
    get_processing_function,
)
from bio_clip.data.protein_datastructure import ProteinStructureSample
from bio_clip.types import BatchDataBioClip, ProteinInteractionBatch


def download_raw(config):
    benchmark = config.training.data.benchmark
    ppi_benchmark_folder = f"{config.training.data.path}/{benchmark}.npy"

    # get the labels and pdb-codes
    data = np.load(ppi_benchmark_folder, allow_pickle=True)
    if benchmark == "cerevisiae":
        # [['P40484', 'P29340', '0', '2HJN', '2Y9M'], ...]
        data = data[:, [3, 4, 2]]
    elif benchmark == "human":
        # [['NP_663777', 'Q13114', '1FLK', 'NP_001233', 'P26842', '5TL5', '1'], ...]
        data = data[:, [2, 5, 6]]  # [pdb_code_1, pdb_code_2, interaction]
    return data


def preprocess_files(pdb_codes, output_h5_file, num_chunks=10):
    """Convert list of pdb codes into alphafold.data.protein.Protein objects in a h5
    file"""
    process_protein = True

    def prot_processing(pdb_code, prot):
        _sample = alphafold_to_legacy_bioclip(
            identifier=pdb_code,
            aatype=prot.aatype,
            residue_index=prot.residue_index,
            atom37_positions=prot.atom_positions,
            atom37_gt_exists=prot.atom_mask,
            bioclip_preprocess_fn=(lambda x: x),
            filter_sample=None,
            verbose=False,
            chain_index=prot.chain_index,
        )
        dict_format = {
            a: getattr(_sample, a)
            for a in [
                "chain_id",
                "nb_residues",
                "aatype",
                "atom37_positions",
                "atom37_gt_exists",
                "atom37_atom_exists",
                "resolution",
                "pdb_cluster_size",
            ]
        }
        return dict_format

    @ray.remote
    def ray_preprocess(code):
        try:
            url = f"https://files.rcsb.org/view/{code.upper()}.pdb"
            response = requests.get(url)
            response.raise_for_status()
            prot = from_pdb_string(response.text)
            if process_protein:
                prot = prot_processing(code, prot)
            else:
                prot = prot.__dict__
        except Exception as e:
            print(e)
            prot = None
        return prot

    def save_to_h5(code, data_dict, mode):
        with h5py.File(output_h5_file, mode) as f:
            if code in f:
                return "a"
            grp = f.create_group(code)
            for key, value in data_dict.items():
                # Check if the dtype of the array is a Unicode string
                if hasattr(value, "dtype") and value.dtype.kind == "U":
                    # Convert to a variable-length string data type
                    dt = h5py.special_dtype(vlen=str)
                    var_length_value = np.array(value, dtype=dt)
                    grp.create_dataset(key, data=var_length_value)
                else:
                    # Handle other data types normally
                    grp.create_dataset(key, data=value)
        return "a"

    def save_subset(codes, mode):
        ray.init()
        futures = [ray_preprocess.remote(pc) for pc in codes]
        results = ray.get(futures)
        for fp, res in zip(codes, results):
            if res is not None:
                mode = save_to_h5(fp, res, mode)
        print(f"prop: {sum(res is not None for res in results) / len(results)}")
        ray.shutdown()
        return mode

    pdb_codes_chunks = np.array_split(pdb_codes, num_chunks)
    mode = "w"
    for chunk in tqdm(pdb_codes_chunks):
        mode = save_subset(chunk, mode)


def cropping_dependent_processing(
    config, tokenizer, num_devices, processed_h5_path, num_chunks=10, save_chunks=2
):
    """This task is now deprecated as there were problems found in the original paper.
    Additionally, this task was previously pre-processed using an old version of the
    BioCLIP code which we do not provide. Data-transform dependent pre-processing takes
    quite long, thus it is processed in parallel, cached, then training is performed.

    Args:
        config (_type_): _description_
        tokenizer (_type_): ESM tokeniser.
        num_devices (int):
        processed_h5_path (str): All the preprocessed data.
        num_chunks (int, optional): _description_. Defaults to 10.
        save_chunks (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    data = download_raw(config)

    fixes_sizes = config.training.data.fixed_sizes
    recompute_vars = (
        f"{fixes_sizes.maximum_padding}pad_{fixes_sizes.graph_max_neighbor}nn"
    )

    chunk_dir = "chunks"
    cache_dir = f"ppi_cache_{recompute_vars}.npy"

    def compute_size(collected):
        m1 = np.array([p in collected for p, _, _ in data])
        m2 = np.array([p in collected for _, p, _ in data])
        (ixs,) = np.where(m1 * m2)
        print(f"We got {len(ixs)} / {len(data)} ({m1.sum()}, {m2.sum()})")
        elems = sum(
            jax.tree_leaves(jax.tree_map(lambda v: v.ravel().shape[0], collected))
        )
        mbs = (elems // (1024 * 1024)) * 8
        print(f"1 chunk of data has size {mbs}MB")
        return ixs

    # remove other caches- for different values of num-neighbours or crop size.
    ppi_caches = glob("ppi_cache_*")
    if cache_dir in ppi_caches:
        ppi_caches.remove(cache_dir)
    for fn in ppi_caches:
        os.system(f"rm {fn}")

    if not os.path.isdir(processed_h5_path):
        print("PREPARING DATA: Downloading and parsing PDB files...")
        codes = set()
        for c1, c2, _ in data:
            codes.add(c1)
            codes.add(c2)
        preprocess_files(list(codes), processed_h5_path)

    if not os.path.isfile(cache_dir):
        print("PREPARING DATA: Creating features...")
        # apply the data-transforms here, in this case num-neighbours and crop size.
        # run in parallel, save in cache.
        bioclip_legacy_save_to_bioclip_training_data = get_processing_function(
            config,
            tokenizer,
            fixes_sizes.maximum_padding,
            fixes_sizes.graph_max_neighbor,
        )
        # Max residue has -2 to account for beginning and end-of-sequence token,
        filter_out_sample_fn = partial(
            filter_out_sample,
            min_number_valid_residues=fixes_sizes.graph_max_neighbor,
            max_number_residues=fixes_sizes.maximum_padding - 2,
        )

        @ray.remote
        def _processing_fn(pdb_code):
            with h5py.File(processed_h5_path, "r") as f:
                sample = ProteinStructureSample(
                    **{k: np.array(v, dtype=v.dtype) for k, v in f[pdb_code].items()}
                )
            if filter_out_sample_fn(sample):
                return None
            try:
                _sample = bioclip_legacy_save_to_bioclip_training_data(sample)
                out = BatchDataBioClip(_sample.graph, _sample.tokens[None, :])
            except Exception as e:
                e
                out = None
            return out

        with h5py.File(processed_h5_path, "r") as f:
            pdb_codes = list(f.keys())

        os.environ["TMPDIR"] = "/tmp/ramdisk"

        def save_chunk(file_paths, processed_structures):
            ray.init()
            futures = [_processing_fn.remote(c) for c in pdb_codes]
            results = ray.get(futures)
            for c, res in zip(file_paths, results):
                if res is not None:
                    processed_structures[c] = res
            print(f"prop: {sum(res is not None for res in results) / len(results)}")
            ray.shutdown()
            return processed_structures

        os.makedirs(chunk_dir, exist_ok=True)
        pdb_code_chunks = np.array_split(pdb_codes, num_chunks)

        def secondary_slice(num_chunks, save_chunks):
            slice_length = num_chunks // save_chunks
            slices = [
                slice(i * slice_length, (i + 1) * slice_length)
                for i in range(save_chunks)
            ]
            if slices[-1].stop < num_chunks:
                slices.append(slice(slices[-1].stop, num_chunks))
            return slices

        outer_chunks = [
            pdb_code_chunks[slc] for slc in secondary_slice(num_chunks, save_chunks)
        ]
        assert len(outer_chunks) == save_chunks
        for i in range(save_chunks):
            print(
                f"Running first {len(outer_chunks[i])} chunks / {len(pdb_code_chunks)}"
            )
            processed_structures = {}
            for chunk in tqdm(outer_chunks[i]):
                processed_structures = save_chunk(chunk, processed_structures)
            compute_size(processed_structures)
            np.save(f"{chunk_dir}/chunk_{i}.npy", processed_structures)
            del processed_structures
        # merge chunks
        processed_structures = {}
        for fn in os.listdir(chunk_dir):
            processed_structures = {
                **processed_structures,
                **np.load(f"{chunk_dir}/{fn}", allow_pickle=True).item(),
            }
            os.system(f"rm {chunk_dir}/{fn}")
        np.save(cache_dir, processed_structures)
    else:
        processed_structures = np.load(cache_dir, allow_pickle=True).item()
    ixs = compute_size(processed_structures)
    data = data[ixs]

    def processing_fn(index):
        code1, code2, target = data[index]
        return ProteinInteractionBatch(
            batch_data_bioclip=processed_structures[code1],
            target=np.array([int(target)]),
            batch_data_bioclip_2=processed_structures[code2],
        )

    trn_bds = (num_devices, config.training.batch.num_per_device_update)

    def dataloader(_ixs):
        # pass by reference, and we're popping
        ixs = list(_ixs)
        print(f"Running on {len(ixs)} examples")

        batch_size = num_devices * config.training.batch.num_per_device_update
        while len(ixs) >= batch_size:
            batch = []
            batch_ixs = []
            while len(batch) < batch_size and len(ixs):
                i = ixs.pop(0)
                # processing function
                try:
                    data = processing_fn(i)
                    if data is not None:
                        batch.append(data)
                        batch_ixs.append(i)
                except Exception as e:
                    e
                    pass
            if len(batch) == batch_size:
                batch = jax.tree_map(
                    lambda *x: np.stack(x).reshape((*trn_bds, *x[0].shape)),
                    *batch,
                )
                mask = np.ones(trn_bds, dtype=bool)
                yield batch_ixs, batch, mask

    return dataloader, len(data), processing_fn, data
