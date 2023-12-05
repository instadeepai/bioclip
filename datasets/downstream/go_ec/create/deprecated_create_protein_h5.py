import gzip
import json
import multiprocessing
import os

import h5py
import numpy as np

#  Make sure to add DeepFRI to your python path
from deepfrier.utils import load_EC_annot, load_GO_annot
from tqdm import tqdm

from bio_clip.data.alphafold_scripts.protein import from_mmcif_string


def add_structure_to_h5(h5_path, identifier, protein):
    with h5py.File(h5_path, "a", libver="latest") as hf:
        hf[f"input_features/{identifier}/aatype"] = protein.aatype.astype(np.int32)
        hf[
            f"input_features/{identifier}/atom_positions"
        ] = protein.atom_positions.astype(
            np.float32
        )  # np.string_)
        hf[f"input_features/{identifier}/atom_mask"] = protein.atom_mask.astype(
            np.int32
        )
        hf[f"input_features/{identifier}/residue_index"] = protein.residue_index.astype(
            np.int32
        )
        hf[f"input_features/{identifier}/b_factors"] = protein.b_factors.astype(
            np.float32
        )

        # hf[f"{identifier}/output_features/{onto}"] = target.astype(np.int32)


def finalise_dataset(identifiers, h5_path, ec_annot_fn, go_annot_fn, delete=False):
    # if the .npz file exist from the official DeepFRI pipeline, provide a list of those
    # then the intersection can be computed later
    original_df_files = os.listdir("./data/annot_pdb_chains_npz/")
    original_df_files = {f.replace(".npz", "") for f in original_df_files}
    identifiers_set = set(identifiers)
    print(f"\nNum examples from AlphaFold pipeline: {len(identifiers_set)}")
    print(f"Num examples from DeepFRI pipeline: {len(original_df_files)}")
    common = original_df_files.intersection(identifiers_set)
    print(f"Num examples common to both pipelines: {len(common)}")
    """
    Num examples from AlphaFold pipeline: 40383
    Num examples from DeepFRI pipeline: 36593
    Num examples common to both pipelines: 36163
    """

    def compute_mask(key_set):
        id_mask = np.zeros(len(identifiers), dtype=np.int32)
        for i, _id in enumerate(identifiers):
            id_mask[i] = _id in key_set
        return id_mask

    # compute the examples common to deepfri:[train, val, set] and the AlphaFold
    # features for [EC, GO]
    prefixes = {
        "GO": "data/nrPDB-EC_2020.04_%s.txt",
        "EC": "data/nrPDB-GO_2019.06.18_%s.txt",
    }
    deepfri_split = {}
    for p, pf in prefixes.items():
        deepfri_split[p] = {}
        for split in ["train", "test", "valid"]:
            with open(pf % split) as f:
                deepfri_split[p][split] = set(f.read().split("\n"))

    print("DeepFRI inputs in split:")
    print({a: {c: len(d) for c, d in b.items()} for a, b in deepfri_split.items()})

    # GO/EC annotation, with both DeepFRI and AF inputs
    af_keys = {
        "EC": set(load_EC_annot(ec_annot_fn)[0].keys()).intersection(common),
        "GO": set(load_GO_annot(go_annot_fn)[0].keys()).intersection(common),
    }
    common_tr_ts_vl = {}
    for ecgo, splits in deepfri_split.items():
        ids = af_keys[ecgo]
        common_tr_ts_vl[ecgo] = {}
        for split, df_ids in splits.items():
            common_tr_ts_vl[ecgo][split] = df_ids.intersection(ids)

    print("Common inputs in split:")
    print({a: {c: len(d) for c, d in b.items()} for a, b in common_tr_ts_vl.items()})

    with h5py.File(h5_path, "a", libver="latest") as hf:
        for ecgo, splits in common_tr_ts_vl.items():
            for split, ids in splits.items():
                if delete:
                    del hf[f"{ecgo}_split/{split}_ids"]
                    del hf[f"{ecgo}_split/{split}_ids_mask"]
                hf[f"{ecgo}_split/{split}_ids"] = np.array(list(ids)).astype(np.string_)
                hf[f"{ecgo}_split/{split}_ids_mask"] = compute_mask(ids)

    dt = "s3://deepchain-research/bio_clip/deepfri_alphafold_protein.h5"
    os.system(
        f"aws s3 cp updated_full_dataset.h5 {dt} --endpoint=https://s3.kao.instadeep.io"
    )


def create_h5_file(h5_path, error_path, pdb_directory, num_processes):
    pdb_filenames = {fn.split(".")[0] for fn in os.listdir(pdb_directory)}
    ec_annot_fn = "/app/DeepFRI/preprocessing/data/nrPDB-EC_2020.04_annot.tsv"
    go_annot_fn = "/app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv"
    ec = load_EC_annot(ec_annot_fn)
    go = load_GO_annot(go_annot_fn)

    ec_pdbs = {code.split("-")[0] for code in ec[0]}
    go_pdbs = {code.split("-")[0] for code in go[0]}
    code_hypn_chains = list(set(ec[0].keys()).union(set(go[0].keys())))

    codes_we_need = list((ec_pdbs.union(go_pdbs)).intersection(pdb_filenames))
    print(
        f"codes_we_need: {len(codes_we_need)}\nec_pdbs: {len(ec_pdbs)}\n"
        f"go_pdbs: {len(go_pdbs)}\npdb_filenames: {len(pdb_filenames)}"
    )
    # codes_we_need: 36301
    # ec_pdbs: 18573
    # go_pdbs: 32397
    # pdb_filenames: 192090

    with h5py.File(h5_path, "w", libver="latest") as hf:
        for name, (prot2annot, goterms, gonames, counts) in [("ec", ec), ("go", go)]:
            for onto, gt in goterms.items():
                hf[f"{name}/goterms/{onto}"] = np.array(gt).astype(np.string_)
            for onto, gn in gonames.items():
                hf[f"{name}/gonames/{onto}"] = np.array(gn).astype(np.string_)
            for onto, ct in counts.items():
                hf[f"{name}/counts/{onto}"] = np.array(ct).astype(np.int32)
            identifiers = list(prot2annot.keys())
            hf[f"{name}/identifiers"] = np.array(identifiers).astype(np.string_)
            for onto in goterms.keys():
                hf[f"{name}/{onto}"] = np.stack(
                    [prot2annot[_id][onto] for _id in identifiers]
                ).astype(np.int32)
                # easier format for data-loading later
                for _id in tqdm(identifiers):
                    hf[f"targets/{name}/{onto}/{_id}"] = prot2annot[_id][onto].astype(
                        np.int32
                    )
    print(f"Written targets to {h5_path}")

    # do multiprocessing of from_pdb_string
    manager = multiprocessing.Manager()
    output = manager.dict()
    temp = "temp"
    os.makedirs(temp, exist_ok=True)

    def run_chunk(slice_pdb_code_chains, chunk_ix):
        print(f"Will run {len(slice_pdb_code_chains)} chains")
        chunk = {}
        failed = []
        for i, pdb_code_chain in enumerate(slice_pdb_code_chains):
            if i % 10 == 0:
                print(f"done: {i} / {len(slice_pdb_code_chains)}")
            with open(
                os.path.join(
                    pdb_directory, f"{str(pdb_code_chain).split('-')[0]}.cif.gz"
                ),
                "rb",
            ) as f:
                cif_str = gzip.decompress(f.read()).decode("ascii")
            try:
                chunk[str(pdb_code_chain)] = from_mmcif_string(
                    cif_str, pdb_code_chain.split("-")[-1]
                )
            except Exception as e:
                failed.append((pdb_code_chain, str(e)))
                print(f"{pdb_code_chain} failed: {e}")
        np.save(os.path.join(temp, f"chunk_{chunk_ix}"), chunk)
        output[chunk_ix] = failed

    def update_remaining(add=None):
        proteins = (
            np.load("done.npy", allow_pickle=True).item()
            if os.path.isfile("done.npy")
            else {}
        )
        for fn in os.listdir(temp):
            chunk = np.load(os.path.join(temp, fn), allow_pickle=True).item()
            for k, v in chunk.items():
                proteins[k] = v
        if add is not None:
            for k, v in add.items():
                proteins[k] = v
        np.save("done.npy", proteins)
        return proteins

    previous_left = -1
    while True:
        """
        - Keep saving numpy dicts to `temp` dir
        - Load them all up, combine into single numpy dict, save as single file, and
        remove temp dir
        - calculate the remaining files, run on them
        - if no more remaining file, break
        """
        proteins = update_remaining()
        os.system(f"rm -rf {temp}")
        os.makedirs(temp, exist_ok=True)

        def remaining():
            return list(set(code_hypn_chains) - set(proteins.keys()))

        print(
            f"TOTAL: {len(code_hypn_chains)}; REMAINING: {len(remaining())}; "
            f"DONE: {len(proteins)}"
        )

        # Create and start all of the processes.
        jobs = []
        ixs = []
        for ix, slice_pdb_code_chains in enumerate(
            np.array_split(remaining(), num_processes)
        ):
            ixs.append(ix)
            p = multiprocessing.Process(
                target=run_chunk, args=(slice_pdb_code_chains, ix)
            )
            jobs.append(p)
            p.start()

        # Sync up all processes.
        for job in jobs:
            job.join()

        # Extract the dictionary.
        failed = output.copy()
        failed_flat = []
        for k, v in failed.items():
            failed_flat += v
        code_hypn_chains = set(code_hypn_chains) - {a for a, _ in failed_flat}
        # print(failed_flat)
        with open(error_path, "w") as f:
            f.write(json.dumps(failed_flat))

        proteins = update_remaining(proteins)
        left = len(remaining())
        if left == 0 or previous_left == left:
            break
        previous_left = left

    proteins = update_remaining()

    identifiers = list(proteins.keys())
    with h5py.File(h5_path, "a", libver="latest") as hf:
        hf["identifiers"] = np.array(identifiers).astype(np.string_)

    # write to h5
    print("writing to h5")
    for identifier in tqdm(identifiers):
        add_structure_to_h5(h5_path, identifier, proteins[identifier])

    finalise_dataset(identifiers, h5_path, ec_annot_fn, go_annot_fn)


def update_dataset_based_on_any_files_that_couldnt_be_run():
    h5_path = "updated_full_dataset.h5"
    with h5py.File(h5_path, "a", libver="latest") as hf:
        identifiers = list(hf["identifiers"])
    identifiers = [i.decode("ascii") for i in identifiers]
    ec_annot_fn = "/app/DeepFRI/preprocessing/data/nrPDB-EC_2020.04_annot.tsv"
    go_annot_fn = "/app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv"
    finalise_dataset(identifiers, h5_path, ec_annot_fn, go_annot_fn, delete=True)


if __name__ == "__main__":
    h5_path = "updated_full_dataset.h5"
    error_path = "errors.json"
    pdb_directory = "/app/DeepFRI/preprocessing/all_cif_files"
    num_processes = 64
    create_h5_file(h5_path, error_path, pdb_directory, num_processes)
