import functools
import json
import os

import h5py
import jax
import numpy as np
import torch as pt
from tree import map_structure

from bio_clip.data.downstream.pesto_src.data_encoding import names_enum, resnames_enum
from bio_clip.data.downstream.pesto_src.data_handler import (
    load_interface_labels,
    load_sparse_mask,
)
from bio_clip.data.downstream.pesto_src.dataset import (
    select_by_interface_types,
    select_by_max_ba,
    select_by_sid,
)
from bio_clip.data.protein_datastructure import ProteinStructureSample
from bio_clip.data.residue_constants import ATOM_ORDER, RESTYPE_3TO1, RESTYPES_ORDER
from bio_clip.model.esm.esm_haiku import (
    esm,
    get_config_and_tokenizer,
    get_model_hyperparameters,
)
from bio_clip.train.data_transforms import preprocess_atoms, preprocess_sample
from bio_clip.utils.checkpointing import prepare_ft_pretraining_cfgs

# flattened from PeSTo/md_analysis/save/i_v4_1_2021-09-07_11-21
categ_to_resnames = {
    "protein": [
        "GLU",
        "LEU",
        "ALA",
        "ASP",
        "SER",
        "VAL",
        "GLY",
        "THR",
        "ARG",
        "PHE",
        "TYR",
        "ILE",
        "PRO",
        "ASN",
        "LYS",
        "GLN",
        "HIS",
        "TRP",
        "MET",
        "CYS",
    ],
    "rna": ["A", "U", "G", "C"],
    "dna": ["DA", "DT", "DG", "DC"],
    "ion": [
        "MG",
        "ZN",
        "CL",
        "CA",
        "NA",
        "MN",
        "K",
        "IOD",
        "CD",
        "CU",
        "FE",
        "NI",
        "SR",
        "BR",
        "CO",
        "HG",
    ],
    "ligand": [
        "SO4",
        "NAG",
        "PO4",
        "EDO",
        "ACT",
        "MAN",
        "HEM",
        "FMT",
        "BMA",
        "ADP",
        "FAD",
        "NAD",
        "NO3",
        "GLC",
        "ATP",
        "NAP",
        "BGC",
        "GDP",
        "FUC",
        "FES",
        "FMN",
        "GAL",
        "GTP",
        "PLP",
        "MLI",
        "ANP",
        "H4B",
        "AMP",
        "NDP",
        "SAH",
        "OXY",
    ],
    "lipid": ["PLM", "CLR", "CDL", "RET"],
}
config_data = {
    "dataset_filepath": "datasets/contacts_rr5A_64nn_8192.h5",
    "masif_filepath": "datasets/masif_codes.txt",
    "train_selection_filepath": "datasets/subunits_train_set.txt",
    "test_selection_filepath": "datasets/subunits_test_set.txt",
    "max_ba": 1,
    "max_size": 1024 * 8,
    "min_num_res": 48,
    "l_types": categ_to_resnames["protein"],
    "r_types": [
        categ_to_resnames["protein"],
        categ_to_resnames["dna"] + categ_to_resnames["rna"],
        categ_to_resnames["ion"],
        categ_to_resnames["ligand"],
        categ_to_resnames["lipid"],
    ],
}


class Dataset(pt.utils.data.Dataset):  # IterableDataset
    def __init__(
        self, dataset_filepath, features_flags=(True, True, True), post_processing=None
    ):
        super(Dataset, self).__init__()
        # store dataset filepath
        self.dataset_filepath = dataset_filepath

        # selected features
        self.ftrs = [fn for fn, ff in zip(["qe", "qr", "qn"], features_flags) if ff]

        # preload data
        with h5py.File(dataset_filepath, "r") as hf:
            # load keys, sizes and types
            self.keys = np.array(hf["metadata/keys"]).astype(np.dtype("U"))
            self.sizes = np.array(hf["metadata/sizes"])
            self.ckeys = np.array(hf["metadata/ckeys"]).astype(np.dtype("U"))
            self.ctypes = np.array(hf["metadata/ctypes"])

            # load parameters to reconstruct data
            self.std_elements = np.array(hf["metadata/std_elements"]).astype(
                np.dtype("U")
            )
            self.std_resnames = np.array(hf["metadata/std_resnames"]).astype(
                np.dtype("U")
            )
            self.std_names = np.array(hf["metadata/std_names"]).astype(np.dtype("U"))
            self.mids = np.array(hf["metadata/mids"]).astype(np.dtype("U"))

        # set default selection mask
        self.m = np.ones(len(self.keys), dtype=bool)

        # prepare ckeys mapping
        self.__update_selection()

        # set default runtime selected interface types
        self.t0 = pt.arange(self.mids.shape[0])
        self.t1_l = [pt.arange(self.mids.shape[0])]
        self.post_processing = (
            (lambda x: x) if post_processing is None else post_processing
        )

    def __update_selection(self):
        # ckeys mapping with keys
        self.ckeys_map = {}
        for key, ckey in zip(self.keys[self.m], self.ckeys[self.m]):
            if key in self.ckeys_map:
                self.ckeys_map[key].append(ckey)
            else:
                self.ckeys_map[key] = [ckey]

        # keep unique keys
        self.ukeys = list(self.ckeys_map)

    def update_mask(self, m):
        # update mask
        self.m &= m

        # update ckeys mapping
        self.__update_selection()

    def set_types(self, l_types, r_types_l):
        self.t0 = pt.from_numpy(np.where(np.isin(self.mids, l_types))[0])
        self.t1_l = [
            pt.from_numpy(np.where(np.isin(self.mids, r_types))[0])
            for r_types in r_types_l
        ]

    def get_largest(self):
        i = np.argmax(self.sizes[:, 0] * self.m.astype(int))
        k = np.where(np.isin(self.ukeys, self.keys[i]))[0][0]
        return self[k]

    def __len__(self):
        return len(self.ukeys)

    def _data_transform(self, k):
        try:
            # get corresponding interface keys
            key = self.ukeys[k]
            ckeys = self.ckeys_map[key]

            # load data
            with h5py.File(self.dataset_filepath, "r") as hf:
                # hdf5 group
                hgrp = hf["data/structures/" + key]

                # topology
                X = pt.from_numpy(np.array(hgrp["X"]).astype(np.float32))
                M = load_sparse_mask(hgrp, "M")
                ids_topk = pt.from_numpy(np.array(hgrp["ids_topk"]).astype(np.int64))

                # features
                q_l = []
                for fn in self.ftrs:
                    q_l.append(load_sparse_mask(hgrp, fn))
                q = pt.cat(q_l, dim=1)

                # interface labels
                y = pt.zeros((M.shape[1], len(self.t1_l)), dtype=pt.bool)
                for ckey in ckeys:
                    y |= load_interface_labels(
                        hf["data/contacts/" + ckey], self.t0, self.t1_l
                    )

            out = X, ids_topk, q, M, y.float()
            training_example = self.post_processing(out)
        except Exception as e:
            print(e)
            training_example = None
        return training_example

    def __getitem__(self, k):
        out = self._data_transform(k)
        if out is None:
            while out is None:
                k = np.random.randint(len(self))
                out = self._data_transform(k)
        return out


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [np.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def setup_dataloader_modified(
    config_data,
    sids_selection_filepath,
    batch_size,
    post_processing,
    shuffle=True,
    fp=None,
    max_assembly=True,
):
    # load selected sids
    sids_sel = np.genfromtxt(sids_selection_filepath, dtype=np.dtype("U"))

    # create dataset
    dataset = Dataset(
        config_data["dataset_filepath"] if fp is None else fp,
        features_flags=(False, True, True),
        post_processing=post_processing,
    )

    # data selection criteria
    m = select_by_sid(dataset, sids_sel)  # select by sids
    # select by max assembly count
    if max_assembly:
        m &= select_by_max_ba(dataset, config_data["max_ba"])
    m &= dataset.sizes[:, 0] <= config_data["max_size"]  # select by max size
    # select by min size
    m &= dataset.sizes[:, 1] >= config_data["min_num_res"]
    m &= select_by_interface_types(
        dataset, config_data["l_types"], np.concatenate(config_data["r_types"])
    )  # select by interface type

    # update dataset selection
    dataset.update_mask(m)

    # set dataset types for labels
    dataset.set_types(config_data["l_types"], config_data["r_types"])

    # define data loader
    dataloader = pt.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        collate_fn=tree_stack,
        pin_memory=True,
        prefetch_factor=2,
    )
    if fp is None:
        return dataloader
    else:
        return dataloader, m


# create conversion map from PeSTo residue encoding to bioclip res ixs
def restype_conversion(pesto_index):
    three_letter_code = resnames_enum[pesto_index]
    one_letter_code = RESTYPE_3TO1.get(three_letter_code, None)
    return RESTYPES_ORDER.get(one_letter_code, len(RESTYPES_ORDER))


PESTO_RES_TO_BIOCLIP_RES = np.empty(len(resnames_enum), dtype=np.int32)
for i in range(len(resnames_enum)):
    PESTO_RES_TO_BIOCLIP_RES[i] = restype_conversion(i)


# create conversion map from PeSTo atom-type index to atom-37 atom index
def atomtype_conversion(pesto_index):
    pdb_atom_code = names_enum[pesto_index]
    return ATOM_ORDER.get(pdb_atom_code, len(ATOM_ORDER))


PESTO_ATOM_TO_AATOM37 = np.empty(len(names_enum), dtype=np.int32)
for i in range(len(names_enum)):
    PESTO_ATOM_TO_AATOM37[i] = atomtype_conversion(i)


# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
residue_atoms = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": [
        "C",
        "CA",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "N",
        "NE1",
        "O",
    ],
    "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
}


def one_hot(n, max_num):
    z = np.zeros((max_num,), dtype=np.int32)
    z[n] = 1
    return z


residue_to_atom37 = {
    r: sum(one_hot(ATOM_ORDER[a], len(ATOM_ORDER)) for a in v)
    for r, v in residue_atoms.items()
}


def reconstruct_protein_structure_sample(code, X, q, M):
    # we only have [qr, qn]
    # len(config_encoding['std_resnames'])  # 'std_names'
    len_qr = len(resnames_enum)
    qr, qn = np.argmax(q[:, :len_qr], axis=-1), np.argmax(q[:, len_qr:], axis=-1)

    # create atom37 representation
    num_res = M.shape[1]
    resix = np.argmax(M, axis=-1)
    atom37 = np.zeros((num_res, 37, 3), dtype=np.float32)
    atom37_gt_exists = np.zeros((num_res, 37), dtype=bool)
    for r, x, a in zip(resix, X, qn):
        atom_ix = PESTO_ATOM_TO_AATOM37[a]
        if atom_ix < 37:
            atom37[r, atom_ix, :] = x
            atom37_gt_exists[r, atom_ix] = 1

    pesto_res_ixs = qr[np.argmax(M, axis=0)]
    res_ix = PESTO_RES_TO_BIOCLIP_RES[pesto_res_ixs]
    atom37_mask = np.zeros((num_res, 37), dtype=bool)
    for i, pesto_resix in enumerate(pesto_res_ixs):
        three_letter_code = resnames_enum[pesto_resix]
        atom37_mask[i, :] = residue_to_atom37[three_letter_code]

    aatype = np.zeros((num_res, 21), dtype=bool)
    for i, r in enumerate(res_ix):
        aatype[i, r] = True

    return ProteinStructureSample(
        chain_id=code.split("_")[1] if code is not None else None,
        nb_residues=num_res,
        aatype=aatype,
        atom37_positions=atom37,
        atom37_gt_exists=atom37_gt_exists,
        atom37_atom_exists=atom37_mask,
        resolution=0,
        pdb_cluster_size=1,
    )


def extract_bioclip_features(
    pesto_feat,
    code,
    tokenizer,
    num_neighbor,
    residue_loc_is_alphac,
    padding_num_residue,
    tokenizer_discards_missing_residues,
    architecture="",
):
    """_summary_

    Args:
        pesto_feat (_type_): _description_

        X, ids_topk, q, M, y = pesto_feat
        X == atom positions
        ids_topk == neighbouring residue indices for each residue
        q ==
        M == one_hot([[atom_index, residue_index],...])
        y == [num_res, 5] confusing, paper implies 79 categories, but from inspecting
        their code they do BCE on these five categories.

    Returns:
        _type_: _description_
    """
    X, _ids_topk, q, M, y = map_structure(np.array, pesto_feat)
    target = y

    pss = reconstruct_protein_structure_sample(code, X, q, M)

    if architecture == "graph_transformer":
        features = preprocess_atoms(pss, tokenizer, num_neighbor, padding_num_residue)
    else:
        features = preprocess_sample(
            pss,
            tokenizer=tokenizer,
            num_neighbor=num_neighbor,
            residue_loc_is_alphac=residue_loc_is_alphac,
            padding_num_residue=padding_num_residue,
            tokenizer_discards_missing_residues=tokenizer_discards_missing_residues,
        )

    num_res = target.shape[0]
    target = np.pad(
        target[:padding_num_residue],
        ((0, padding_num_residue - num_res), *((0, 0) for _ in target.shape[1:])),
    )

    return features, target


def prepare_resppi_dataloaders(cfg, num_devices):
    clip_cfg, cfg, devices = prepare_ft_pretraining_cfgs(cfg)
    number_of_transformer_blocks = int(
        (cfg.model.plm.esm_model_name).split("_")[1].replace("t", "")
    )
    embeddings_layer_to_save = int(
        clip_cfg.training.proportion_esm_layer * number_of_transformer_blocks
    )

    model_name = cfg.model.plm.esm_model_name
    model, alphabet = getattr(esm.pretrained, model_name)()
    hyperparam_dict = get_model_hyperparameters(model, model_name, alphabet)
    config, tokenizer = get_config_and_tokenizer(
        hyperparam_dict,
        model_name,
        [embeddings_layer_to_save],
        attention_maps_to_save=(),
        max_positions=1024,
    )
    fixes_sizes = clip_cfg.training.data.fixed_sizes

    extract_bioclip_features_fn = functools.partial(
        extract_bioclip_features,
        tokenizer=tokenizer,
        num_neighbor=fixes_sizes.graph_max_neighbor,
        residue_loc_is_alphac=(
            clip_cfg.training.data.datatransforms.graph_residue_loc_is_alphac
        ),
        padding_num_residue=fixes_sizes.maximum_padding,
        tokenizer_discards_missing_residues=True,
        code=None,
        architecture=clip_cfg.model.architecture,
    )
    p = "/app/bio-clip/datasets/downstream/per_residue_ppi/sample/"
    local_baselines = f"{p}masif_benchmark_ppi.json"
    config_data["masif_filepath"] = f"{p}masif_codes.txt"
    use_example_data = True
    if use_example_data:
        config_data["dataset_filepath"] = f"{p}selection_contacts_rr5A_64nn_8192.h5"
        config_data["train_selection_filepath"] = f"{p}selection_subunits_train_set.txt"
        config_data["test_selection_filepath"] = f"{p}selection_subunits_test_set.txt"
    else:
        aws_base = cfg.training.data.aws_base
        config_data["dataset_filepath"] = os.path.join(
            "/tmp/ramdisk", config_data["dataset_filepath"]
        )
        local_baselines = os.path.join("datasets", cfg.training.data.masif_baselines)
        paths = [
            (
                os.path.join(aws_base, cfg.training.data.raw_data),
                config_data["dataset_filepath"],
            ),
            (
                os.path.join(aws_base, cfg.training.data.train_examples),
                config_data["train_selection_filepath"],
            ),
            (
                os.path.join(aws_base, cfg.training.data.test_examples),
                config_data["test_selection_filepath"],
            ),
            (
                os.path.join(aws_base, cfg.training.data.masif_baselines),
                local_baselines,
            ),
        ]

        # download three files: the two text files for train and test examples, the hdf5
        # file with the raw data
        for remote, local_path in paths:
            os.makedirs(os.path.split(local_path)[0], exist_ok=True)
            if not os.path.isfile(local_path):
                os.system(
                    f"aws s3 cp {remote} {local_path} --endpoint "
                    f"{cfg.training.data.endpoint}"
                )

    with open(local_baselines) as f:
        baselines_dict = json.loads(f.read())
        baselines_dict = {
            "pr_aucs_bench": tree_stack(baselines_dict["pr_aucs_bench"]),
            "all_metrics_masif": tree_stack(baselines_dict["all_metrics_masif"]),
        }
        _nanmean = (
            lambda x: np.mean(x[~np.isnan(x)])
            if x.dtype != np.array(["3H6G_B"]).dtype
            else x
        )
        _nanmedian = (
            lambda x: np.median(x[~np.isnan(x)])
            if x.dtype != np.array(["3H6G_B"]).dtype
            else x
        )
        mean_baselines_dict = jax.tree_map(_nanmean, baselines_dict)
        median_baselines_dict = jax.tree_map(_nanmedian, baselines_dict)
        del mean_baselines_dict["all_metrics_masif"]["sid"]
        del mean_baselines_dict["pr_aucs_bench"]["sid"]
        del median_baselines_dict["all_metrics_masif"]["sid"]
        del median_baselines_dict["pr_aucs_bench"]["sid"]
        baselines_dict = {"mean": mean_baselines_dict, "median": median_baselines_dict}

    # Train / Test
    bs_inf = cfg.training.batch.num_per_device_inference * num_devices
    bs_upd = cfg.training.batch.num_per_device_update * num_devices
    dataloader_train = setup_dataloader_modified(
        config_data,
        config_data["train_selection_filepath"],
        bs_upd,
        post_processing=extract_bioclip_features_fn,
    )
    dataloader_test = setup_dataloader_modified(
        config_data,
        config_data["test_selection_filepath"],
        bs_inf,
        post_processing=extract_bioclip_features_fn,
    )
    dataloader_masif = setup_dataloader_modified(
        config_data,
        config_data["masif_filepath"],
        bs_inf,
        post_processing=extract_bioclip_features_fn,
    )

    dataloaders = {
        "train": dataloader_train,
        "test": dataloader_test,
        "masif": dataloader_masif,
    }
    return dataloaders, baselines_dict, clip_cfg
