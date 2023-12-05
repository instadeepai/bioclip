from functools import partial

import numpy as np

from bio_clip.data.alphafold_scripts.residue_constants import (
    atom_order,
    restype_1to3,
    restype_name_to_atom14_names,
    restype_order,
)
from bio_clip.data.protein_datastructure import ProteinStructureSample
from bio_clip.train.data_transforms import preprocess_atoms, preprocess_sample

RES_ATOM_MASK = np.zeros((len(restype_order), len(atom_order)), dtype=np.int32)
for r, i in restype_order.items():
    RES_ATOM_MASK[i][
        [
            atom_order[atom_name]
            for atom_name in restype_name_to_atom14_names[restype_1to3[r]]
            if atom_name in atom_order
        ]
    ] = 1
# X <==> len(restype_order)
RES_ATOM_MASK = np.concatenate(
    [RES_ATOM_MASK, np.prod(RES_ATOM_MASK, axis=0, keepdims=True)], axis=0
)


def filter_out_sample(
    sample: ProteinStructureSample,
    min_number_valid_residues: int,
    max_number_residues: int,
) -> bool:
    missing_coords_residue_mask = sample.get_missing_backbone_coords_mask()
    num_residue_coords_known = np.sum(~missing_coords_residue_mask)
    too_small = num_residue_coords_known < min_number_valid_residues
    too_big = sample.nb_residues > max_number_residues
    print(f"too_big: {too_big}; too_small: {too_small}")
    return bool(too_small or too_big)


def alphafold_to_legacy_bioclip(
    identifier,
    aatype,
    residue_index,
    atom37_positions,
    atom37_gt_exists,
    bioclip_preprocess_fn,
    filter_sample,
    verbose,
    chain_index=None,
):
    # create atom37_atom_exists from aatype
    atom37_atom_exists = RES_ATOM_MASK[aatype]

    if chain_index is not None:
        # make increasing residue indices, as it resets for each chain
        unique = np.unique(chain_index)
        chain_mask = unique[:, None] == chain_index[None, :]
        order = np.argsort(np.argmax(chain_mask, axis=1))
        chain_mask = chain_mask[order]
        chain_ixs = chain_mask * residue_index[None, :]
        start_ixs = (chain_ixs + (1 - chain_mask) * 100000).min(-1)
        chain_ixs = chain_ixs - start_ixs[:, None]
        chain_lengths = chain_ixs.max(-1) + 1
        chain_lengths_cumulative = np.concatenate(
            [np.array([0]), np.cumsum(chain_lengths)[:-1]]
        )
        shift = (chain_mask * (chain_lengths_cumulative - start_ixs)[:, None]).sum(0)
        _residue_index = residue_index + shift

        def pr(x):
            return ",".join([str(y) for y in x])

        assert ((_residue_index[1:] - _residue_index[:-1]) > 0).all(), (
            f"non-increasing residue index: {pr(_residue_index)};\nresidue_index = "
            f"[{pr(residue_index)}];\nchain_index = [{pr(chain_index)}]"
        )
        residue_index = _residue_index
    num_res = int(residue_index.max() + 1)

    # re-index everything according to `residue_index`; this is important for ESM, we
    #  would like contiguous residues, i.e. no gaps.
    #  this is relevant if an edge feature depended on the relative sequence position
    #  or a node feature depended on the absolute sequence position.
    def reindex(arr_leading_residue_dim):
        data = np.zeros(
            (num_res,) + arr_leading_residue_dim.shape[1:],
            dtype=arr_leading_residue_dim.dtype,
        )
        data[residue_index] = arr_leading_residue_dim
        return data

    # these aren't processed...
    resolution = np.zeros(1, dtype=np.float32)
    pdb_cluster_size = np.ones(1, dtype=np.int32)

    aatype_onehot = np.zeros((aatype.shape[0], RES_ATOM_MASK.shape[0]), dtype=bool)
    for i, aai in enumerate(aatype):
        aatype_onehot[i, aai] = 1
    aatype_onehot = reindex(aatype_onehot)
    expanded = np.where(aatype_onehot.sum(axis=-1) == 0)[0]
    aatype_onehot[expanded, -1] = 1  # unknown
    atom37_exists = reindex(atom37_atom_exists)
    atom37_exists[expanded, :] = RES_ATOM_MASK[-np.ones_like(expanded)]

    # legacy datastructure...
    pss = ProteinStructureSample(
        chain_id=identifier,
        nb_residues=num_res,
        aatype=aatype_onehot,
        atom37_positions=reindex(atom37_positions),
        atom37_gt_exists=reindex(atom37_gt_exists.astype(bool)),
        atom37_atom_exists=atom37_exists.astype(bool),
        resolution=resolution,
        pdb_cluster_size=pdb_cluster_size,
    )
    # apply the filtering function here...
    if filter_sample is not None and filter_sample(pss):
        if verbose:
            print(f"Filtered: {identifier}")
        return None
    return bioclip_preprocess_fn(pss)


def get_processing_function(cfg, tokenizer, maximum_padding, graph_max_neighbor):
    if cfg.model.architecture == "graph_transformer":
        bioclip_legacy_save_to_bioclip_training_data = partial(
            preprocess_atoms,
            tokenizer=tokenizer,
            num_neighbor=graph_max_neighbor,
            padding_num_residue=maximum_padding,
        )
    else:
        bioclip_legacy_save_to_bioclip_training_data = partial(
            preprocess_sample,
            tokenizer=tokenizer,
            num_neighbor=graph_max_neighbor,
            residue_loc_is_alphac=(
                cfg.training.data.datatransforms.graph_residue_loc_is_alphac
            ),
            padding_num_residue=maximum_padding,
        )
    return bioclip_legacy_save_to_bioclip_training_data
