import functools
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.spatial as spa
import tree
from scipy.special import softmax

from bio_clip.data.protein_datastructure import (
    ProteinStructureSample,
    onehot_to_sequence,
)
from bio_clip.data.residue_constants import (
    ATOM37_IX_TO_PESTO_ELEM_IX,
    CA_INDEX,
    RESTYPE_1TO3,
)
from bio_clip.types import (
    BatchDataWithTokensBioClip,
    Coordinates,
    EdgeFeatures,
    NodeFeatures,
    ProteinGraph,
    Residue,
    RotationMatrix,
    TranslationVector,
)


def residue_embedding(residue: Residue) -> int:
    """
    Returns residue index (between 0 and 20)

    Args:
        residue (Residue): residue sequence

    Returns:
        index (int)
    """

    dit = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "HIP": "H",
        "HIE": "H",
        "TPO": "T",
        "HID": "H",
        "LEV": "L",
        "MEU": "M",
        "PTR": "Y",
        "GLV": "E",
        "CYT": "C",
        "SEP": "S",
        "HIZ": "H",
        "CYM": "C",
        "GLM": "E",
        "ASQ": "D",
        "TYS": "Y",
        "CYX": "C",
        "GLZ": "G",
    }

    rare_residues = {
        "HIP": "H",
        "HIE": "H",
        "TPO": "T",
        "HID": "H",
        "LEV": "L",
        "MEU": "M",
        "PTR": "Y",
        "GLV": "E",
        "CYT": "C",
        "SEP": "S",
        "HIZ": "H",
        "CYM": "C",
        "GLM": "E",
        "ASQ": "D",
        "TYS": "Y",
        "CYX": "C",
        "GLZ": "G",
    }

    if residue in rare_residues.keys():
        print("Some rare residue: ", residue)

    indicator = {
        "Y": 0,
        "R": 1,
        "F": 2,
        "G": 3,
        "I": 4,
        "V": 5,
        "A": 6,
        "W": 7,
        "E": 8,
        "H": 9,
        "C": 10,
        "N": 11,
        "M": 12,
        "D": 13,
        "T": 14,
        "S": 15,
        "K": 16,
        "L": 17,
        "Q": 18,
        "P": 19,
    }
    res_name = residue
    if res_name not in dit.keys():
        return 20
    else:
        res_name = dit[res_name]
        return indicator[res_name]


def residue_list_featurizer(residue_list: List[str]) -> np.ndarray:
    """
    Creates residual features given PDB data as input.
    A residue is simply embedded by an index.

    Args:
        residue_list (List[str]): list of 3-letter amino-acid codes

    Returns:
        Residual features
    """
    feature_list = [[residue_embedding(residue)] for residue in residue_list]
    return np.array(feature_list).astype(np.int32)


def distance_list_featurizer(dist_list: List[float]) -> np.ndarray:
    """Computes graph features based on the distance between residues

    Args:
        dist_list (List[float]): list of distances between residues

    Returns:
        np.ndarray: distance features
    """
    length_scale_list = [1.5**x for x in range(15)]
    center_list = [0.0 for _ in range(15)]

    num_edge = len(dist_list)
    dist_list = np.array(dist_list)

    transformed_dist_list = [
        np.exp(-((dist_list - center) ** 2) / float(length_scale))
        for length_scale, center in zip(length_scale_list, center_list)
    ]

    transformed_dist = np.array(transformed_dist_list).T.reshape((num_edge, -1))

    return transformed_dist.astype(np.float32)


def compute_nearest_neighbors_graph(
    protein_num_residues: int,
    list_atom_coordinates: List[Coordinates],
    residue_list: List[str],
    stacked_residue_coordinates: Coordinates,
    protein_n_i_feat: np.ndarray,
    protein_u_i_feat: np.ndarray,
    protein_v_i_feat: np.ndarray,
    num_neighbor: int,
) -> Tuple[
    int,
    int,
    NodeFeatures,
    Coordinates,
    NodeFeatures,
    EdgeFeatures,
    jnp.ndarray,
    jnp.ndarray,
]:

    """Computes kNN graph based on residues coordinates

    Args:
        protein_num_residues (int): number of residues
        list_atom_coordinates (List[Coordinates]):
            atom coordinates in residues
        residue_list (List[str]): list of 3-letter amino-acid codes
        stacked_residue_coordinates (Coordinates):
            residue coordinates
        protein_n_i_feat (np.ndarray): residues features
        protein_u_i_feat (np.ndarray): residues features
        protein_v_i_feat (np.ndarray): residues features
        num_neighbor (int): maximum number of nearest neighbors

    Returns:
        n_node (int): number of nodes
        n_edge (int): number of edges
        nodes_res_feat (NodeFeatures): residual features
        nodes_x (Coordinates): nodes coordinates
        nodes_surface_aware_features (NodeFeatures): node features
        edges_features (EdgeFeatures): edge features
        senders (jnp.ndarray): indexes of edge senders
        receivers (jnp.ndarray): indexes of edges receivers
    """

    assert protein_num_residues == stacked_residue_coordinates.shape[0]
    assert stacked_residue_coordinates.shape[1] == 3

    means_of_atom_coordinates = np.stack(
        [np.mean(res_coordinates, axis=0) for res_coordinates in list_atom_coordinates]
    )

    # scipy is MUCH faster that np.linalg.norm with broadcasting, it uses a double
    # python loop, but very fast metric code.
    protein_distance = spa.distance.cdist(
        means_of_atom_coordinates, means_of_atom_coordinates
    )

    n_node = protein_num_residues
    n_edge = num_neighbor * protein_num_residues

    sigma = np.array([1.0, 2.0, 5.0, 10.0, 30.0])[..., np.newaxis]
    valid_src = np.argsort(protein_distance, axis=-1)[:, 1 : (num_neighbor + 1)]
    valid_dst = np.repeat(
        np.arange(protein_num_residues)[..., np.newaxis], num_neighbor, axis=-1
    )
    valid_dist = np.stack(
        [protein_distance[i, valid_src[i]] for i in range(protein_num_residues)]
    )
    weights = softmax(-valid_dist[:, np.newaxis, :] ** 2 / sigma, axis=-1)
    diff_vecs = np.stack(
        [
            stacked_residue_coordinates[valid_dst[i]]
            - stacked_residue_coordinates[valid_src[i]]
            for i in range(protein_num_residues)
        ]
    )
    mean_vec = np.einsum("nsi,nik->nsk", weights, diff_vecs)
    denominator = np.einsum("nsi,ni->ns", weights, np.linalg.norm(diff_vecs, axis=-1))
    mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=-1) / denominator

    senders = valid_src.flatten()
    receivers = valid_dst.flatten()
    protein_dist_list = list(valid_dist.flatten())

    nodes_res_feat = residue_list_featurizer(residue_list)

    edges_features = distance_list_featurizer(protein_dist_list)

    # Build the various p_ij, q_ij, k_ij, t_ij pairs
    basis_matrices = np.stack(
        [protein_n_i_feat, protein_u_i_feat, protein_v_i_feat], axis=1
    )
    stacked_res = stacked_residue_coordinates
    diff_stacked_res = stacked_res[:, np.newaxis, :] - stacked_res[np.newaxis, :, :]
    p_ij = np.einsum("ijk,nik->inj", basis_matrices, diff_stacked_res)
    q_ij = np.einsum("ijk,nk->inj", basis_matrices, protein_n_i_feat)
    k_ij = np.einsum("ijk,nk->inj", basis_matrices, protein_u_i_feat)
    t_ij = np.einsum("ijk,nk->inj", basis_matrices, protein_v_i_feat)
    s_ij = np.concatenate([p_ij, q_ij, k_ij, t_ij], axis=-1)

    protein_edge_feat_ori_list = [
        s_ij[receivers[i], senders[i]] for i in range(len(protein_dist_list))
    ]

    protein_edge_feat_ori_feat = np.stack(
        protein_edge_feat_ori_list, axis=0
    )  # shape (num_edges, 4, 3)

    edges_features = np.concatenate(
        [edges_features, protein_edge_feat_ori_feat], axis=1
    )

    nodes_x = stacked_residue_coordinates

    nodes_surface_aware_features = mean_vec_ratio_norm

    return (
        n_node,
        n_edge,
        nodes_res_feat,
        nodes_x,
        nodes_surface_aware_features,
        edges_features,
        senders,
        receivers,
    )


def rigid_transform_kabsch_3d(
    a: Coordinates, b: Coordinates
) -> Tuple[RotationMatrix, TranslationVector]:
    """Applies Kabsch algorithm: it find the right rotation/translation to move
        a point cloud a_1...N to another point cloud b_1...N`

    Args:
        a (Coordinates): 3D point cloud
        b (Coordinates): 3D point cloud

    Raises:
        Exception: if data point cloud a has wrong size
        Exception: if data point cloud b has wrong size

    Returns:
        r: rotation matrix
        t: translation vector
    """
    # find mean column wise: 3 x 1

    centroid_a = jax.device_put(
        jnp.mean(a, axis=1, keepdims=True), device=jax.devices("cpu")[0]
    )
    centroid_b = jax.device_put(
        jnp.mean(b, axis=1, keepdims=True), device=jax.devices("cpu")[0]
    )

    # subtract mean
    am = a - centroid_a
    bm = b - centroid_b

    h = am @ bm.T

    # find rotation
    u, s, vt = jnp.linalg.svd(h)

    r = vt.T @ u.T

    # special reflection case
    if jnp.linalg.det(r) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        ss = jnp.diag(np.array([1.0, 1.0, -1.0]))
        r = (vt.T @ ss) @ u.T

    t = -r @ centroid_a + centroid_b
    return r, t


def protein_align_unbound_and_bound(
    stacked_residue_representatives_coordinates: Coordinates,
    protein_n_i_feat: jnp.ndarray,
    protein_u_i_feat: jnp.ndarray,
    protein_v_i_feat: jnp.ndarray,
    alphac_atom_coordinates: Coordinates,
) -> Tuple[Coordinates, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Aligns the bound and unbound structures.
    In the bound structure, the residue coordinate
    is defined by the centroid coordinates of the atoms of. In the unbound structure,
    the residue coordinates are defined by the alpha-c coordinates.

    Args:
        stacked_residue_representatives_coordinates (Coordinates): coordinates of
            atoms centroids of residues
        protein_n_i_feat (jnp.ndarray): protein features
        protein_u_i_feat (jnp.ndarray): protein features
        protein_v_i_feat (jnp.ndarray): protein features
        alphac_atom_coordinates (Coordinates): coordinates of alpha-c atoms
    """

    ret_r_protein, ret_t_protein = rigid_transform_kabsch_3d(
        stacked_residue_representatives_coordinates.T,
        alphac_atom_coordinates.T,
    )
    list_residue_representatives_coordinates = (
        (ret_r_protein @ (stacked_residue_representatives_coordinates).T)
        + ret_t_protein
    ).T
    protein_n_i_feat = (ret_r_protein @ (protein_n_i_feat).T).T
    protein_u_i_feat = (ret_r_protein @ (protein_u_i_feat).T).T
    protein_v_i_feat = (ret_r_protein @ (protein_v_i_feat).T).T
    return (
        list_residue_representatives_coordinates,
        protein_n_i_feat,
        protein_u_i_feat,
        protein_v_i_feat,
    )


def onehot(x, n):
    m = x[:, None] == np.arange(n)[None, :]
    return np.concatenate([m, ~np.any(m, axis=1).reshape(-1, 1)], axis=1)


def get_nn(x, num_nn):
    # scipy is MUCH faster that np.linalg.norm with broadcasting, it uses a double
    # python loop, but very fast metric code.
    d = spa.distance.cdist(x, x)

    # find nearest neighbors
    knn = min(num_nn, d.shape[0])
    ids_topk = np.argsort(d, axis=-1)[:, 1 : (knn + 1)]
    return ids_topk


def create_pesto_features(atom37_coords, atom37_mask, num_neighbor):
    num_res = atom37_coords.shape[0]
    atom14_unordered = np.zeros((num_res, 14, 3))
    atom14_unordered_mask = np.zeros((num_res, 14), dtype=bool)
    q_14 = np.zeros((num_res, 14, ATOM37_IX_TO_PESTO_ELEM_IX.max() + 1), dtype=bool)
    for i, (row37, r37_mask) in enumerate(zip(atom37_coords, atom37_mask)):
        (ixs,) = np.where(r37_mask)
        if len(ixs):
            atom14_unordered_mask[i, : len(ixs)] = 1
            atom14_unordered[i, : len(ixs)] = row37[ixs]
            q_14[i, : len(ixs)] = onehot(
                ATOM37_IX_TO_PESTO_ELEM_IX[ixs], ATOM37_IX_TO_PESTO_ELEM_IX.max()
            )

    flat_atom14_unordered = atom14_unordered.reshape(-1, 3)
    flat_atom14_unordered_mask = atom14_unordered_mask.reshape(-1)

    ids_topk_ = np.zeros(
        (flat_atom14_unordered_mask.shape[0], num_neighbor), dtype=np.int64
    )
    indices = get_nn(flat_atom14_unordered[flat_atom14_unordered_mask], num_neighbor)
    # shift the indices of the neighbours to account for the masked values
    (shift_indices,) = np.where(flat_atom14_unordered_mask)
    ids_topk_[flat_atom14_unordered_mask] = shift_indices[indices]

    q_flat = q_14.reshape(-1, q_14.shape[-1])
    atom14_mask = flat_atom14_unordered_mask.reshape(num_res, 14)
    return flat_atom14_unordered, ids_topk_, q_flat, atom14_mask


def preprocess_atoms(
    sample: ProteinStructureSample,
    tokenizer,
    num_neighbor: int,
    padding_num_residue: int,
) -> BatchDataWithTokensBioClip:
    sequence = onehot_to_sequence(sample.aatype)
    tokens = np.array(tokenizer.batch_tokenize([sequence])[0][1], dtype=np.int32)
    (flat_atom14, ids_topk_, q_flat, atom14_mask) = create_pesto_features(
        sample.atom37_positions,
        sample.atom37_gt_exists & sample.atom37_atom_exists,
        num_neighbor=num_neighbor,
    )
    n_node = atom14_mask.shape[0]

    def crop_pad(data, multiply_by):
        take = padding_num_residue * multiply_by
        pad = (padding_num_residue - n_node) * multiply_by
        return jax.tree_map(
            (
                lambda x: np.pad(
                    x[:take],
                    ((0, pad), *((0, 0) for _ in x.shape[1:])),
                )
            ),
            data,
        )

    # the following datastructures have leading dim num_res * 14
    (flat_atom14, ids_topk_, q_flat) = tree.map_structure(
        functools.partial(crop_pad, multiply_by=14), (flat_atom14, ids_topk_, q_flat)
    )
    # the following datastructures have leading dim num_res
    (atom14_mask,) = tree.map_structure(
        functools.partial(crop_pad, multiply_by=1), (atom14_mask,)
    )
    graph = (
        flat_atom14,
        ids_topk_,
        q_flat.astype(np.float32),
        atom14_mask.astype(np.float32),
    )

    # I am cropping tokens manually here, because the filter guarentees the num residues
    # is below padding_num_residue
    tokens = tokens[:padding_num_residue]

    return BatchDataWithTokensBioClip(tokens=tokens, graph=graph)


def preprocess_sample(
    sample: ProteinStructureSample,
    tokenizer,
    num_neighbor: int,
    residue_loc_is_alphac: bool,
    padding_num_residue: int,
    tokenizer_discards_missing_residues: bool = False,
) -> BatchDataWithTokensBioClip:
    sequence = onehot_to_sequence(sample.aatype)
    atom37_coords = sample.atom37_positions
    atom37_mask = sample.atom37_gt_exists & sample.atom37_atom_exists
    missing_coords_residue_mask = sample.get_missing_backbone_coords_mask()

    sequence_wo_missing_residue = "".join(
        [
            res_type
            for res_type, are_coords_missing in zip(
                sequence, missing_coords_residue_mask
            )
            if not are_coords_missing
        ]
    )

    tokens = np.array(
        tokenizer.batch_tokenize(
            [
                sequence_wo_missing_residue
                if tokenizer_discards_missing_residues
                else sequence
            ]
        )[0][1],
        dtype=np.int32,
    )

    (u_i_feat, v_i_feat, n_i_feat) = sample.get_local_reference_frames()

    num_residues_with_coords = np.sum(~missing_coords_residue_mask)

    n_i_feat, u_i_feat, v_i_feat, atom37_coords, atom37_mask = tree.map_structure(
        lambda x: x[~missing_coords_residue_mask],
        (
            n_i_feat,
            u_i_feat,
            v_i_feat,
            atom37_coords,
            atom37_mask,
        ),
    )

    res_representatives_loc_feat = (
        atom37_coords[:, CA_INDEX]
        if residue_loc_is_alphac
        else np.mean(atom37_coords, axis=1, where=atom37_mask)
    )

    # Align unbound and bound structures, if needed
    if not residue_loc_is_alphac:
        (
            res_representatives_loc_feat,
            n_i_feat,
            u_i_feat,
            v_i_feat,
        ) = protein_align_unbound_and_bound(
            stacked_residue_representatives_coordinates=res_representatives_loc_feat,
            protein_n_i_feat=n_i_feat,
            protein_u_i_feat=u_i_feat,
            protein_v_i_feat=v_i_feat,
            alphac_atom_coordinates=atom37_coords[:, CA_INDEX],
        )

    # Build the k-NN graph
    (
        n_node,
        n_edge,
        nodes_res_feat,
        nodes_x,
        nodes_surface_aware_features,
        edges_features,
        senders,
        receivers,
    ) = compute_nearest_neighbors_graph(
        protein_num_residues=num_residues_with_coords,
        list_atom_coordinates=[
            atom37_coords[i, atom37_mask[i]] for i in range(num_residues_with_coords)
        ],
        residue_list=[
            RESTYPE_1TO3[res_type]
            for res_type, are_coords_missing in zip(
                sequence, missing_coords_residue_mask
            )
            if not are_coords_missing
        ],
        stacked_residue_coordinates=res_representatives_loc_feat,
        protein_n_i_feat=n_i_feat,
        protein_u_i_feat=u_i_feat,
        protein_v_i_feat=v_i_feat,
        num_neighbor=num_neighbor,
    )

    nodes_mask = np.ones((n_node,), dtype=bool)

    def crop_pad(data):
        return jax.tree_map(
            lambda x: np.pad(
                x[:padding_num_residue],
                ((0, padding_num_residue - n_node), *((0, 0) for _ in x.shape[1:])),
            ),
            data,
        )

    if nodes_res_feat.shape[0]:
        nodes_res_feat = crop_pad(nodes_res_feat)
    nodes_x = crop_pad(nodes_x)
    nodes_surface_aware_features = crop_pad(nodes_surface_aware_features)
    nodes_mask = crop_pad(nodes_mask)

    padding_num_edges = num_neighbor * padding_num_residue

    edges_features = np.pad(
        edges_features[:padding_num_edges], ((0, padding_num_edges - n_edge), (0, 0))
    )

    (senders, receivers) = jax.tree_map(
        lambda x: np.concatenate(
            [
                np.array(x),
                np.repeat(np.arange(n_node, padding_num_residue), num_neighbor),
            ],
            axis=0,
        )[:padding_num_edges],
        (senders, receivers),
    )

    protein_graph = ProteinGraph(  # type: ignore
        n_node=np.expand_dims(n_node, axis=-1),
        n_edge=np.expand_dims(n_edge, axis=-1),
        nodes_residue_features=nodes_res_feat,
        nodes_original_coordinates=nodes_x,
        nodes_ground_truth_coordinates=nodes_x,
        nodes_surface_aware_features=nodes_surface_aware_features,
        nodes_mask=np.expand_dims(nodes_mask, axis=-1),
        edges_features=edges_features,
        senders=senders,
        receivers=receivers,
    )

    return BatchDataWithTokensBioClip(  # type: ignore
        tokens=tokens,
        graph=protein_graph,
    )
