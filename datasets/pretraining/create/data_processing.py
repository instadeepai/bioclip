"""This is a flattened version of the original code which produced the pretraining
dataset."""
import logging
import math
import os
import pickle
import warnings
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.spatial as spa
from biopandas.pdb import PandasPdb
from joblib import Parallel, cpu_count, delayed
from numpy import linalg as la
from scipy.special import softmax

from bio_clip.types import (
    Coordinates,
    EdgeFeatures,
    NodeFeatures,
    ProteinGraph,
    Residue,
    RotationMatrix,
    SortedPdbData,
    TranslationVector,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def padded_graph(graph: ProteinGraph, pad: int) -> ProteinGraph:
    """Pads a ProteinGraph to a number of residues equal to "padding"

    Args:
        graph (ProteinGraph): protein graph
        pad (int): padding number

    Returns:
        ProteinGraph: padded graph
    """
    nodes_res_feat = graph.nodes_residue_features
    nodes_x = graph.nodes_original_coordinates
    nodes_gt = graph.nodes_ground_truth_coordinates
    nodes_surface_aware_features = graph.nodes_surface_aware_features
    n_node = graph.n_node
    n_edge = graph.n_edge
    edges_features = graph.edges_features
    senders = graph.senders
    receivers = graph.receivers
    nodes_mask = graph.nodes_mask

    if nodes_mask.shape[0] < pad:
        padded_mask = jnp.reshape(
            jnp.pad(
                nodes_mask,
                ((0), (pad - nodes_mask.shape[0])),
            ),
            (-1, 1),
        )[:pad]
    else:
        padded_mask = nodes_mask[:pad]

    return ProteinGraph(
        n_node=n_node,
        n_edge=n_edge,
        nodes_residue_features=jnp.pad(
            jnp.array(nodes_res_feat),
            ((0, pad - nodes_res_feat.shape[0]), (0, 0)),
        ),
        nodes_original_coordinates=jnp.pad(
            jnp.array(nodes_x),
            ((0, pad - nodes_x.shape[0]), (0, 0)),
        ),
        nodes_ground_truth_coordinates=jnp.pad(
            jnp.array(nodes_gt),
            ((0, pad - nodes_x.shape[0]), (0, 0)),
        ),
        nodes_surface_aware_features=jnp.pad(
            jnp.array(nodes_surface_aware_features),
            ((0, pad - nodes_surface_aware_features.shape[0]), (0, 0)),
            mode="maximum",
        ),
        nodes_mask=padded_mask,
        edges_features=jnp.pad(
            jnp.array(edges_features),
            ((0, pad * 10 - edges_features.shape[0]), (0, 0)),
        ),
        senders=jnp.concatenate(
            [
                jnp.array(senders),
                jnp.repeat(np.arange(n_node, pad), 10),
            ],
            axis=0,
        )[: pad * 10],
        receivers=jnp.concatenate(
            [
                jnp.array(receivers),
                jnp.repeat(np.arange(n_node, pad), 10),
            ],
            axis=0,
        )[: pad * 10],
    )


def extract_3D_coordinates_and_feature_vectors(
    protein_predic: List[SortedPdbData], residue_loc_is_alphac: bool
) -> Tuple[
    List[Coordinates],
    jnp.ndarray,
    NodeFeatures,
    NodeFeatures,
    NodeFeatures,
    int,
]:
    """Extract 3D coordinates and features vectors of representative residues

    Args:
        protein_predic (List[SortedPdbData]): Raw PDB data of a protein
        residue_loc_is_alphac (bool, optional): whether the alpha-C atom
            is used for residue center.
    Returns:
        protein_all_atom_coords_in_residue_list (List[Coordinates]):
            list of atom coordinates ordered by residues
        protein_residue_representatives_loc_feat (jnp.ndarray):
            residues coordinates
        protein_n_i_feat (NodeFeatures): residues features
        protein_u_i_feat (NodeFeatures): residues features
        protein_v_i_feat (NodeFeatures): residues features
        protein_num_residues (int): number of residues
    """
    protein_all_atom_coords_in_residue_list = []
    protein_residue_representatives_loc_list = []
    protein_n_i_list = []
    protein_u_i_list = []
    protein_v_i_list = []

    for residue in protein_predic:
        df = residue[1]
        coord = df[["x", "y", "z"]].to_numpy().astype(np.float32)
        protein_all_atom_coords_in_residue_list.append(coord)

        natom = df[df["atom_name"] == "N"]
        alphacatom = df[df["atom_name"] == "CA"]
        catom = df[df["atom_name"] == "C"]

        if (natom.shape[0] != 1) or (alphacatom.shape[0] != 1) or (catom.shape[0] != 1):
            logging.warning(
                "protein utils compute_graph_of_protein," + "no N/CA/C exists"
            )
            raise ValueError

        n_loc = natom[["x", "y", "z"]].to_numpy().squeeze().astype(np.float32)
        alphac_loc = alphacatom[["x", "y", "z"]].to_numpy().squeeze().astype(np.float32)
        c_loc = catom[["x", "y", "z"]].to_numpy().squeeze().astype(np.float32)

        u_i = (n_loc - alphac_loc) / la.norm(n_loc - alphac_loc)
        t_i = (c_loc - alphac_loc) / la.norm(c_loc - alphac_loc)
        n_i = np.cross(u_i, t_i) / la.norm(np.cross(u_i, t_i))
        v_i = np.cross(n_i, u_i)
        if math.fabs(la.norm(v_i) - 1.0) > 1e-5:
            logging.warning(
                "protein utils protein_to_graph_dips, v_i norm larger than 1"
            )
            raise ValueError

        protein_n_i_list.append(n_i)
        protein_u_i_list.append(u_i)
        protein_v_i_list.append(v_i)

        if residue_loc_is_alphac:
            protein_residue_representatives_loc_list.append(alphac_loc)
        else:
            heavy_df = df[df["element"] != "H"]
            residue_loc = (
                heavy_df[["x", "y", "z"]].mean(axis=0).to_numpy().astype(np.float32)
            )  # average of all atom coordinates
            protein_residue_representatives_loc_list.append(residue_loc)

    protein_residue_representatives_loc_feat = np.stack(
        protein_residue_representatives_loc_list, axis=0
    )  # (N_res, 3)
    protein_n_i_feat = np.stack(protein_n_i_list, axis=0)
    protein_u_i_feat = np.stack(protein_u_i_list, axis=0)
    protein_v_i_feat = np.stack(protein_v_i_list, axis=0)

    protein_num_residues = len(protein_predic)
    if protein_num_residues <= 1:
        logging.warning("protein contains only 1 residue!")
        raise ValueError

    return (
        protein_all_atom_coords_in_residue_list,
        protein_residue_representatives_loc_feat,
        protein_n_i_feat,
        protein_u_i_feat,
        protein_v_i_feat,
        protein_num_residues,
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


def residue_list_featurizer(
    residue_list: List[str],
) -> np.ndarray:
    """
    Creates residual features given PDB data as input.
    A residue is simply embedded by an index.

    Args:
        residue_list (List[str]): list of 3-letter amino-acid codes

    Returns:
        Residual features
    """
    feature_list = [[residue_embedding(residue)] for residue in residue_list]
    feature_list = np.array(feature_list)
    return feature_list.astype(np.int32)


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

    transformed_dist = [
        np.exp(-((dist_list - center) ** 2) / float(length_scale))
        for length_scale, center in zip(length_scale_list, center_list)
    ]

    transformed_dist = np.array(transformed_dist).T
    transformed_dist = transformed_dist.reshape((num_edge, -1))

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


def compute_graph_of_protein(
    protein_name: str,
    graph_folder: str,
    list_sorted_pdb_data: List[SortedPdbData],
    stacked_residue_coordinates: Coordinates,
    num_neighbor: int = 10,
    residue_loc_is_alphac: bool = True,
    save_graph: bool = True,
) -> Union[ProteinGraph, None]:
    """Computes the protein graph features and stores the
        graph in a file inside graph_folder

    Args:
        protein_name (str): Name of protein
        graph_folder (str): Graph folder of protein
        list_sorted_pdb_data (List[SortedPdbData]):
            raw PDB data for the ligand sorted by residue
        stacked_residue_coordinates (Coordinates):
            list of residue coordinates for the ligand
        num_neighbor (int, optional): number of nearest neighbors in the graph.
            Defaults to None.
        residue_loc_is_alphac (bool, optional): whether the alpha-C atom
            is used for residue center. Defaults to True.
        save_graph (bool): whether we save the graph or not

    Return:
        The corresponding ProteinGraph
    """

    if len(glob(os.path.join(graph_folder, f"{protein_name}_num_*"))) > 0:
        logging.warning("Graph already created")
        try:
            return pickle.load(
                open(glob(os.path.join(graph_folder, f"{protein_name}_num_*"))[0], "rb")
            )
        except (EOFError, pickle.UnpicklingError):
            return None

    if len(list_sorted_pdb_data) == 0:
        logging.warning("PDB badly separated of graph already done")
        return None

    try:
        (
            all_atom_coords_in_residue_list,
            residue_representatives_loc_feat,
            n_i_feat,
            u_i_feat,
            v_i_feat,
            num_residues,
        ) = extract_3D_coordinates_and_feature_vectors(
            list_sorted_pdb_data, residue_loc_is_alphac
        )
    except ValueError:
        logging.warning("Problem with 3D feature extraction")
        return None

    # Align unbound and bound structures, if needed
    res_shortcut = residue_representatives_loc_feat
    (
        residue_representatives_loc_feat,
        n_i_feat,
        u_i_feat,
        v_i_feat,
    ) = protein_align_unbound_and_bound(
        stacked_residue_representatives_coordinates=res_shortcut,
        protein_n_i_feat=n_i_feat,
        protein_u_i_feat=u_i_feat,
        protein_v_i_feat=v_i_feat,
        alphac_atom_coordinates=stacked_residue_coordinates,
    )

    try:
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
            protein_num_residues=num_residues,
            list_atom_coordinates=all_atom_coords_in_residue_list,
            residue_list=[term[1]["resname"].iloc[0] for term in list_sorted_pdb_data],
            stacked_residue_coordinates=residue_representatives_loc_feat,
            protein_n_i_feat=n_i_feat,
            protein_u_i_feat=u_i_feat,
            protein_v_i_feat=v_i_feat,
            num_neighbor=num_neighbor,
        )
    except (ValueError, TypeError):
        logging.warning("Problem with K-NN graph")
        return None

    protein_graph = ProteinGraph(
        n_node=jnp.array([n_node]),
        n_edge=jnp.array([n_edge]),
        nodes_residue_features=jnp.array(nodes_res_feat),
        nodes_original_coordinates=jnp.array(nodes_x),
        nodes_ground_truth_coordinates=jnp.array(nodes_x),
        nodes_surface_aware_features=jnp.array(nodes_surface_aware_features),
        nodes_mask=jnp.ones(n_node),
        edges_features=jnp.array(edges_features),
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
    )

    # Pad graph to the nearest multiple of 100
    padded_protein_graph = padded_graph(
        protein_graph, (1 + int(protein_graph.n_node / 100)) * 100
    )

    filename = os.path.join(
        graph_folder, f"{protein_name}_num_{str(padded_protein_graph.n_node[0])}.pkl"
    )

    if stacked_residue_coordinates.shape[0] != protein_graph.n_node[0]:
        logging.warning(
            "stacked_residue_coordinates.shape[0] != protein_graph.n_node[0]"
        )
        return None

    logging.warning(filename)
    if save_graph:
        pickle.dump(padded_protein_graph, open(filename, "wb"))

    return padded_protein_graph


def filter_residues(residues: List[SortedPdbData]) -> List[SortedPdbData]:
    """
    Filters residues that verify certain biological conditions (N, CA, and C atoms
    must have one occurrence)
    Args:
        residues (List[SortedPdbData]): raw PDB data sorted by residue

    Returns:
        residues_filtered (List[SortedPdbData]): raw PDB data sorted by residue,
        only certain residues are kept
    """
    residues_filtered = []
    for residue in residues:
        df = residue[1]
        natom = df[df["atom_name"] == "N"]
        alphacatom = df[df["atom_name"] == "CA"]
        catom = df[df["atom_name"] == "C"]
        if (
            (natom.shape[0] == 1)
            and (alphacatom.shape[0] == 1)
            and (catom.shape[0] == 1)
        ):
            residues_filtered.append(residue)

    return residues_filtered


def get_alphac_loc_array(bound_predic_clean_list: List[SortedPdbData]) -> Coordinates:
    """
    Args:
        residues (List[SortedPdbData]): raw PDB data sorted by residue

    Returns:
        (Coordinates): array of coordinates of the alpha-c atoms for each
        residue
    """
    bound_alphac_loc_clean_list = []
    for residue in bound_predic_clean_list:
        df = residue[1]
        alphacatom = df[df["atom_name"] == "CA"]
        alphac_loc = alphacatom[["x", "y", "z"]].to_numpy().squeeze().astype(np.float32)
        assert alphac_loc.shape == (
            3,
        ), f"alphac loc shape problem, shape: {alphac_loc.shape} \
                residue {df} resid {df['residue']}"
        bound_alphac_loc_clean_list.append(alphac_loc)
    if len(bound_alphac_loc_clean_list) <= 1:
        bound_alphac_loc_clean_list.append(np.zeros(3))
    return np.stack(bound_alphac_loc_clean_list, axis=0)


def get_residue_coordinates(
    residues: List[SortedPdbData],
) -> Tuple[List[SortedPdbData], Coordinates]:
    """Separates raw PDB data into residues and creates ligand and receptor coordinates

    Args:
        residues (List[SortedPdbData]): raw PDB data sorted by residue

    Returns:

        list_sorted_pdb (List[SortedPdbData]):
            list of "sorted PDB data" for the protein of interest
            Sorted PDB data consists of atom coordinates indexed by chain/residue
            information (e.g, ('B', 1, 'SER'))
        stacked_residue_coordinates (Coordinates):
            list of residue coordinates
    """

    bound_predic_filtered = filter_residues(residues)
    unbound_predic_filtered = bound_predic_filtered

    bound_predic_clean_list = bound_predic_filtered
    list_sorted_pdb_data = unbound_predic_filtered

    stacked_residue_coordinates = get_alphac_loc_array(bound_predic_clean_list)

    return (
        list_sorted_pdb_data,
        stacked_residue_coordinates,
    )


def pmap_multi(
    pickleable_fn: Callable, data: Any, n_jobs: int = 1, verbose: int = 1, **kwargs: Any
) -> Any:
    """
    Extends dgllife pmap function.

    Parallel map using joblib.

    Args
        pickleable_fn (Callable): Function to map over data.
        data (Any):
            Data over which we want to parallelize the function call.
        n_jobs (int):
            The maximum number of concurrently running jobs.
            By default, it is one less than
            the number of CPUs.
        verbose (int):
            The verbosity level.
            If nonzero, the function prints the progress messages.
            The frequency of the messages increases with the verbosity level.
            If above 10,
            it reports all iterations. If above 50, it sends the output to stdout.
        kwargs (Any):
            Additional arguments for :attr:`pickleable_fn`.

    Returns
        Output of applying attr:`pickleable_fn` to :attr:`data`.
    """
    if n_jobs is None:
        n_jobs = cpu_count() - 1

    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
        delayed(pickleable_fn)(*d, **kwargs) for i, d in enumerate(data)
    )

    return results


def get_residues_db5(pdb_filename: str) -> List[SortedPdbData]:
    """Sorts the pdb files of DB5 dataset by chain/residue

    Args:
        pdb_filename : filename of pdb

    Returns:
        List of residues, each residue being represented by:
            chain, residue, resname and all atom information
    """
    df = PandasPdb().read_pdb(pdb_filename)
    df = df.df["ATOM"]
    df.rename(
        columns={
            "chain_id": "chain",
            "residue_number": "residue",
            "residue_name": "resname",
            "x_coord": "x",
            "y_coord": "y",
            "z_coord": "z",
            "element_symbol": "element",
        },
        inplace=True,
    )
    residues = list(
        df.groupby(["chain", "residue", "resname"])
    )  # Not the same as sequence order !

    # Filter the first model of the PDB
    new_residues = []
    for i in range(len(residues)):
        res = np.array(residues[i][1].index)
        try:
            index_stop = np.where(np.diff(np.array(res)) > 1)[0][0]
            new_residues.append((residues[i][0], residues[i][1].head(index_stop)))
        except IndexError:
            new_residues.append(residues[i])

    return new_residues


@dataclass
class BioClipDataProcessingConfig:
    """Set of configurations to process proteins into graphs

    Args:
        cache_path (str): path of the cache directory
        graph_max_neighbor (int): number of nearest neighbors to build graph
        maximum_padding (int): maximum padding size
        minimum_padding (int): minimum padding size
        n_jobs (int): number of jobs to preprocess data
        pocket_cutoff (int): pocket cutoff
        graph_residue_loc_is_alphac (bool): whether residue coordinate is
            the coordinate of alpha-C or the average of atom coordinates
        sequence_pad_max (int): maximum padding size for sequences
    """

    cache_path: List[str]
    graph_max_neighbor: int = 10
    maximum_padding: int = 600
    minimum_padding: int = 0
    n_jobs: int = 96
    pocket_cutoff: float = 8.0
    graph_residue_loc_is_alphac: bool = True
    sequence_pad_max: int = 2500


def build_dataset_bioclip(
    data_config: BioClipDataProcessingConfig, raw_data_path: Path
) -> None:
    """
    Args:
        data_config (DataProcessingConfig): data configuration parameters
        raw_data_path (Path): path to raw data
    """

    # Filenames of all PDB files
    filenames_list = [
        [str(f.name)]
        for f in Path(raw_data_path).rglob("*")
        if ((str(f.name)[-3:] == "pdb") and ("._" not in str(f.name)))
    ]

    graph_folder = data_config.cache_path[0]
    os.makedirs(os.path.join(graph_folder, f"graph_train"), exist_ok=True)
    os.makedirs(os.path.join(graph_folder, f"graph_test"), exist_ok=True)
    os.makedirs(os.path.join(graph_folder, f"graph_valid"), exist_ok=True)

    def filename_to_graph(
        filename: str,
        graph_folder: str,
        graph_max_neighbor: int,
        graph_residue_loc_is_alphac: bool,
    ) -> None:
        residues = get_residues_db5(
            os.path.join(raw_data_path, str(filename).replace("._", ""))
        )
        preprocess_result = get_residue_coordinates(residues)
        (unbound_predic, bound_repres_nodes_loc_array) = preprocess_result
        protein_name = filename.split("/")[-1]
        # We randomly select the split
        sample_to_determine_split = np.random.uniform()
        if sample_to_determine_split < 0.8:
            split = "train"
        elif sample_to_determine_split < 0.9:
            split = "valid"
        else:
            split = "test"
        # We build the protein graph
        compute_graph_of_protein(
            protein_name=protein_name,
            graph_folder=os.path.join(graph_folder, f"graph_{split}"),
            list_sorted_pdb_data=unbound_predic,
            stacked_residue_coordinates=bound_repres_nodes_loc_array,
            num_neighbor=graph_max_neighbor,
            residue_loc_is_alphac=graph_residue_loc_is_alphac,
            save_graph=True,
        )

    pmap_multi(
        filename_to_graph,
        filenames_list,
        n_jobs=data_config.n_jobs,
        graph_folder=graph_folder,
        graph_max_neighbor=data_config.graph_max_neighbor,
        graph_residue_loc_is_alphac=data_config.graph_residue_loc_is_alphac,
    )
