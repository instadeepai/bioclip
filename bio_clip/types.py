# from dataclasses import dataclass
from typing import Any, Tuple, Union

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pandas
from typing_extensions import TypeAlias

Mask: TypeAlias = jnp.ndarray
Coordinates: TypeAlias = jnp.ndarray
Loss: TypeAlias = jnp.float32
RNGKey: TypeAlias = jnp.ndarray
Residue: TypeAlias = str
Sequence: TypeAlias = str
NodeFeatures: TypeAlias = jnp.ndarray
EdgeFeatures: TypeAlias = jnp.ndarray
PdbRawData: TypeAlias = pandas.DataFrame
SortedPdbData: TypeAlias = Tuple[Tuple[Residue, int, Residue], PdbRawData]
TranslationVector: TypeAlias = jnp.ndarray
RotationMatrix: TypeAlias = jnp.ndarray
Metrics: TypeAlias = Tuple[Any, ...]
GraphTransformerFeatures: TypeAlias = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


# @dataclass
@jdc.pytree_dataclass
class ProteinGraph:
    """
    n_node (jnp.ndarray): number of nodes
    n_edge (jnp.ndarray): number of edges
    nodes_residue_features (NodeFeatures): embeddings of residues
    nodes_original_coordinates (Coordinates): original coordinates (i.e. coordinates
        at the beginning of the message passing process. This corresponds to
        rotated coordinates of the protein)
    nodes_ground_truth_coordinates (Coordinates): ground-truth coordinates
    nodes_surface_aware_features (NodeFeatures):
        Surface Aware Node Features (cf Equation 16 of Equidock paper)
    nodes_mask (Mask): mask indicating which residues are padded
    edges_features (EdgeFeatures): edge features (cf Appendix of Equidock paper)
    senders (ndarray): origin of graph vertices
    receivers (jnp.ndarray): destination of graph vertices
    """

    n_node: jnp.ndarray
    n_edge: jnp.ndarray
    nodes_residue_features: NodeFeatures
    nodes_original_coordinates: Coordinates
    nodes_ground_truth_coordinates: Coordinates
    nodes_surface_aware_features: NodeFeatures
    nodes_mask: Mask
    edges_features: EdgeFeatures
    senders: jnp.ndarray
    receivers: jnp.ndarray


# @dataclass
@jdc.pytree_dataclass
class BatchDataBioClip:
    """Batch data of BioCLIP
    Args:
        graph (ProteinGraph): graph of the protein
        sequence (jnp.ndarray): sequence (indexes) of the protein
    """

    graph: Union[ProteinGraph, GraphTransformerFeatures]
    sequence: jnp.ndarray


@jdc.pytree_dataclass
# @dataclass
class ProteinInteractionBatch:
    """Batch data of Protein Interaction task
    Args:
        batch_data_bioclip (BatchDataBioClip): batch data for protein 1
        batch_data_bioclip_2 (BatchDataBioClip): batch data for protein 2
        target (jnp.ndarray): target value indicating whtether the two
            proteins interact
    """

    batch_data_bioclip: BatchDataBioClip
    batch_data_bioclip_2: BatchDataBioClip
    target: jnp.ndarray


@jdc.pytree_dataclass
# @dataclass
class FunctionPredictionBatch:
    """Batch data of Function Prediction task
    Args:
        batch_data_bioclip (BatchDataBioClip): batch data for protein
        target (jnp.ndarray): target array indicating the labels of protein
    """

    batch_data_bioclip: BatchDataBioClip
    target: jnp.ndarray


# @dataclass
@jdc.pytree_dataclass
class BatchDataWithTokensBioClip:
    """Batch data of BioCLIP
    Args:
        graph (ProteinGraph): graph of the protein
        tokens (jnp.ndarray): tokenized sequence of the protein
    """

    graph: Union[ProteinGraph, GraphTransformerFeatures]
    tokens: jnp.ndarray


# @dataclass
@jdc.pytree_dataclass
class MultimodalEmbeddings:
    """Multimodal embeddings of BioCLIP (+ temperature)
    Args:
        projected_structure_embedding (jnp.ndarray): projected embedding of structure
        projected_sequence_embedding (jnp.ndarray): projected embedding of sequence
        structure_embedding (jnp.ndarray): embedding of structure
        sequence_embedding (jnp.ndarray): embedding of sequence
        temperature (jnp.ndarray): temperature coefficient in BioCLIP logits
        predicted_coordinates (Coordinates): coordinates predicted by GNN
        residue_representations (jnp.ndarray): nodes embeddings predicted by GNN
    """

    projected_structure_embedding: jnp.ndarray
    projected_sequence_embedding: jnp.ndarray
    structure_embedding: jnp.ndarray
    sequence_embedding: jnp.ndarray
    temperature: jnp.ndarray
    predicted_coordinates: Coordinates
    residue_representations: jnp.ndarray


# @dataclass
@jdc.pytree_dataclass
class ProteinFeatures:
    """Set of features that characterize a protein graph
    Args:
        predicted_coordinates (Coordinates): 3D coordinates of protein
        original_coordinates (Coordinates): original 3D coordinates of
            protein (does not change)
        predicted_embeddings (NodeFeatures): node embeddings that are
            updated at every layer iteration
        original_node_features (NodeFeatures): node features
        original_edge_features (EdgeFeatures): edge features
    """

    predicted_coordinates: Coordinates
    original_coordinates: Coordinates
    predicted_embeddings: NodeFeatures
    original_node_features: NodeFeatures
    original_edge_features: EdgeFeatures
