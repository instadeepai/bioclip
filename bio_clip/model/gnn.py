import functools
import math
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from ml_collections import ConfigDict

from bio_clip.model.gnn_layers import GATLayer, GraphTransformerModel, MPNNLayer
from bio_clip.types import (
    BatchDataBioClip,
    EdgeFeatures,
    MultimodalEmbeddings,
    NodeFeatures,
    ProteinFeatures,
    ProteinGraph,
)

_GNN_layer_classes = {"GATLayer": GATLayer, "MPNNLayer": MPNNLayer}


class PositionalEncodingLayer(hk.Module):
    """Independent-Equivariant Graph Matching Network"""

    def __init__(self, positional_encoding_dimension: int):
        """Initializes a Positional Encoding Layer

        Args:
            positional_encoding_dimension (int): dimension of positional embedding
        """

        super(PositionalEncodingLayer, self).__init__()
        self.positional_encoding_dimension = positional_encoding_dimension

        # Create random orthogonal matrix

        matrix1 = np.random.randn(
            positional_encoding_dimension, positional_encoding_dimension
        )
        u, s, vh = np.linalg.svd(matrix1, full_matrices=False)
        matrix2 = u @ vh
        self.orthogonal_matrix = matrix2 @ matrix2.T

    def sinusoidal_positional_encoding(
        self, x: int, n: int, d: int, k: int
    ) -> jnp.float32:
        """Computes positional encoding for two indices x and k

        Args:
            x (int): position index
            n (int): number of indices
            d (int): number of feature dimensions
            k (int): position index

        Returns:
            sinusoidal positional encoding (jnp.float32)
        """

        return jnp.mod(k, 2) * jnp.cos((x * math.pi) / n ** (2 * (k - 1) / d)) - (
            jnp.mod(k, 2) - 1
        ) * jnp.sin((x * math.pi) / n ** (2 * k / d))

    def sinusoidal_positional_encoding_features(
        self, indice_i: int, indice_j: int, number_residues: int
    ) -> jnp.array:
        """Computes positional encoding vector for two indices i and j

        Args:
            indice_i (int): position index
            indice_j (int): position index
            number_residues (int): number of residue indices

        Returns:
            sinusoidal positional encoding vector (jnp.array)
        """

        difference_indices = indice_i - indice_j
        list_indices = jnp.arange(1, self.positional_encoding_dimension + 1)

        sinusoidal_fn = functools.partial(
            self.sinusoidal_positional_encoding,
            difference_indices,
            number_residues,
            self.positional_encoding_dimension,
        )
        return jax.vmap(sinusoidal_fn)(list_indices)

    def __call__(
        self,
        n_node: int,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        diffusion: bool,
        diffusion_time_step: int,
    ) -> Tuple[NodeFeatures, EdgeFeatures]:
        """Computes positional embeddings for nodes and edges of a give protein

        Args:
            n_node (int):  number of nodes
            senders (jnp.ndarray): origins of graph vertices
            receivers (jnp.ndarray): destinations of graph vertices
            diffusion (bool): whether we perform diffusion
            diffusion_time_step (int): diffusion time step

        Returns:
            node_positional_encoding (NodeFeatures): node positional embeddings
            edge_positional_encoding (EdgeFeatures): edge positional embeddings
        """

        # Build nodes positional embeddings

        node_sinusoidal_fn = functools.partial(
            self.sinusoidal_positional_encoding_features,
            indice_j=0,
            number_residues=n_node,
        )

        nodes_positional_embeddings = jax.vmap(node_sinusoidal_fn)(jnp.arange(n_node))

        if diffusion:
            diffusion_embedding = jnp.matmul(
                jax.vmap(node_sinusoidal_fn)(diffusion_time_step * jnp.ones(n_node)),
                self.orthogonal_matrix,
            )
            nodes_positional_embeddings = (
                nodes_positional_embeddings + diffusion_embedding
            )

        # Build edges positional embeddings

        edge_sinusoidal_fn = functools.partial(
            self.sinusoidal_positional_encoding_features, number_residues=n_node
        )

        edges_positional_embeddings = jax.vmap(edge_sinusoidal_fn)(
            indice_i=senders, indice_j=receivers
        )

        return nodes_positional_embeddings, edges_positional_embeddings

    def __repr__(self) -> str:
        return "PositionalEncodingLayer" + str(self.__dict__)


class GraphNeuralNetwork(hk.Module):
    """Independent-Equivariant Graph Matching Network"""

    def __init__(
        self, config: ConfigDict, feature_config: ConfigDict, graph_max_neighbor: int
    ):
        """Initialise a GNN
        Based on:
            Ganea, Octavian-Eugen, et al. "Independent se (3)-equivariant models for
            end-to-end rigid protein docking." arXiv preprint arXiv:2111.07786 (2021).

        However, we only consider latent node embeddings and omit learned/transformed
        node positional embeddings. In practice this means that, our model is invariant
        under SE(3), rather than equivariant. The network is implemented, essentially,
        as a conventional message passing neural network. SE(3) invariance is achieved
        by pre-processing of the features (see e.g. bioclip.training.preprocessing) In
        particular, a local coordinate system is induced for each residue by
        constructing a plane from:
            - the vector from the alpha-carbon atom to the nitrogen atom
            - the vector from the alpha-carbon atom to the carbon atom of carboxyl
        The planes for any two residues can then be used to describe SE(3) invariant
        relative positions and orientations. As all node + edge features are SE(3)
        invariant at input, then the output is also SE(3) invariant when using any
        conventional message passing network. It should also be noted that as nodes and
        edges are assigned positional encodings according to the nodes' original
        positions in the sequence this architecture is not permutation invariant.

        Args:
            config (ConfigDict): GNN hyperparameters. Some important parameters:
                gnn_layer:
                    layer_cls (string): The named class for GNN layers. Can be any from
                    _GNN_layer_classes e.g.:
                        "GATLayer" -- Gated Attention Layer -- message passing with
                        multi-head attention applied across the neighborhood "MPNNLayer"
                        -- Standard message passing layer
                    hidden_dimension (int): The hidden dimension of GNN layers
                gnn_number_layers (int): Number of message passing layers shared_layers
                (bool): Whether to re-use the same hidden message passing layer for each
                of the gnn_number_layers steps of message passing
                positional_encoding_dimension (int): The dimension of the positional
                encoding of the nodes and edges in the graph according to their
                    indices in the original sequence data. Note; this feature is
                    responsible for the architecture being not permutation invariant.
                residue_embedding_dim (int): Dimension of learned embeddings of residues

        """

        super(GraphNeuralNetwork, self).__init__()
        self.feature_config = feature_config
        self.config = config

        layer_cls = _GNN_layer_classes[self.config.gnn_layer.layer_cls]

        # The first GNN layer is always unique
        self.gnn_layers = [
            layer_cls(
                config=self.config.gnn_layer,
                graph_max_neighbor=graph_max_neighbor,
            )
        ]

        # If using shared layers, create a single instance of the GNN layer class and
        # repeat it
        if self.config.shared_layers:
            intermediate_layer = layer_cls(
                config=self.config.gnn_layer,
                graph_max_neighbor=graph_max_neighbor,
            )
            for _ in range(1, self.config.gnn_number_layers):
                self.gnn_layers.append(intermediate_layer)
        # Otherwise instantiate gnn_number_layers GNN layers total
        else:
            for _ in range(1, self.config.gnn_number_layers):
                self.gnn_layers.append(
                    layer_cls(
                        config=self.config.gnn_layer,
                        graph_max_neighbor=graph_max_neighbor,
                    )
                )

    def __call__(self, graph: ProteinGraph) -> List[jnp.ndarray]:
        """Computes the representations of the residues at each layer of the GNN

        Args:
            graph (ProteinGraph):  graph representing a protein e.g. containing residue
            information, edge information, structure etc.

        Returns:
            representations (List[jnp.ndarray]): the learned representations of the
            residues taken from each layer e.g.:
                    the initial representation consisting of positional information,
                    residue information etc under a linear projection a representation
                    for each GNN layer a representation after a final MLP
                each representation has shape [V, H] where V is the number of nodes and
                H is the dimension of that representation
        """
        pos_enc_dim = self.config.positional_encoding_dimension

        # === Initial node and edge features ===

        # Use node order (w.r.t. protein sequence) to create an embedding for each node
        # and edge
        (
            node_positional_embedding,
            edge_positional_embedding,
        ) = PositionalEncodingLayer(positional_encoding_dimension=pos_enc_dim)(
            n_node=graph.nodes_original_coordinates.shape[0],
            senders=graph.senders,
            receivers=graph.receivers,
            diffusion=False,
            diffusion_time_step=0,
        )
        print(f"edge_positional_embedding: {edge_positional_embedding.shape}")
        print(f"graph.edges_features: {graph.edges_features.shape}")

        # Build node features as a list and cat
        protein_node_features = []

        if self.feature_config.use_positional_features:
            protein_node_features.append(node_positional_embedding)

        if self.feature_config.use_residual_information:
            # If we are using residual information, each of the 21 residue types is
            # associated with a learned embedding that is used as a node feature. It is
            # assumed that the residue type is the first column of the
            # node_residue_features.
            residual_embedding = hk.Embed(
                vocab_size=21,
                embed_dim=self.config.residue_embedding_dim,
                name="residue_embedding_layer",
            )(graph.nodes_residue_features[:, 0])
            protein_node_features.append(residual_embedding)

        if self.feature_config.use_mean_node_features:
            # If we are using the surface-aware node features (see the EquiDock paper)
            # then we take their log to arrive at a more NN-friendly scaling
            protein_node_features.append(
                jnp.log(jnp.array(graph.nodes_surface_aware_features))
            )

        protein_node_features = jnp.concatenate(protein_node_features, axis=-1)

        original_edge_features_protein = jnp.concatenate(
            [
                self.feature_config.use_positional_features * edge_positional_embedding,
                graph.edges_features,
            ],
            axis=1,
        )

        # === Initial Projection ===

        # A linear projection lets us ensure that the node and edge features are in the
        # configurable hidden dimension
        node_proj = hk.Linear(
            self.config.gnn_layer.hidden_dimension, name="init_node_embed"
        )
        protein_node_features = node_proj(protein_node_features)
        original_protein_node_features = protein_node_features

        edge_proj = hk.Linear(
            self.config.gnn_layer.hidden_dimension // 4, name="init_edge_embed"
        )
        original_edge_features_protein = edge_proj(original_edge_features_protein)

        representations = [protein_node_features]

        # === Message Passing ===

        for layer in self.gnn_layers:
            # Gather features to pass to the GNN layer
            features = ProteinFeatures(
                predicted_coordinates=graph.nodes_original_coordinates,
                predicted_embeddings=protein_node_features,
                original_coordinates=graph.nodes_original_coordinates,
                original_node_features=original_protein_node_features,
                original_edge_features=original_edge_features_protein,
            )
            # Apply layer and store the output representation
            protein_node_features = layer(
                graph_protein=graph, features_protein=features
            )
            representations.append(protein_node_features)

        print(f"representation shapes: {[a.shape for a in representations]}")
        return representations

    def __repr__(self) -> str:
        return "GraphNeuralNetwork " + str(self.__dict__)


class ForwardBioClip(hk.Module):
    """Graph Neural Network projected in multimodal BioCLIP space"""

    def __init__(
        self,
        feature_config,
        model_config,
        data_config,
        training_config,
        use_seq=True,
        target_type="both",
        gnn_layer_to_use=-1,
    ):
        """Class to compute multimodal embeddings for the structure and sequence

        Args:
            hyperparameters (BioClipConfig): model hyperparameters
            data_config (DataProcessingConfig): data processing configuration
        """

        super(ForwardBioClip, self).__init__()
        self.feature_cfg = feature_config
        self.model_cfg = model_config
        self.data_config = data_config
        self.use_seq = use_seq
        self.target_type = target_type
        self.gnn_layer_to_use = gnn_layer_to_use
        self.use_projected_sequence_embedding = (
            training_config.use_projected_sequence_embedding
        )
        self.esm_embedding_size = training_config.esm_embedding_size

        if training_config.mixed_precision:
            # Use mixed precision (only support A100 GPU and TPU for now)
            half = jnp.bfloat16
            full = jnp.float32

            policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)
            hk.mixed_precision.set_policy(GraphNeuralNetwork, policy)

            # Remove it in batch norm to avoid instabilities
            policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=half)
            hk.mixed_precision.set_policy(hk.BatchNorm, policy)
            hk.mixed_precision.set_policy(hk.LayerNorm, policy)

    def __call__(self, batch_data: BatchDataBioClip) -> MultimodalEmbeddings:
        """Computes multimodal embeddings for the structure and sequence

        Args:
            batch_data (BatchDataBioClip): batch data

        Returns:
            multimodal_embedding (MultimodalEmbeddings): embedding of multimodal
                projections into the BioCLIP embedding space
        """
        if self.model_cfg.architecture == "gnn":
            mask = jnp.squeeze(batch_data.graph.nodes_mask)
            nodes_original_coordinates = batch_data.graph.nodes_original_coordinates
            print("\nInstantiating: GNN\n")
            all_layers_gnn_representation = GraphNeuralNetwork(
                config=self.model_cfg.gnn,
                feature_config=self.feature_cfg,
                graph_max_neighbor=self.data_config.fixed_sizes.graph_max_neighbor,
            )(batch_data.graph)
        elif self.model_cfg.architecture == "graph_transformer":
            *_, atom14_mask = batch_data.graph
            mask = jnp.squeeze(atom14_mask.any(-1))

            def cf(nn):
                return {"Ns": 32, "Nh": 2, "Nk": 3, "nn": nn}

            bl = self.model_cfg.gnn.blocks  # PeSTo used 8
            official_config = {
                "em": {"N0": 30, "N1": 32},
                "sum": [cf(8)] * bl + [cf(16)] * bl + [cf(32)] * bl + [cf(64)] * bl,
                "spl": {"N0": 32, "N1": 32, "Nh": 4},
                "dm": {"N0": 32, "N1": 32, "N2": 5},
            }
            # overwrite with clip size
            official_config["dm"]["N2"] = self.esm_embedding_size
            all_layers_gnn_representation = [
                GraphTransformerModel(official_config, self.model_cfg.remat_policy)(
                    *batch_data.graph
                )
            ]
            self.gnn_layer_to_use = -1
            nodes_original_coordinates = jnp.zeros(0)  # not used- legacy bioclip
        elif self.model_cfg.architecture == "evoformer":
            raise NotImplementedError
        else:
            raise Exception(
                f"architecture: {self.model_cfg.architecture} not in "
                "['gnn', 'evoformer']"
            )

        # We have an MLP at index -1, and last GNN representation at -1
        gnn_representation = all_layers_gnn_representation[self.gnn_layer_to_use]

        if self.use_projected_sequence_embedding:
            # this is to make sure we match the hidden size in the multimodal space
            h = self.model_cfg.dimension_multimodal_space
        else:
            h = self.esm_embedding_size
        print(f"target_type: {self.target_type}")

        chain_representation_on = self.target_type in ["per-chain", "both"]
        residue_representation_on = self.target_type in ["per-res", "both"]

        representations = {"gnn": {}, "esm": {}}
        language_model_output = jnp.zeros(1)  # Defaults
        projected_language_model = jnp.zeros(1)
        g_h = gnn_representation.shape[-1]

        if chain_representation_on:
            query = jnp.ones((1, 1))

            # Optional: dont send gradients through the aggregations
            if (
                self.model_cfg.gnn.stop_aggregation_gradient
                and self.target_type == "both"
            ):
                memory = jax.lax.stop_gradient(gnn_representation)
            else:
                memory = gnn_representation

            stddev = 1.0 / np.sqrt(memory.shape[-1])

            aggregated_structure = jnp.squeeze(
                hk.MultiHeadAttention(
                    self.model_cfg.aggregation_attention.num_heads,
                    self.model_cfg.aggregation_attention.key_size,
                    w_init=hk.initializers.TruncatedNormal(stddev=stddev),
                    value_size=self.model_cfg.aggregation_attention.value_size,
                    model_size=h,
                    name="gat_multi_head_attention",
                )(query=query, key=memory, value=memory, mask=mask[None, None, :])
            )

            # this is mostly needed simply so that we have a 'head' to cut off to ensure
            # that the structure representation is distinct (under a non-linear
            # transformation) from the sequence representation
            hidden_dims = [g_h * 2, g_h * 2, h]
            final_per_chain_structure_rep = hk.nets.MLP(
                hidden_dims,
                activation=jax.nn.relu,
                activate_final=False,
                name="post_aggregation_mlp",
            )(aggregated_structure)

            representations["gnn"]["chain"] = (
                aggregated_structure,
                final_per_chain_structure_rep,
            )

            if self.use_seq:
                esm_out = batch_data.sequence  # [seq_length, channels]
                language_model_output = jnp.mean(
                    esm_out,
                    axis=0,
                    where=jnp.repeat(mask[:, None], esm_out.shape[-1], axis=-1),
                )
                projected_language_model = hk.Linear(h, name="seq_proj")(
                    language_model_output
                )

            representations["esm"]["chain"] = (
                language_model_output,
                projected_language_model,
            )

        if residue_representation_on:
            # === Per-residue MLP ===

            # this is mostly needed simply so that we have a 'head' to cut off to ensure
            # that the structure representation is distinct (under a non-linear
            # transformation) from the sequence representation
            hidden_dims = [g_h * 2, g_h * 2, h]
            final_per_residue_structure_rep = hk.nets.MLP(
                hidden_dims,
                activation=jax.nn.relu,
                activate_final=False,
                name="per_residue_mlp",
            )(gnn_representation)

            representations["gnn"]["residue"] = (
                gnn_representation,
                final_per_residue_structure_rep,
            )

            if self.use_seq:
                # ESM
                language_model_output = batch_data.sequence
                projected_language_model = hk.Linear(h, name="seq_proj")(esm_out)

            representations["esm"]["residue"] = (
                language_model_output,
                projected_language_model,
            )

        temperature = hk.get_parameter(
            "temperature",
            shape=[],
            dtype=gnn_representation.dtype,
            init=hk.initializers.Constant(self.model_cfg.temperature_initialization),
        )

        # Annoying un-pack and re-pack
        if self.target_type == "per-chain":
            structure_embedding, projected_structure_embedding = representations["gnn"][
                "chain"
            ]
            language_model_output, projected_language_model = representations["esm"][
                "chain"
            ]
        elif self.target_type == "both":
            (
                chain_structure_embedding,
                chain_projected_structure_embedding,
            ) = representations["gnn"]["chain"]
            (
                chain_language_model_output,
                chain_projected_language_model,
            ) = representations["esm"]["chain"]
            (
                residue_structure_embedding,
                residue_projected_structure_embedding,
            ) = representations["gnn"]["residue"]
            (
                residue_language_model_output,
                residue_projected_language_model,
            ) = representations["esm"]["residue"]
            structure_embedding = (
                residue_structure_embedding,
                chain_structure_embedding,
            )
            projected_structure_embedding = (
                residue_projected_structure_embedding,
                chain_projected_structure_embedding,
            )
            language_model_output = (
                residue_language_model_output,
                chain_language_model_output,
            )
            projected_language_model = (
                residue_projected_language_model,
                chain_projected_language_model,
            )
        elif self.target_type == "per-res":
            structure_embedding, projected_structure_embedding = representations["gnn"][
                "residue"
            ]
            language_model_output, projected_language_model = representations["esm"][
                "residue"
            ]
        else:
            raise ValueError(f"target_type is not in ['per-chain', 'per-res', 'both']")

        multimodal_embeddings = MultimodalEmbeddings(
            projected_structure_embedding=projected_structure_embedding,
            projected_sequence_embedding=projected_language_model,
            structure_embedding=structure_embedding,
            sequence_embedding=language_model_output,
            predicted_coordinates=nodes_original_coordinates,
            residue_representations=gnn_representation,
            temperature=temperature,
        )

        return multimodal_embeddings

    def __repr__(self) -> str:
        return "ForwardBioClip" + str(self.__dict__)
