"""The following GraphTransformerModel is a reimplementation of the graph transformer
from PeSTo: https://www.nature.com/articles/s41467-023-37701-8, however, ELU is replaced
by leaky-relu due to numerical issues.

def elu(x, alpha=1.0):
    #ELU activation function using jnp.
    # return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))
    return (x > 0) * x + (x <= 0) * alpha * (jnp.exp(x) - 1)
"""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from bio_clip.types import NodeFeatures, ProteinFeatures, ProteinGraph
from bio_clip.utils.mask_utils import MaskedLayerNorm
from bio_clip.utils.safe_key import SafeKey


def torch_mlp(sizes, name, with_bias=True):
    return hk.nets.MLP(
        sizes,
        with_bias=with_bias,
        activation=jax.nn.leaky_relu,
        activate_final=False,
        name=name,
    )


# Due to the issue that nans in the gradient can enter through masks described here:
# https://github.com/google/jax/issues/1052#issuecomment-514083352
# we must perform the sqrt as follows, with the abs and eps.
def sqrt(x):
    return ((jnp.abs(x) + 1e-6) ** 0.5) * (x >= 0.0).astype(float)


def norm(x, axis):
    return sqrt((x * x).sum(axis))


class MPNNLayer(hk.Module):
    """Layer of the Graph Neural Network"""

    def __init__(self, graph_max_neighbor: int, config):  #: BioClipConfig,
        """
        Constructs a Layer of an Independent Equivariant Graph Matching Network

        Args:
            graph_max_neighbor (DataProcessingConfig): data processing configuration
            config (EquidockConfig): config of the model
        """
        super(MPNNLayer, self).__init__()
        self.config = config
        self.graph_max_neighbor = graph_max_neighbor

    def __call__(
        self,
        graph_protein: ProteinGraph,
        features_protein: ProteinFeatures,
    ) -> NodeFeatures:
        """Computes node/edge embeddings of a protein graph using message passing

        Args:
            graph_protein (ProteinGraph): graph of the protein of interest
            features_protein (ProteinFeatures): features of the protein of interest

        Returns:
            node_features (NodeFeatures): embeddings of residues
        """
        dim = self.config.hidden_dimension

        node_inp = features_protein.predicted_embeddings
        node_mask = graph_protein.nodes_mask

        msg_input = jnp.concatenate(
            [
                node_inp[graph_protein.senders],
                node_inp[graph_protein.receivers],
                features_protein.original_edge_features,
                # x_rel_mag,
            ],
            axis=-1,
        )

        messages = hk.nets.MLP(
            [2 * dim, dim],
            activation=jax.nn.swish,
            name="edge_mlp",
        )(msg_input)

        agg_messages = (
            jax.ops.segment_sum(
                messages,
                graph_protein.receivers,
                num_segments=len(node_inp),
                indices_are_sorted=True,
                unique_indices=False,
                bucket_size=None,
                mode=None,
            )
            / self.graph_max_neighbor
        )

        agg_messages = MaskedLayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="norm_msg"
        )(agg_messages, node_mask)

        node_features = jnp.concatenate(
            [
                node_inp,
                agg_messages,
            ],
            axis=-1,
        )

        node_features = hk.nets.MLP(
            [2 * dim, dim],
            activation=jax.nn.swish,
            name="node_mlp",
        )(node_features)

        node_features = node_inp + node_features
        node_features = MaskedLayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="norm_node"
        )(node_features, node_mask)

        return node_features


class GATLayer(hk.Module):
    """Layer of the Gated Attention Network, but it attends to neighbouring nodes and
    edges."""

    def __init__(self, graph_max_neighbor: int, config):
        """
        Constructs the forward pass of the Gated Attention Network.

        Args:
            graph_max_neighbor (DataProcessingConfig): data processing configuration
            config (): config of the model
        """
        super(GATLayer, self).__init__()
        self.config = config
        self.graph_max_neighbor = graph_max_neighbor

    def __call__(
        self,
        graph_protein: ProteinGraph,
        features_protein: ProteinFeatures,
    ) -> NodeFeatures:
        """Computes a transformer inspired update where attention is performed of k
        nearest spatial neighbours.

        Args:
            graph_protein (ProteinGraph): graph of the protein of interest
            features_protein (ProteinFeatures): features of the protein of interest

        Returns:
            node_features (NodeFeatures): embeddings of residues
        """
        # [num_nodes, num_channels] activations for each node
        node_representation = features_protein.predicted_embeddings
        node_mask = jnp.squeeze(graph_protein.nodes_mask)
        num_nodes = graph_protein.nodes_mask.shape[0]
        num_neighbours = graph_protein.senders.shape[0] // num_nodes

        edge_feat = features_protein.original_edge_features
        assert edge_feat.ndim == 2
        assert node_representation.ndim == 2

        def mha_over_neighbours(node_embeddings, neighbour_nodes_and_edges):
            assert node_embeddings.ndim == 2 and neighbour_nodes_and_edges.ndim == 3
            stddev = 1.0 / np.sqrt(node_embeddings.shape[-1])

            def _wrap_mha(query, memory, mask=None):
                # mask is None because the neighbours are always full
                assert query.ndim == 2 and query.shape[0] == 1
                mha = hk.MultiHeadAttention(
                    self.config.num_heads,
                    self.config.key_size,
                    w_init=hk.initializers.TruncatedNormal(stddev=stddev),
                    value_size=self.config.value_size,
                    model_size=self.config.output_size,
                    name="gat_multi_head_attention",
                )
                return mha(query=query, key=memory, value=memory, mask=mask)

            vmha = jax.vmap(_wrap_mha, in_axes=(0, 0))
            out = vmha(node_embeddings[:, None, :], neighbour_nodes_and_edges)
            assert out.shape == (num_nodes, 1, self.config.output_size,), (
                f"out: {out.shape} num_nodes, "
                f"num_channels: {(num_nodes, self.config.output_size)}"
            )
            return jnp.squeeze(out, axis=1)

        def transition_block(act):
            channels = act.shape[-1]
            num_intermediate = int(channels * self.config.num_intermediate_factor)

            act = hk.LayerNorm(
                axis=[-1],
                create_scale=True,
                create_offset=True,
                name="input_layer_norm",
            )(act)

            transition_module = hk.Sequential(
                [
                    hk.Linear(num_intermediate, name="transition1"),
                    jax.nn.relu,
                    hk.Linear(channels, name="transition2"),
                ]
            )
            return transition_module(act)

        # layer norm the node representation
        node_representation = MaskedLayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name="norm_node_representation",
        )(node_representation, node_mask[:, None])

        # for each node, get the adjacent edge features and the neighbouring node
        # representations. [num_nodes, num_neighbours, node_channels+edge_feature_dim]
        neighbour_input = jnp.concatenate(
            [
                node_representation[graph_protein.senders].reshape(
                    num_nodes, num_neighbours, node_representation.shape[-1]
                ),
                edge_feat.reshape(num_nodes, num_neighbours, edge_feat.shape[-1]),
            ],
            axis=-1,
        )
        assert (
            neighbour_input.shape[:2] == (num_nodes, num_neighbours)
            and neighbour_input.ndim == 3
        ), (
            f"neighbour_input: {neighbour_input.shape}. "
            f"we want: {(num_nodes, num_neighbours)}"
        )

        print([node_representation.shape, neighbour_input.shape])
        # Note: allow each node i to attend to each of it's neighbours j \in
        # Neighbours(i) This is: attention(Q=node_i, K=neigh_i, V=neigh_i) where neigh_i
        # are edge features ij cat node rep j for j \in Neighbours(i) this is vmapped
        # over i (nodes) which is the leading dim of node_representation and
        # neighbour_input
        residual = mha_over_neighbours(node_representation, neighbour_input)
        residual *= node_mask[:, None]

        # In keeping with standard transformer architectures have a transition MLP with
        # layer-norm
        residual = transition_block(residual)
        residual *= node_mask[:, None]

        # apply dropout to the residual
        safe_key = SafeKey(hk.next_rng_key())
        safe_key, sub_key = safe_key.split()
        residual = hk.dropout(sub_key.get(), self.config.dropout_rate, residual)

        # residually apply the transformation to the node representation
        node_representation += residual

        return node_representation


class StateUpdate(hk.Module):
    """Graph Neural Network projected in multimodal BioCLIP space"""

    def __init__(self, Nh, Nk, Ns):
        """Class to compute multimodal embeddings for the structure and sequence

        Args:
            hyperparameters (BioClipConfig): model hyperparameters
            data_config (DataProcessingConfig): data processing configuration
        """

        super(StateUpdate, self).__init__(name="su")
        self.Nh = Nh
        self.Nk = Nk
        self.Ns = Ns

    def __call__(self, q, p, q_nn, p_nn, d_nn, r_nn):
        """

        ef = edge_features = 193
        nn = nearest-neighbours
        Nk = key-size = 3
        Ns = num-state = 32
        Nh = num-heads = 2

        q: (a, 32); p: (a, 3, 32); q_nn: (a, nn, 32); p_nn: (a, nn, 3, 32);
            d_nn: (a, nn); r_nn: (a, nn, 3)

        Zp:  (a, 3, Nh*Ns)
        qh:  (a, Ns)
        ph:  (a, 3, Ns)
        X_n: (a, 2*Ns)
        Q:   (a, 2, Nh, Nk)
        Zq:  (a, Nh*Ns)
        for nn in [8, 16, 32, 64]:
            q_nn: (a, nn, Ns)
            X_e:  (a, nn, ef)
            Kq:   (a, Nk, nn)
            Kp:   (a, Nk, 3*nn)
            V:    (a, 2, nn, Ns)
            Vp:   (a, 3, 3*nn, 32)
            Mq:   (a, 2, nn)
            Mp:   (a, 2, 3*nn)
            ...(x8)...
        """
        nqm = torch_mlp(
            [self.Ns, self.Ns, 2 * self.Nk * self.Nh], name="nqm"
        )  # "node_query_model")  # 2 * Ns <- input channels
        eqkm = torch_mlp(
            [self.Ns, self.Ns, self.Nk], name="eqkm"
        )  # "edges_scalar_keys_model")  # 6 * Ns + 1
        epkm = torch_mlp(
            [self.Ns, self.Ns, 3 * self.Nk], name="epkm"
        )  # "edges_vector_keys_model")  # 6 * Ns + 1
        evm = torch_mlp(
            [2 * self.Ns] * 3, name="evm"
        )  # "edges_value_model")  # 6 * Ns + 1
        qpm = torch_mlp(
            [self.Ns, self.Ns, self.Ns], name="qpm"
        )  # "scalar_projection_model")  # Nh * Ns
        ppm = torch_mlp(
            [self.Ns], name="ppm", with_bias=False
        )  # "vector_projection_model", with_bias=False)  # Nh * Ns

        sdk = np.sqrt(self.Nk)

        # Unpacking dimensions
        N, n, S = q_nn.shape
        # Node inputs packing
        X_n = jnp.concatenate([q, norm(p, axis=1)], axis=1)

        # Edge inputs packing
        X_e = jnp.concatenate(
            [
                d_nn[:, :, None],
                jnp.tile(X_n[:, None, :], (1, n, 1)),
                q_nn,
                norm(p_nn, axis=2),
                jnp.sum(p[:, None, :] * r_nn[:, :, :, None], axis=2),
                jnp.sum(p_nn * r_nn[:, :, :, None], axis=2),
            ],
            axis=2,
        )

        # Node queries
        Q = nqm(X_n).reshape((N, 2, self.Nh, self.Nk))

        # Scalar edges keys
        Kq = eqkm(X_e).reshape((N, n, self.Nk)).transpose((0, 2, 1))

        # Vector edges keys
        Kp = jnp.concatenate(jnp.split(epkm(X_e), self.Nk, axis=2), axis=1).transpose(
            (0, 2, 1)
        )

        # Edges values
        V = evm(X_e).reshape((N, n, 2, S)).transpose((0, 2, 1, 3))

        # Vectorial inputs packing
        Vp = jnp.concatenate(
            [
                V[:, 1, :, :][:, :, None, :] * r_nn[:, :, :, None],
                jnp.tile(p[:, None, :, :], (1, n, 1, 1)),
                p_nn,
            ],
            axis=1,
        ).transpose((0, 2, 1, 3))

        # Queries and keys collapse
        Mq = jax.nn.softmax(jnp.matmul(Q[:, 0], Kq) / sdk, axis=2)
        Mp = jax.nn.softmax(jnp.matmul(Q[:, 1], Kp) / sdk, axis=2)

        # Scalar state attention mask and values collapse
        Zq = jnp.matmul(Mq, V[:, 0]).reshape((N, self.Nh * self.Ns))
        Zp = jnp.matmul(Mp[:, None, :], Vp).reshape((N, 3, self.Nh * self.Ns))

        # Decode outputs
        qh = qpm(Zq)
        ph = ppm(Zp)

        # Update state with residual
        qz = q + qh
        pz = p + ph
        return qz, pz


class StateUpdateLayer(hk.Module):
    def __init__(self, layer_params, remat_policy):
        super(StateUpdateLayer, self).__init__(name="sum")
        # define operation
        self.su = StateUpdate(*[layer_params[k] for k in ["Nh", "Nk", "Ns"]])
        # store number of nearest neighbors
        self.m_nn = layer_params["nn"]
        self.remat_policy = remat_policy

    def __call__(self, Z):
        # unpack input
        q, p, ids_topk, D_topk, R_topk = Z

        # update q, p
        ids_nn = ids_topk[:, : self.m_nn]

        D_nn = D_topk[:, : self.m_nn]
        R_nn = R_topk[:, : self.m_nn]

        def remat_wrap(_q, _p):
            return self.su(_q, _p, _q[ids_nn], _p[ids_nn], D_nn, R_nn)

        if len(self.remat_policy):
            policy = getattr(jax.checkpoint_policies, self.remat_policy)
            fn = hk.remat(
                remat_wrap, policy=policy
            )  # jax.checkpoint_policies.checkpoint_dots)
        else:
            fn = hk.remat(remat_wrap)
        q, p = fn(q, p)

        # sink
        q = q.at[0].set(0.0)
        p = p.at[0].set(0.0)

        return q, p, ids_topk, D_topk, R_topk


class StatePoolLayer(hk.Module):
    def __init__(self, N0, N1, Nh):
        super(StatePoolLayer, self).__init__(name="spl")
        self.N0 = N0
        self.N1 = N1
        self.Nh = Nh

    def __call__(self, q, p, atom_mask):
        r = atom_mask.shape[0]

        # pack features
        z = jnp.concatenate([q, norm(p, axis=1)], axis=1)

        # multiple attention pool on state
        sam_out = torch_mlp([self.N0, self.N0, 2 * self.Nh], name="sam")(z)

        sa = sam_out.reshape(*atom_mask.shape, *sam_out.shape[1:])
        p_ = p.reshape(*atom_mask.shape, *p.shape[1:])
        q_ = q.reshape(*atom_mask.shape, *q.shape[1:])

        # LR: I have replaced the expensive [num_atoms, num_residues] cross attention
        # with per-residue attention over atom14 which is much cheaper but functionally
        # equivalent.
        def attn(si, pi, qi, a14):
            logit_mask_ = (1.0 - a14 + 1e-6) / (a14 - 1e-6)
            aw = jax.nn.softmax(si + logit_mask_[:, None], axis=0)
            aw = aw.reshape(a14.shape[0], -1, 2)  # (14, 4, 2)
            _qh = jnp.einsum("ac,ah->ch", qi, aw[..., 0])
            _ph = jnp.einsum("axc,ah->xch", pi, aw[..., 1])
            return _qh, _ph

        qh, ph = jax.vmap(attn)(sa, p_, q_, atom_mask)

        # attention heads decoding
        x = p.shape[1]
        qh_flat = qh.reshape(r, -1)  # [r, c*h]
        ph_flat = ph.reshape(r, x, -1)  # [r, 3, c*h]

        res_mask = atom_mask.any(-1).astype(ph_flat.dtype)
        qh_flat *= res_mask[:, None]
        ph_flat *= res_mask[:, None, None]

        qr = torch_mlp([self.N0, self.N0, self.N1], name="zdm")(qh_flat)
        pr = torch_mlp([self.N1], name="zdm_vec", with_bias=False)(ph_flat)
        return qr, pr


def unpack_state_features(X, ids_topk, q):
    # compute displacement vectors
    R_nn = X[ids_topk - 1] - X[:, None]
    # compute distance matrix
    D_nn = norm(R_nn, axis=2)
    # mask distances
    D_nn = D_nn + jnp.max(D_nn) * (D_nn < 1e-2)
    # normalize displacement vectors
    R_nn = R_nn / D_nn[:, :, None]

    # prepare sink
    q = jnp.concatenate([jnp.zeros((1, q.shape[1])), q], axis=0)
    ids_topk = jnp.concatenate(
        [jnp.zeros((1, ids_topk.shape[1]), dtype=jnp.int64), ids_topk], axis=0
    )
    D_nn = jnp.concatenate([jnp.zeros((1, D_nn.shape[1])), D_nn], axis=0)
    R_nn = jnp.concatenate([jnp.zeros((1, R_nn.shape[1], R_nn.shape[2])), R_nn], axis=0)
    return q, ids_topk, D_nn, R_nn


class GraphTransformerModel(hk.Module):
    def __init__(self, config, remat_policy):
        super(GraphTransformerModel, self).__init__()
        self.config = config
        self.remat_policy = remat_policy

    def __call__(self, X, ids_topk, q0, atom_mask):
        # "encode_structure")  # config['em']['N0'] <- input channels
        # encode features
        q = torch_mlp([self.config["em"]["N1"]] * 3, name="em")(q0)

        # initial state vectors
        p0 = jnp.zeros((q.shape[0] + 1, X.shape[1], q.shape[1]))

        # unpack state features with sink
        q, ids_topk, D_nn, R_nn = unpack_state_features(X, ids_topk, q)

        # atomic tsa layers
        # atomic level state update model
        qa, pa, *_ = hk.Sequential(
            [
                StateUpdateLayer(layer_params, self.remat_policy)
                for i, layer_params in enumerate(self.config["sum"])
            ]
        )((q, p0, ids_topk, D_nn, R_nn))

        spl = StatePoolLayer(
            self.config["spl"]["N0"], self.config["spl"]["N1"], self.config["spl"]["Nh"]
        )  # atomic to residue reduction layer
        # atomic to residue attention pool (without sink)
        qr, pr = spl(qa[1:], pa[1:], atom_mask)

        res_mask = atom_mask.any(-1)

        # decode state
        zr = jnp.concatenate([qr, norm(pr, axis=1)], axis=1)
        dm = torch_mlp(
            [self.config["dm"]["N1"]] * 2 + [self.config["dm"]["N2"]], name="dm"
        )  # "decode_structure")  # 2*config['dm']['N0'] <- input channels
        z = dm(zr) * res_mask[:, None]

        return z
