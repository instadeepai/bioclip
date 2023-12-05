import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jmp

from bio_clip.model.esm.esm_forward import prep_esm_wrapper
from bio_clip.model.gnn import ForwardBioClip


def sigmoid_binary_cross_entropy(logits, labels, pos_weight):
    # LR: copied from optax, but added pytorch pos_weight functionality
    chex.assert_type([logits], float)
    labels = labels.astype(logits.dtype)
    log_p = jax.nn.log_sigmoid(logits)
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
    log_not_p = jax.nn.log_sigmoid(-logits)
    pos_weight = pos_weight[tuple([None] * (labels.ndim - 1) + [slice(None)])]
    return -labels * log_p * pos_weight - (1.0 - labels) * log_not_p


class DownstreamModel(hk.Module):
    def __init__(self, gnn_config, aux_config, n_classes, esm_forward, architecture):
        super(DownstreamModel, self).__init__(name="fnpr_downstream_model")
        self.gnn_config = gnn_config
        self.aux_config = aux_config
        self.n_classes = n_classes
        self.esm_forward = esm_forward
        self.architecture = architecture

    def __call__(self, batch_data, compute_loss):
        batch_data, target, loss_vars = batch_data
        # Use mixed precision (only support A100 GPU and TPU for now)
        half = jnp.bfloat16
        full = jnp.float32

        policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)
        hk.mixed_precision.set_policy(ForwardBioClip, policy)

        # Remove it in batch norm to avoid instabilities
        policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=half)
        hk.mixed_precision.set_policy(hk.BatchNorm, policy)
        hk.mixed_precision.set_policy(hk.LayerNorm, policy)

        # TODO: LR: modify the forward pass to NOT aggregate the per-residue embeddings
        forward = ForwardBioClip(
            feature_config=self.gnn_config.features,
            model_config=self.gnn_config.model,
            data_config=self.gnn_config.training.data,
            training_config=self.gnn_config.training,
            use_seq=False,
            gnn_layer_to_use=self.aux_config.gnn_layer_to_use,
            target_type="per-res",
        )
        num_res = batch_data.tokens.shape[0]
        if self.architecture == "graph_transformer":
            *_, atom14_mask = batch_data.graph
            mask = jnp.squeeze(atom14_mask.any(-1))[
                :, None
            ]  # un-squeeze to match the legacy bioclip code.
        else:
            mask = batch_data.graph.nodes_mask

        if self.aux_config.use_gnn_embedding:
            structure_embedding = forward(batch_data).structure_embedding

        if self.aux_config.use_esm_embedding:
            if self.aux_config.tune_esm:

                forward_esm_model = prep_esm_wrapper(
                    self.esm_forward,
                    num_res,
                    max_batch_size=self.aux_config.batch_size_esm_per_device,
                    embeddings_layer_to_save=self.aux_config.embeddings_layer_to_save,
                    compute_mean_embedding=self.aux_config.compute_mean_embedding,
                    pad_token_id=self.aux_config.pad_token_id,
                )

                if len(batch_data.tokens.shape) == 1:
                    tokens = batch_data.tokens[None]
                batch_esm_embeddings = forward_esm_model(tokens)
                assert self.aux_config.batch_size_esm_per_device == 1, ""
                batch_esm_embeddings = batch_esm_embeddings[0]
            else:
                print(f"Using pre-computed ESM embeddings.")
                raise Exception(
                    "~LR I have not pre-computed ESM for this task yet. Just turn confi"
                    "g.training.tune_esm=True and config.training.train_esm_from=100"
                )
            if self.aux_config.use_gnn_embedding:
                print(
                    f"CONCAT: {structure_embedding.shape}; {batch_esm_embeddings.shape}"
                )
                protein_embedding = jnp.concatenate(
                    [structure_embedding, batch_esm_embeddings], axis=-1
                )
            else:
                print(
                    "Not using the structure embedding. Only using the ESM embedding."
                )
                protein_embedding = batch_esm_embeddings
        else:
            print("Not using ESM embedding.")
            if self.aux_config.use_gnn_embedding:
                protein_embedding = structure_embedding
            else:
                raise Exception("Must use structure or sequence embedding.")

        h = protein_embedding.shape[-1] + self.n_classes
        hidden_dims = [h] * self.gnn_config.model.final_hidden_layers + [self.n_classes]
        prediction = hk.nets.MLP(
            hidden_dims,
            activation=jax.nn.relu,
            activate_final=False,
            name="classification_mlp",
        )(protein_embedding)
        pred_prob_class_1 = jax.nn.sigmoid(prediction).astype(jnp.float32)

        if compute_loss:
            assert (
                target.ndim == 2 and target.shape[-1] == 5
            ), f"Invalid target shape: {target.shape}"

            pos_ratios = loss_vars["pos_ratios"]
            global_step = loss_vars["global_step"]
            pos_weight_factor = loss_vars["pos_weight_factor"]

            new_est = (target * jnp.squeeze(mask)[:, None]).sum(0) / mask.sum()
            pos_ratios += (new_est - pos_ratios) / (1.0 + (global_step**0.5))
            pos_weight = pos_weight_factor * (1.0 - pos_ratios) / (pos_ratios + 1e-6)

            dloss = sigmoid_binary_cross_entropy(prediction, target, pos_weight)
            # dloss: [num_res, num_class]

            # re-weighted losses
            loss_factors = pos_ratios / jnp.sum(pos_ratios)
            losses = loss_factors[None, :] * dloss

            loss = jnp.mean(losses.sum(axis=-1), where=jnp.squeeze(mask))

            auxiliary_data = {
                "metrics": (loss,),
                "prediction": pred_prob_class_1,
                "pos_ratios": pos_ratios,
            }
            return loss, auxiliary_data
        else:
            return pred_prob_class_1
