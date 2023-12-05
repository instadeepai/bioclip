import haiku as hk
import jax
import jax.numpy as jnp
import jax.profiler
import jmp
import optax

from bio_clip.model.esm.esm_forward import prep_esm_wrapper
from bio_clip.model.gnn import ForwardBioClip


class DownstreamModel(hk.Module):
    def __init__(self, gnn_config, aux_config, esm_forward):
        super(DownstreamModel, self).__init__(name="ppi_downstream_model")
        self.gnn_config = gnn_config
        self.aux_config = aux_config
        self.esm_forward = esm_forward

    def __call__(self, batch_data, compute_loss):
        # Use mixed precision (only support A100 GPU and TPU for now)
        half = jnp.bfloat16
        full = jnp.float32
        # Remove it in batch norm to avoid instabilities
        policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=half)
        hk.mixed_precision.set_policy(hk.BatchNorm, policy)
        hk.mixed_precision.set_policy(hk.LayerNorm, policy)

        # gnn_hidden = self.gnn_config.model.gnn.gnn_layer.hidden_dimension

        num_res = batch_data.batch_data_bioclip.sequence.shape[0]

        if self.aux_config.use_gnn_embedding:
            policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)
            hk.mixed_precision.set_policy(ForwardBioClip, policy)

            forward = ForwardBioClip(
                feature_config=self.gnn_config.features,
                model_config=self.gnn_config.model,
                data_config=self.gnn_config.training.data,
                training_config=self.gnn_config.training,
                use_seq=False,
                gnn_layer_to_use=self.aux_config.gnn_layer_to_use,
                target_type="per-chain",
            )
            out1 = forward(batch_data.batch_data_bioclip)
            out2 = forward(batch_data.batch_data_bioclip_2)
            structure_embedding = jnp.concatenate(
                [out1.structure_embedding, out2.structure_embedding], axis=-1
            )

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
                batch_esm_embeddings = forward_esm_model(
                    batch_data.batch_data_bioclip.sequence
                )
                batch_esm_embeddings_2 = forward_esm_model(
                    batch_data.batch_data_bioclip_2.sequence
                )
                assert self.aux_config.batch_size_esm_per_device == 1, ""
                batch_esm_embeddings = jnp.concatenate(
                    [batch_esm_embeddings[0], batch_esm_embeddings_2[0]], axis=-1
                )
            else:
                print(f"Using pre-computed ESM embeddings.")
                batch_esm_embeddings = jnp.concatenate(
                    [
                        batch_data.batch_data_bioclip.sequence,
                        batch_data.batch_data_bioclip_2.sequence,
                    ],
                    axis=-1,
                )
            if self.aux_config.use_gnn_embedding:
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

        n_classes = 1
        h = protein_embedding.shape[-1] + n_classes
        hidden_dims = [h] * self.gnn_config.model.final_hidden_layers + [n_classes]
        prediction = hk.nets.MLP(
            hidden_dims,
            activation=jax.nn.relu,
            activate_final=False,
            name="classification_mlp",
        )(protein_embedding)

        pred_prob_class_1 = jax.nn.sigmoid(prediction)

        if compute_loss:
            # target = jax.nn.one_hot(batch_data.target, 2)
            # loss = jnp.mean(optax.softmax_cross_entropy(prediction, target))
            loss = jnp.mean(
                optax.sigmoid_binary_cross_entropy(prediction, batch_data.target)
            )
            auxiliary_data = {
                "metrics": (loss,),
                "prediction": pred_prob_class_1,
            }
            return loss, auxiliary_data
        else:
            return pred_prob_class_1
