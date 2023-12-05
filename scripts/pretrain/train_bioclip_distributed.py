# DEBUG = False
# # if DEBUG:
# #     from jax import config
# #     config.update("jax_debug_nans", True)

import copy
import functools
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from multiprocessing import set_start_method
from typing import List

import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from cloudpathlib import AnyPath

from bio_clip.data.protein_datastructure import ProteinStructureSample
from bio_clip.model.esm.esm_forward import forward_esm_model
from bio_clip.model.esm.esm_haiku import get_pretrained_esm_model
from bio_clip.model.gnn import ForwardBioClip
from bio_clip.train.data_transforms import preprocess_atoms, preprocess_sample
from bio_clip.train.pretrain.contrastive_loss import loss_bioclip_distributed_fn
from bio_clip.train.pretrain.dataloader import (
    BioClipDataloader,
    get_filepaths_to_samples,
)
from bio_clip.train.pretrain.trainer import Trainer, TrainingState
from bio_clip.types import BatchDataBioClip
from bio_clip.utils.checkpointing import Checkpointer, load_pretrained_checkpoint_config
from bio_clip.utils.neptune import NeptunePlaceholder
from bio_clip.utils.utils import (
    ThreadedIterable,
    convert_to_ml_dict,
    get_logger,
    show_git,
    tmpdir_manager,
)

logger = get_logger(__name__)


DEBUG = False
TPU_NODES = 1
OVERRIDE_BACKEND = "cpu"


def load_config(name, job_name, config_path="../config"):
    with hydra.initialize(config_path=config_path, job_name=job_name):
        config = hydra.compose(config_name=name, overrides=[])
        return convert_to_ml_dict(config)


def filter_out_sample(
    sample: ProteinStructureSample,
    min_number_valid_residues: int,
    max_number_residues: int,
) -> bool:
    missing_coords_residue_mask = sample.get_missing_backbone_coords_mask()
    num_residue_coords_known = np.sum(~missing_coords_residue_mask)
    return bool(
        num_residue_coords_known < min_number_valid_residues
        or sample.nb_residues > max_number_residues
    )


def filter_out_sample_graph_transformer(sample, min_size, max_size):
    num_res = sample.atom37_gt_exists.shape[0]
    resolved = sample.atom37_gt_exists.any(-1).sum()
    return bool(resolved < min_size or num_res > max_size)


def prepare_neptune(cfg):
    neptune_run = None
    # Prepare Neptune run.
    if jax.process_index() == 0:
        nep_cfg = cfg.training.neptune
        tags = nep_cfg.tags
        tags += [
            "final_run",
            f"architecture_{cfg.model.architecture}",
            f"{cfg.model.gnn.gnn_layer.hidden_dimension}-gnn-hidden",
            f"{cfg.model.gnn.gnn_number_layers}-gnn-layers",
            f"{cfg.training.data.fixed_sizes.graph_max_neighbor}-graph_max_neighbor",
        ]
        if cfg.training.direct_supervision:
            tags.append("direct_supervision")
        if cfg.training.train_esm_layers:
            number_of_transformer_blocks = int(
                (cfg.model.plm.esm_model_name).split("_")[1].replace("t", "")
            )
            train_esm_from = cfg.training.train_esm_from
            if train_esm_from < 0:
                train_esm_from = number_of_transformer_blocks + train_esm_from
            tags.append(
                f"training-esm-from-{train_esm_from}-to-{number_of_transformer_blocks}"
            )
        else:
            tags.append("fixing-esm")

        tags.append(f"target_type_{cfg.training.target_type}")
        tags.append(f"objective_type_{cfg.training.objective_type}")

        if cfg.features.use_positional_features:
            tags.append("positional_features")

        cfgdl = cfg.training.data.dataloader
        if cfgdl.conditional_cluster_sampling.active:
            n_seq = cfgdl.conditional_cluster_sampling.take_n_seq_per_cluster
            tags += ["conditional-cluster-sampling", f"{n_seq}-within-cluster"]
        elif cfgdl.use_weighted_sampling:
            tags.append("weighted_sampling")
        else:
            tags.append("uniform_sampling")

        neptune_run = NeptunePlaceholder(
            project=nep_cfg.project_name,
            name=nep_cfg.experiment_name,
            tags=tags,
            config=cfg.to_dict(),
            resume=bool(len(cfg.training.checkpoints.resume_from)),
        )
    return neptune_run


class RunTrainer:
    @staticmethod
    def prepare_devices(cfg):
        ###
        # Check devices
        ###
        backend = OVERRIDE_BACKEND if len(OVERRIDE_BACKEND) else cfg.training.backend

        # TMP HACK
        cfg.training.batch_size = 1
        cfg.training.batch_size_gnn_per_device = 1
        cfg.training.batch_size_esm_per_device = 1

        global_device_count = jax.device_count(backend=backend)
        local_device_count = jax.local_device_count(backend=backend)

        devices = jax.devices(backend=backend)
        local_devices = jax.local_devices(backend=backend)

        print(
            f"Configured to run on backend {cfg.training.backend}..."
            f"found {local_device_count}/{global_device_count} local/global devices."
        )

        assert (cfg.training.batch_size % global_device_count) == 0, (
            f"Batch size {cfg.training.batch_size} must be divisible by"
            f" global device count ({global_device_count})"
        )
        batch_size_per_device = cfg.training.batch_size // global_device_count

        bs_gnn_per_device = cfg.training.batch_size_gnn_per_device
        bs_esm_per_device = cfg.training.batch_size_esm_per_device
        if batch_size_per_device > bs_gnn_per_device:
            assert (batch_size_per_device % bs_gnn_per_device) == 0, (
                f"Total batch size per device {batch_size_per_device} must be divisible"
                f" by max GNN batch size per device ({bs_gnn_per_device})"
            )
        else:
            cfg.training.batch_size_gnn_per_device = batch_size_per_device

        if batch_size_per_device > bs_esm_per_device:
            assert (batch_size_per_device % bs_esm_per_device) == 0, (
                f"Total batch size per device {batch_size_per_device} must be divisible"
                f" by max GNN batch size per device ({bs_esm_per_device})"
            )
        else:
            cfg.training.batch_size_esm_per_device = batch_size_per_device

        logger.info(
            f"Total batch size {cfg.training.batch_size} will require "
            f"{batch_size_per_device} samples per device."
        )
        return (
            cfg,
            devices,
            local_devices,
            global_device_count,
            local_device_count,
            batch_size_per_device,
        )

    @staticmethod
    def prepare_optimiser(cfg):
        # Optimizer - should be moved into trainer...
        learning_rate = cfg.training.optimiser.learning_rate

        if cfg.training.optimiser.optimiser_type == "adamw":
            optimizer = optax.adamw(
                learning_rate=learning_rate,
                weight_decay=cfg.training.optimiser.weight_decay,
            )
        elif cfg.training.optimiser.optimiser_type == "adam":
            optimizer = optax.adam(learning_rate=learning_rate)
        else:
            raise NotImplementedError()
        optimizer = optax.chain(
            optax.scale_by_adam(b1=0.9, b2=0.98),
            optax.scale_by_schedule(lambda step: learning_rate),
            optax.clip_by_global_norm(1.0),
            optax.scale(-1.0),
        )
        return optimizer

    @staticmethod
    def prepare_jitted_fns(
        cfg,
        devices,
        local_devices,
        optimizer,
    ):
        ###
        # Set up ESM-model
        ###
        number_of_transformer_blocks = int(
            (cfg.model.plm.esm_model_name).split("_")[1].replace("t", "")
        )
        embeddings_layer_to_save = int(
            cfg.training.proportion_esm_layer * number_of_transformer_blocks
        )
        (esm_parameters, forward_esm_fn, tokenizer,) = get_pretrained_esm_model(
            cfg.model.plm.esm_model_name,
            embeddings_layers_to_save=[embeddings_layer_to_save],
        )
        esm_init = hk.transform(forward_esm_fn).init
        forward_esm_fn = hk.transform(forward_esm_fn).apply

        def _forward_bioclip_fn(x):
            return ForwardBioClip(
                feature_config=cfg.features,
                model_config=cfg.model,
                data_config=cfg.training.data,
                training_config=cfg.training,
                target_type=cfg.training.target_type,
            )(x)

        forward_bioclip_fn = hk.transform(_forward_bioclip_fn)
        # esm_parameters = jax.device_put_replicated(esm_parameters, local_devices)
        esm_fwd = functools.partial(
            forward_esm_model,
            max_batch_size=cfg.training.batch_size_esm_per_device,
            embeddings_layer_to_save=embeddings_layer_to_save,
            compute_mean_embedding=False,
            forward_esm_fn=forward_esm_fn,
            pad_token_id=tokenizer.pad_token_id,
        )
        if cfg.training.train_esm_layers:
            train_esm_from = cfg.training.train_esm_from
            if train_esm_from < 0:
                train_esm_from = number_of_transformer_blocks + train_esm_from - 1

            def parameters_partition_fn(
                module_name: str,
                _name: str,
                _value: jnp.array,
            ):
                trainable = False
                if "attention_layer_" in module_name:
                    layer = int(module_name.split("attention_layer_")[-1].split("/")[0])
                    trainable = layer > train_esm_from
                if "forward_bio_clip" in module_name:
                    trainable = True
                return trainable

        else:

            def parameters_partition_fn(
                module_name: str,
                _name: str,
                _value: jnp.array,
            ):
                return "forward_bio_clip" in module_name

        # Trainer
        loss_fn = functools.partial(
            loss_bioclip_distributed_fn,
            forward_fn=forward_bioclip_fn.apply,
            train_cfg=cfg.training,
            esm_forward=esm_fwd,
            gnn_type=cfg.model.architecture,
            turn_off_pmap=DEBUG,
        )
        trainer = Trainer(
            esm_init,
            forward_bioclip_fn.init,
            loss_fn,
            optimizer,
            optimizer_type=cfg.training.optimiser.optimiser_type,
            parameters_partition_fn=parameters_partition_fn,
            turn_off_pmap=DEBUG,
        )

        # Distribute functions
        if DEBUG:

            def pmap_update(state, batch):
                return trainer.update(state, jax.tree_map(lambda x: x[0], batch))

            def pmap_predict(k, p, b):
                return trainer.predict(*jax.tree_map(lambda x: x[0], (k, p, b)))

        else:
            pmap_update = jax.pmap(
                trainer.update, axis_name="p", devices=devices, donate_argnums=(0,)
            )
            pmap_predict = jax.pmap(trainer.predict, axis_name="p", devices=devices)

        # pmap_forward_esm = jax.pmap(esm_fwd, devices=devices)
        # trainer_init = partial(trainer.init, esm_params=esm_parameters)
        return (
            trainer.init,
            tokenizer,
            esm_parameters,
            pmap_update,
            pmap_predict,
            parameters_partition_fn,
        )

    @staticmethod
    def prepare_data(
        random_key,
        global_device_count,
        local_device_count,
        cfg,
        batch_size_per_device,
        tokenizer,
    ):
        s3_sess = None  # create_s3_session(cfg.training.data.aws_endpoint)
        # s3_sess = create_s3_session(cfg.training.data.aws_endpoint)

        def get_filepaths_for_dataloader(
            fasta_path: AnyPath,
            num_repeats: int = 1,
            shuffle: bool = False,
            key: jax.random.PRNGKey = None,
        ) -> List:
            filepaths = get_filepaths_to_samples(
                fasta_path, are_pdb_samples=True, s3_sess=s3_sess
            )
            logger.info(f"Found {len(filepaths)} total samples.")
            # This assumes that all devices are self.filepaths in the same order!
            # If loaded from get_train_val_filepaths_gcp(...) this means either running
            # with get_train_val_filepaths_gcp(..., shuffle=False) or a fixed shuffle
            # seed on all devices!
            total_processes = global_device_count // local_device_count
            step = len(filepaths) // total_processes
            loc_id = jax.process_index()
            # Select the subset of filepaths for the local processes.
            filepaths = filepaths[step * loc_id : step * (loc_id + 1)]
            # Repeat filepaths requested number of times.
            repeated_filepaths = []
            for _ in range(num_repeats):
                if shuffle:
                    key, shuffle_key = jax.random.split(key)
                    repeated_filepaths.append(
                        # TODO: have shuffle controlled by a jax key?
                        # jax.random.shuffle(shuffle_key, filepaths)
                        random.sample(filepaths, len(filepaths))
                    )
                else:
                    repeated_filepaths.append(filepaths)
            # filepaths = list(jnp.concatenate(repeated_filepaths))
            filepaths = [item for sublist in repeated_filepaths for item in sublist]
            return filepaths

        if cfg.training.data.dataloader.use_weighted_sampling:
            if cfg.training.data.pre_shuffle_training_set:
                logger.info(
                    "Turning off pre-shuffling as cfg.training.data.dataloader."
                    "use_weighted_sampling = True"
                )
            pre_shuffle_train = False
        else:
            pre_shuffle_train = cfg.training.data.pre_shuffle_training_set

        random_key, k1, k2 = jax.random.split(random_key, 3)
        # LR: pre-filtered validation data to ensure the dataloaders sync up
        all_filepaths = {
            "train": get_filepaths_for_dataloader(
                fasta_path=AnyPath(cfg.training.data.train_set_path),
                num_repeats=cfg.training.num_epochs,
                shuffle=pre_shuffle_train,
                key=k1,
            ),
            "unif": get_filepaths_for_dataloader(
                fasta_path=AnyPath(cfg.training.data.val_set_path),
                num_repeats=1,
                shuffle=False,
                key=k2,
            ),
            "clust": get_filepaths_for_dataloader(
                fasta_path=AnyPath(cfg.training.data.val_set_clust_path),
                num_repeats=1,
                shuffle=False,
                key=k2,
            ),
        }

        def get_code(filepath):
            return filepath.split("/")[4]

        if cfg.training.data.dataloader.use_weighted_sampling:
            with tmpdir_manager(base_dir="/tmp") as tmp_dir:
                local_filepath = os.path.join(tmp_dir, "weights.json")
                AnyPath(cfg.training.data.cluster_size_path).download_to(local_filepath)
                with open(local_filepath, "r") as file_to_read:
                    code2cluster_size = json.loads(file_to_read.read())

            def get_weights(filepaths):
                cluster_sizes = np.array(
                    [code2cluster_size[get_code(fp)] for fp in filepaths], np.float64
                )
                weights = 1 / cluster_sizes
                return weights

            sampling_weights = {k: get_weights(v) for k, v in all_filepaths.items()}
        else:
            sampling_weights = {}

        if cfg.training.data.dataloader.conditional_cluster_sampling.active:
            ccs_cfg = cfg.training.data.dataloader.conditional_cluster_sampling
            # load the clusters
            with tmpdir_manager(base_dir="/tmp") as tmp_dir:
                local_filepath = os.path.join(tmp_dir, "clusters.json")
                AnyPath(ccs_cfg.cluster_map_filepath).download_to(local_filepath)
                with open(local_filepath, "r") as file_to_read:
                    text = file_to_read.read()
            pairs = [row.split("\t") for row in text.split("\n")]
            cluster2codes = defaultdict(list)
            code2cluster = {}
            for c, s in pairs[:-1]:
                cluster2codes[c].append(s)
                code2cluster[s] = c

            take_n_seq_per_cluster = ccs_cfg.take_n_seq_per_cluster
            code2cluster = {
                c: cluster for cluster, codes in cluster2codes.items() for c in codes
            }

            conditional_cluster_sampling_args = {}
            for split, file_paths in all_filepaths.items():
                codes2fp = {get_code(fp): fp for fp in file_paths}
                clusters = list(set(code2cluster.values()))
                cluster_sizes = np.array([len(cluster2codes[c]) for c in clusters])
                cluster_weights = cluster_sizes**0.5
                cluster2filepaths = {
                    c: [
                        codes2fp[code] for code in cluster2codes[c] if code in codes2fp
                    ]  # only take codes in the split
                    for c in clusters
                }
                conditional_cluster_sampling_args[split] = (
                    clusters,
                    cluster_weights,
                    cluster2filepaths,
                    take_n_seq_per_cluster,
                )
        else:
            conditional_cluster_sampling_args = {}

        fixes_sizes = cfg.training.data.fixed_sizes
        if cfg.model.architecture == "graph_transformer":
            preprocess_sample_fn = functools.partial(
                preprocess_atoms,
                tokenizer=tokenizer,
                num_neighbor=fixes_sizes.graph_max_neighbor,
                padding_num_residue=fixes_sizes.maximum_padding,
            )
        elif cfg.model.architecture == "gnn":
            res_alphac = cfg.training.data.datatransforms.graph_residue_loc_is_alphac
            preprocess_sample_fn = functools.partial(
                preprocess_sample,
                tokenizer=tokenizer,
                num_neighbor=fixes_sizes.graph_max_neighbor,
                residue_loc_is_alphac=res_alphac,
                padding_num_residue=fixes_sizes.maximum_padding,
            )
        elif cfg.model.architecture == "evoformer":
            raise NotImplementedError
        else:
            raise Exception(
                f"architecture: {cfg.model.architecture} not in ['gnn', 'evoformer']"
            )

        # Max residue has -2 to account for beginning and end-of-sequence token,
        max_number_residues = fixes_sizes.maximum_padding - 2
        if cfg.model.architecture == "graph_transformer":
            filter_out_sample_fn = functools.partial(
                filter_out_sample_graph_transformer,
                min_size=fixes_sizes.graph_max_neighbor,  # fixes_sizes.minimum_padding
                max_size=max_number_residues,
            )
        else:
            filter_out_sample_fn = functools.partial(
                filter_out_sample,
                min_number_valid_residues=fixes_sizes.graph_max_neighbor,
                max_number_residues=max_number_residues,
            )

        def get_dataloader(data_split: str = "train") -> BioClipDataloader:
            dataloader_cfg = copy.copy(cfg.training.data.dataloader)
            dataloader_cfg.batch_dims = (local_device_count, batch_size_per_device)
            dataloader_cfg.filepaths = all_filepaths[data_split]
            dataloader_cfg.weights = sampling_weights.get(data_split, None)
            dataloader_cfg.conditional_cluster_sampling_args = (
                conditional_cluster_sampling_args.get(data_split, None)
            )
            dataloader_cfg.aws_endpoint = cfg.training.data.aws_endpoint
            return BioClipDataloader(
                dataloader_cfg, preprocess_sample_fn, filter_out_sample_fn
            )

        return random_key, get_dataloader

    @staticmethod
    def run(
        random_key,
        parameters_partition_fn,
        cfg,
        resume_path,
        global_device_count,
        local_devices,
        local_device_count,
        batch_size_per_device,
        trainer_init,
        get_dataloader,
        esm_parameters,
        pmap_update,
        pmap_predict,
    ):
        neptune_run = prepare_neptune(cfg)
        if jax.process_index() == 0:
            run_id = neptune_run.run_id
            description = "_".join(list(set(cfg.training.neptune.tags)))
            checkpointer = Checkpointer(
                cfg.training.checkpoints,
                run_id,
                description,
                parameters_partition_fn,
                resume_from=resume_path,
            )
        elif resume_path:  # multi-node TPU, other processes only load checkpoints!
            checkpointer = Checkpointer(
                cfg.training.checkpoints,
                "",
                "",
                parameters_partition_fn,
                resume_from=resume_path,
            )

        def init_training_state(data_batch, key: jax.random.PRNGKey) -> TrainingState:
            logger.info("Initializing the model")
            # Only need one sample per device to init ([dev, batch, ...]-->[dev, ...]).
            gnn_data_batch = BatchDataBioClip(
                graph=jax.tree_map(lambda x: x[:, 0], data_batch).graph,
                sequence=jnp.zeros(
                    (
                        local_device_count,
                        cfg.training.data.fixed_sizes.maximum_padding,
                        cfg.training.esm_embedding_size,
                    )
                ),
            )
            if DEBUG:
                training_state = trainer_init(
                    key, esm_parameters, jax.tree_map(lambda x: x[0], gnn_data_batch)
                )
            else:
                # Distribute identical keys across all devices.
                key = jax.device_put_replicated(key, local_devices)
                esm_parameters_dist = jax.device_put_replicated(
                    esm_parameters, local_devices
                )
                _print_info = jax.tree_map(
                    lambda a: a.shape, (key, esm_parameters_dist, gnn_data_batch)
                )
                print(f"jax.pmap(trainer_init): {_print_info}")
                training_state = jax.pmap(
                    trainer_init, devices=local_devices, axis_name="p"
                )(
                    key,
                    esm_parameters_dist,
                    gnn_data_batch,
                )
                param_count = float(
                    sum(x[0].size for x in jax.tree_leaves(training_state.params))
                )

                param_count_structure_encoder = float(
                    sum(
                        x[0].size
                        for x in jax.tree_leaves(
                            {
                                k: v
                                for k, v in training_state.params.items()
                                if k.startswith("forward_bio_clip")
                            }
                        )
                    )
                )
                logger.info(f"Done initializing: model has {param_count:.2E}M params.")
                logger.info(
                    f"structure enc: {int(param_count_structure_encoder)} params."
                )
            return training_state

        def run_validation(
            training_state: TrainingState, key: jax.random.PRNGKey, val_name: str
        ) -> float:
            all_val_loss = []
            with get_dataloader(data_split=val_name) as dataloader:
                start_time_get_batch = time.perf_counter()
                for batch_id, data_batch in enumerate(ThreadedIterable(dataloader)):
                    runtime_get_batch = time.perf_counter() - start_time_get_batch
                    logger.info(f"Fetched batch in {runtime_get_batch} seconds.")

                    key, key_data, key_val = jax.random.split(key, 3)
                    # data_batch = prepare_batch(data_batch, key_data)

                    logger.info("Running predict")
                    start_time_predict = time.perf_counter()
                    key_data = jax.random.split(key_data, local_device_count)
                    loss, metrics = pmap_predict(
                        key_data, training_state.params, data_batch
                    )
                    _ = jax.tree_map(lambda x: x.block_until_ready(), metrics)
                    runtime_predict = time.perf_counter() - start_time_predict
                    logger.info(f"Done running predict (took {runtime_predict} secs)")

                    all_val_loss.append(float(jnp.mean(loss)))
                    start_time_get_batch = time.perf_counter()

            return float(np.mean(all_val_loss))

        step_id = 1
        is_initialized = False

        cfg.training.validation_freq = 5000

        with get_dataloader() as dataloader:

            start_time_get_batch = time.perf_counter()
            for batch_id, data_batch in enumerate(ThreadedIterable(dataloader)):
                runtime_get_batch = time.perf_counter() - start_time_get_batch
                logger.info(f"Fetched batch in {runtime_get_batch} seconds.")

                # Run ESM model to prepare batch
                # random_key, k = jax.random.split(random_key)
                # start_time_forward_esm = time.perf_counter()
                # # data_batch = prepare_batch(data_batch, k)
                # runtime_forward_esm = time.perf_counter() - start_time_forward_esm
                # logger.info(f"Ran esm model in {runtime_forward_esm} secs.")

                if not is_initialized:
                    is_initialized = True
                    # Initialise model parameters.
                    random_key, k = jax.random.split(random_key)
                    if len(cfg.training.checkpoints.resume_from):
                        _, fixed_params = hk.data_structures.partition(
                            checkpointer.partition_fn, esm_parameters
                        )
                        print("loading a checkpoint!")
                        training_state, step_id = checkpointer.load(
                            local_devices, fixed_params
                        )
                    else:
                        print(f"init_tr: {jax.tree_map(lambda a: a.shape, data_batch)}")
                        print(f"(k) init_training_state: {k.shape}")
                        training_state = init_training_state(data_batch, k)
                    if not DEBUG:
                        param_count = float(  # [0] indexes the device dim
                            sum(
                                x[0].size
                                for x in jax.tree_leaves(training_state.params)
                            )
                        )
                        param_count_structure_encoder = float(
                            sum(
                                x[0].size
                                for x in jax.tree_leaves(
                                    {
                                        k: v
                                        for k, v in training_state.params.items()
                                        if k.startswith("forward_bio_clip")
                                    }
                                )
                            )
                        )
                        # # load in the ESM parameters
                        # for k in training_state.params:
                        #     training_state.params[k] = esm_parameters[k]
                        print(f"model has {param_count:.2E}M params.")
                        print(
                            f"structure enc: {param_count_structure_encoder}M params."
                        )

                if step_id % cfg.training.validation_freq == 0:
                    # Run validation.
                    loss = {}
                    for val_name in ["unif", "clust"]:
                        random_key, k = jax.random.split(random_key)
                        logger.info(f"Running validation.")
                        start_time_validation = time.perf_counter()
                        loss[val_name] = run_validation(training_state, k, val_name)
                        validation_time = time.perf_counter() - start_time_validation
                        logger.info(
                            f"Validation complete in {validation_time} seconds."
                        )

                        if jax.process_index() == 0:
                            # Log metrics and save checkpoint from primary process only.
                            neptune_run.log(
                                name=f"validation_{val_name}/loss",
                                value=loss[val_name],
                                step=step_id,
                            )
                            neptune_run.log(
                                name=f"validation_{val_name}/time",
                                value=validation_time,
                                step=step_id,
                            )

                    if jax.process_index() == 0:
                        checkpointer(
                            cfg,
                            training_state,
                            step=step_id,
                            current_validation_loss=loss,
                        )

                # Take training step.
                logger.info("Running update")
                start_time_update = time.perf_counter()
                training_state, metrics, summaries = pmap_update(
                    training_state, data_batch
                )
                if not DEBUG:
                    training_state = jax.tree_map(
                        lambda x: x.block_until_ready(), training_state
                    )
                runtime_update = time.perf_counter() - start_time_update
                logger.info(f"Update took {runtime_update} secs.")

                metrics = jax.tree_map(
                    lambda x: jax.device_put(
                        jnp.mean(x), jax.devices(backend="cpu")[0]
                    ),
                    metrics.metrics,
                )

                if jax.process_index() == 0:
                    if not DEBUG:
                        summaries = {
                            f"{k1}_{k2}": float(v[0])
                            for k1, _v in summaries.items()
                            for k2, v in _v.items()
                        }
                        print(summaries)
                        # Log metrics from primary process only.
                        metrics |= {
                            "train_step_runtime": runtime_update,
                            "get_batch_runtime": runtime_get_batch,
                            # "esm_forward_runtime": runtime_forward_esm,
                            **summaries,
                        }
                        for k, v in metrics.items():
                            if jax.process_index() == 0:
                                neptune_run.log(
                                    name=f"training/{k}", value=v, step=step_id
                                )

                        start_time_get_batch = time.perf_counter()
                    step_id += 1

                if not DEBUG:
                    if jax.process_index() == 0:
                        checkpointer(cfg, training_state, step=step_id)


def main():
    """Important note on checkpointing:
    Checkpoint names will be derived from neptune run name and some parts of the config.
    When resuming a run, the user only need to provide the initial neptune run number.
    """
    show_git()
    set_start_method("spawn")

    cfg = load_config(
        name="clip_pretraining",
        job_name="test_pretraining",
        config_path="../../bio_clip/config",
    )

    if len(cfg.training.checkpoints.resume_from):
        resume_from = cfg.training.checkpoints.resume_from
        # The problem with loading previous checkpoints, means that we will still be
        # using a new version of the source code but the old version of the config.
        # For new features in the source code we can assume that feature was previously
        # off before.

        # find the checkpoint with the prefix given in resume_from
        _resume = cfg.training.checkpoints.resume_from
        try:
            if _resume.startswith("s3://"):
                endpoint = cfg.training.checkpoints.aws_endpoint
                cmd = f"aws s3 ls --endpoint={endpoint} {_resume}"
                p = subprocess.run(cmd.split(" "), capture_output=True, check=True)
                possible_paths = (
                    p.stdout.decode("utf-8")
                    .replace(" " * 27 + "PRE ", "")
                    .split("\n")[:-1]
                )
            else:
                possible_paths = [os.path.join(_resume, p) for p in os.listdir(_resume)]
            if len(possible_paths) == 1:
                [path] = possible_paths
                print(f"Unique path found: {path}")
            else:
                raise Exception(f"possible_paths invalid: {possible_paths}")
        except Exception as e:
            print(
                "\n"
                + "-" * 80
                + f"CHECKPOINT PATTERN NOT FOUND:\n"
                + f"resume_from: {cfg.training.checkpoints.resume_from}"
                + "-" * 80
                + "\n"
            )
            raise Exception(str(e))

        base_path = os.path.join(cfg.training.checkpoints.checkpoint_base, path)
        resume_path = os.path.join(base_path, "latest_checkpoint")
        print(
            "\n\n"
            + "-" * 80
            + "\nYou have clip_pretraining.training.checkpoints.resume set to true"
        )
        print(
            f"Therefore the config and checkpoint here will be used: {resume_path}\n"
            + "-" * 80
        )
        new_cfg = load_pretrained_checkpoint_config(
            resume_path, cfg.training.checkpoints.aws_endpoint
        )
        cfg.update(new_cfg)
        # this is the only thing we need to copy back
        cfg.training.checkpoints.resume_from = resume_from
        resume_path = cfg.training.checkpoints.resume_from
    else:
        os.makedirs(cfg.training.checkpoints.checkpoint_base, exist_ok=True)
        resume_path = ""

    print(cfg)

    # package training functions into static methods to explicitly control scopes / show
    # dependencies of each function
    run_trainer = RunTrainer()

    # modify the batch-size-per-device in the config according to the device count
    (
        cfg,
        devices,
        local_devices,
        global_device_count,
        local_device_count,
        batch_size_per_device,
    ) = run_trainer.prepare_devices(cfg)

    optimiser = run_trainer.prepare_optimiser(cfg)

    # prepare all jax / haiku updates / forward functions
    (
        trainer_init,
        tokenizer,
        esm_parameters,
        pmap_update,
        pmap_predict,
        parameters_partition_fn,
    ) = run_trainer.prepare_jitted_fns(
        cfg=cfg,
        devices=devices,
        local_devices=local_devices,
        optimizer=optimiser,
    )

    random_key = jax.random.PRNGKey(seed=cfg.random_seed)

    # Set up dataloading
    random_key, get_dataloader = run_trainer.prepare_data(
        random_key=random_key,
        cfg=cfg,
        global_device_count=global_device_count,
        local_device_count=local_device_count,
        batch_size_per_device=batch_size_per_device,
        tokenizer=tokenizer,
    )

    # full training routine
    run_trainer.run(
        random_key=random_key,
        parameters_partition_fn=parameters_partition_fn,
        cfg=cfg,
        resume_path=resume_path,
        global_device_count=global_device_count,
        local_devices=local_devices,
        local_device_count=local_device_count,
        batch_size_per_device=batch_size_per_device,
        trainer_init=trainer_init,
        get_dataloader=get_dataloader,
        esm_parameters=esm_parameters,
        pmap_update=pmap_update,
        pmap_predict=pmap_predict,
    )


if __name__ == "__main__":
    if TPU_NODES > 1:
        jax.config.update("jax_platform_name", "tpu")
        jax.distributed.initialize()
    main()
