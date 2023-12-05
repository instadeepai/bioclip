import json
import os
import subprocess

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

from bio_clip.train.pretrain.trainer import TrainingState
from bio_clip.utils.utils import convert_to_ml_dict, tmpdir_manager


def load_pretrained_checkpoint_config(checkpoint_dir, aws_endpoint):
    if checkpoint_dir.startswith("s3://"):
        local_path = "checkpoint/config.json"
        loc_dir = os.path.split(local_path)[0]
        os.makedirs(loc_dir, exist_ok=True)
        cmd = [
            "aws",
            "s3",
            "cp",
            os.path.join(checkpoint_dir, "config.json"),
            local_path,
            f"--endpoint={aws_endpoint}",
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, capture_output=True, check=True)
    else:
        local_path = os.path.join(checkpoint_dir, "config.json")
    with open(local_path) as f:
        clip_cfg = json.loads(f.read())
        clip_cfg = config_dict.ConfigDict(clip_cfg)
    if not hasattr(clip_cfg, "features"):
        # legacy checkpoints had features config in a different place
        clip_cfg.features = clip_cfg.model.features
        print(f"Using legacy checkpoint. clip_cfg.model.features -> clip_cfg.features")
    return clip_cfg


def device(config):
    def find_device(device_type):
        try:
            return jax.devices(device_type)
        except Exception as e:
            print(f"Failed to find {device_type}: {e}")

    for device_type in ["tpu", "gpu", "cpu"]:
        devices = find_device(device_type)
        if devices is not None:
            break

    if device_type == "cpu" or len(devices) == 1:
        print("\n! Overriding batch size as no devices found.\n")
        config.training.batch.batch_size = config.training.batch.num_per_device_update
    return config, devices


def prepare_ft_pretraining_cfgs(cfg, download_checkpoint_but_dont_load_weights=True):
    if cfg.training.checkpoints.checkpoint_dir.startswith("null|"):
        # load fine-tuning config
        clip_cfg = cfg
        # fine-tuning config doesn't have this feature...
        cfg.training.use_projected_sequence_embedding = False
    elif cfg.training.load_bioclip_params or download_checkpoint_but_dont_load_weights:
        print(
            "In the case of random GNN, we download a checkpoint anyway, to get the "
            "config from."
        )
        # load the clip config
        clip_cfg = load_pretrained_checkpoint_config(
            cfg.training.checkpoints.checkpoint_dir,
            cfg.training.checkpoints.aws_endpoint,
        )
        print("LOADED CONFIG FROM CHECKPOINT")
        clip_cfg.training.use_projected_sequence_embedding = False
        cfg.training.use_projected_sequence_embedding = False
        # # update any new values... TODO
        clip_cfg.model.remat_policy = cfg.model.remat_policy
    else:  # assume that the model is as specified in the fine-tuning hydra
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize(config_path="../../config", job_name="test"):
            clip_cfg = convert_to_ml_dict(
                hydra.compose(config_name="clip_pretraining", overrides=[])
            )
        clip_cfg.update(cfg)
        print("NOT LOADING CONFIG FROM CHECKPOINT")
        print(
            "\n\n! since we are not using pre-trained weights the base model config is"
            " going to use the clip_pretraining config\n\n"
        )
        # older checkpoints don't have this feature...
        cfg.training.use_projected_sequence_embedding = False
        clip_cfg.model.final_hidden_layers = cfg.model.final_hidden_layers

    cfg, devices = device(cfg)
    return clip_cfg, cfg, devices


class Checkpointer:
    def __init__(self, chk_cfg, run_id, description, partition_fn, resume_from=""):
        _base = os.path.join(
            chk_cfg.checkpoint_base,
            (f"{run_id}_{description}" if resume_from == "" else resume_from),
        )
        self.remote_checkpoints = _base.startswith("s3://")
        self.endpoint = chk_cfg.aws_endpoint
        self.regular_every = chk_cfg.regular_every
        self.latest_every = chk_cfg.latest_every
        self.regular_checkpoint_path = os.path.join(_base, chk_cfg.regular_rel_path)
        self.latest_checkpoint_path = os.path.join(_base, chk_cfg.latest_rel_path)
        self.best_checkpoint_path = os.path.join(_base, chk_cfg.best_rel_path)
        self.best_validation_loss = {"clust": jnp.array(1e10), "unif": jnp.array(1e10)}
        self.resume_from = self.latest_checkpoint_path
        self.partition_fn = partition_fn

    @staticmethod
    def wrap_aws_cp(from_path, destination_path, endpoint):
        print(f"calling wrap_aws_cp with {from_path} {destination_path} {endpoint}")
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                from_path,
                destination_path,
                f"--endpoint={endpoint}",
                "--recursive",
            ],
            capture_output=True,
            check=True,
        )

    def _upload(self, destination_path, training_state, cfg):
        def dev_0_to_cpu(data):
            return jax.device_put(
                jax.tree_map(lambda x: x[0], data), jax.devices("cpu")[0]
            )

        state = TrainingState(
            step=np.array(self.step),
            best_validation_cluster_loss=np.array(self.best_validation_loss["clust"]),
            best_validation_unif_loss=np.array(self.best_validation_loss["unif"]),
            params=dev_0_to_cpu(training_state.params),
            optimizer_state=dev_0_to_cpu(training_state.optimizer_state),
            random_key=dev_0_to_cpu(training_state.random_key),
        )

        def save(path):
            state.save(save_dir=path, partition_fn=self.partition_fn)
            with open(os.path.join(path, "config.json"), "w") as f:
                f.write(json.dumps(cfg.to_dict()))

        if self.remote_checkpoints:
            with tmpdir_manager(base_dir="/tmp") as tmp_dir:
                path = os.path.join(tmp_dir, "local_checkpoint")
                save(path)
                Checkpointer.wrap_aws_cp(
                    from_path=path,
                    destination_path=destination_path,
                    endpoint=self.endpoint,
                )
        else:
            save(destination_path)

    def load(self, devices, fixed_params=None):
        print(f"loading checkpoint: {self.latest_checkpoint_path}")

        def load(path):
            new_training_state = TrainingState.load(path, fixed_params=fixed_params)
            step_id = int(new_training_state.step + 1)
            multi_device_training_state = jax.device_put_replicated(
                new_training_state,
                devices,
            )
            return multi_device_training_state, step_id

        if self.remote_checkpoints:
            with tmpdir_manager(base_dir="/tmp") as tmp_dir:
                path = os.path.join(tmp_dir, "local_checkpoint")
                Checkpointer.wrap_aws_cp(
                    from_path=self.resume_from,
                    destination_path=path,
                    endpoint=self.endpoint,
                )
                multi_device_training_state, step_id = load(path)
        else:
            multi_device_training_state, step_id = load(self.resume_from)
        return multi_device_training_state, step_id

    def update_latest(self, cfg, current_state):
        self._upload(self.latest_checkpoint_path, current_state, cfg)
        print("Uploaded latest checkpoint.")

    def update_best(self, cfg, validation_loss, current_state):
        update = {"unif": False, "clust": False}
        for val_set in update:
            if validation_loss[val_set] < self.best_validation_loss[val_set]:
                self.best_validation_loss[val_set] = validation_loss[val_set]
                update[val_set] = True

        for val_set, ud in update.items():
            if ud:
                self._upload(
                    os.path.join(self.best_checkpoint_path, val_set), current_state, cfg
                )
                print(f"Best {val_set} validation performance so far.")

    def periodic_checkpoint(self, cfg, current_state, step):
        self._upload(self.regular_checkpoint_path.format(step), current_state, cfg)
        print("Periodic upload of checkpoint.")

    def __call__(self, cfg, current_state, step, current_validation_loss=None):
        self.step = step
        if current_validation_loss is None:
            if step % self.regular_every == 0:
                self.periodic_checkpoint(cfg, current_state, step)
            if step % self.latest_every == 0:
                self.update_latest(cfg, current_state)
        else:
            self.update_best(cfg, current_validation_loss, current_state)
            self.update_latest(cfg, current_state)
