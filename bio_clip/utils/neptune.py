import json
import logging
import os
from datetime import datetime


class NeptunePlaceholder:
    def __init__(
        self,
        project,
        name,
        tags,
        config,
        resume=False,
        neptune_directory="/app/bio-clip/neptune_runs",
    ):
        self.project = project
        self.name = name
        self.tags = sorted(tags)
        self.directory = os.path.join(neptune_directory, self.project)
        self.tags_file_path = os.path.join(self.directory, "tags_to_run_id.json")
        self.config = dict(config)

        # Load or initialize the tags to run_id mapping
        self.tags_to_run_id = self._load_or_initialize_tags_mapping()

        if resume:
            self.run_id = self._resume_run_id()
        else:
            self.run_id = self._initialize_run_id()
            if ":".join(self.tags) in self.tags_to_run_id:
                raise NotImplementedError(
                    "Tags must be unique to previous runs if not resuming."
                )

        self.file_path = os.path.join(self.directory, f"run_{self.run_id}.json")
        self.fields = self._load_run_data()

        # Create the directory if it does not exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self._update_tags_file()  # Update the tags file with the new run

    def _load_or_initialize_tags_mapping(self):
        # Load tags_to_run_id.json if it exists, else return an empty dict
        if os.path.exists(self.tags_file_path):
            with open(self.tags_file_path, "r") as file:
                return json.load(file)
        else:
            return {}

    def _resume_run_id(self):
        # Check if these tags were used before and return the associated run_id
        k = ":".join(self.tags)
        if k in self.tags_to_run_id:
            return self.tags_to_run_id[k]["run_id"]
        return self._initialize_run_id()

    def _initialize_run_id(self):
        # If the directory does not exist or is empty, start with run_id 0
        if not os.path.exists(self.directory) or not os.listdir(self.directory):
            return 0

        # Otherwise, find the highest run_id in the directory
        max_run_id = 0
        for file_name in os.listdir(self.directory):
            if file_name.startswith("run_") and file_name.endswith(".json"):
                try:
                    run_id = int(file_name[4:-5])  # Extract run_id from file name
                    max_run_id = max(max_run_id, run_id)
                except ValueError:
                    continue

        return max_run_id + 1

    def _load_run_data(self):
        # Load existing data for the run if the file exists
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                return json.load(file)
        return {}

    def _update_tags_file(self):
        # Update the tags file with the current run_id
        self.tags_to_run_id[":".join(self.tags)] = {
            "run_id": self.run_id,
            "config": self.config,
        }

        with open(self.tags_file_path, "w") as file:
            json.dump(self.tags_to_run_id, file)

    def log(self, field_name, value, step=None):
        if field_name not in self.fields:
            self.fields[field_name] = []

        # Determine the step if not provided
        if step is None:
            step = len(self.fields[field_name])

        # Create a log entry
        log_entry = {
            "value": value,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
        }
        self.fields[field_name].append(log_entry)

        # Write the entire fields dictionary to the file
        with open(self.file_path, "w") as file:
            json.dump(self.fields, file)

    def __getitem__(self, field_name):
        return NeptuneFieldLogger(self, field_name)


class NeptuneFieldLogger:
    def __init__(self, neptune_run, field_name):
        self.neptune_run = neptune_run
        self.field_name = field_name

    def log(self, value, step=None):
        self.neptune_run.log(self.field_name, value, step)


def prepare_downstream_neptune(config, name_expe, other_tags):
    """run_id: e.g. PES-205"""
    required = [
        "use_esm_embedding",
        "tune_esm",
        "train_esm_from",
        "use_gnn_embedding",
        "gnn_layer_to_use",
        "first_trainable_gnn_layer",
        "multilayer_classifier",
        "use_projected_structure_embedding",
        "optimise_everything_overide",
        "load_bioclip_params",
    ]
    tags_expe = [f"{k}_{getattr(config.training, k)}" for k in required]
    use_proj_structure = config.training.use_projected_structure_embedding
    tags_expe += [
        config.training.checkpoints.checkpoint_dir,
        f"learning_rate_{config.training.optimiser.learning_rate}",
        f"weight_decay_{config.training.optimiser.weight_decay}",
        f"batch_size_{config.training.batch.batch_size}",
        f"gnn_nlayers_{config.model.gnn.gnn_number_layers}",
        f"multimodal_dim_{config.model.dimension_multimodal_space}",
        f"use_projected_structure_embedding_{use_proj_structure}",
        f"mlp_final_hidden_layers_{config.model.final_hidden_layers}",
        f"num_epochs_{config.training.num_epochs}",
        f"{config.experiment_name}",
    ] + other_tags
    neptune_run = NeptunePlaceholder(
        project=config.training.neptune.project_name,
        name=name_expe,
        tags=tags_expe,
        config=config.to_dict(),
        resume=False,
    )
    logging.warning(
        f"Using the following neptune project: {config.training.neptune.project_name}"
    )
    return neptune_run
