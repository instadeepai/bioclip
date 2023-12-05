import subprocess

PRETRAINING_BASE = "s3://deepchain-research/bio_clip/checkpoints/pretraining"


def create_config_override(values):
    names = [
        "use_gnn_embedding",
        "first_trainable_gnn_layer",
        "load_bioclip_params",
        "use_esm_embedding",
        "train_esm_from",  # Setting to 100 means no ESM parameters will be tuned.
        "gnn_layer_to_use",  # should always be -1 apart from GNN+ESM runs
    ]
    return {f"training.{name}": value for name, value in zip(names, values)}


ESM_ONLY = ["Tuned ESM"]  # ["Frozen ESM", "Tuned ESM"]
GNN_ONLY = ["Random GNN", "BioCLIP GNN"]
ESM_AND_GNN = [
    "Tuned ESM, tuned BioCLIP GNN"
]  # ["Frozen ESM, tuned BioCLIP GNN", "Tuned ESM, tuned BioCLIP GNN"]
PROBE_EXPS = ["Probe BioCLIP GNN", "Probe Random GNN", "Probe ESM, BioCLIP GNN"]
ALL_EXPS = ESM_ONLY + GNN_ONLY + ESM_AND_GNN + PROBE_EXPS
experiment_types = {
    "Random GNN": create_config_override([True, 0, False, False, 100, -2]),
    # "Frozen ESM": create_config_override([False, 100, False,  True, 100, -2]),
    "Tuned ESM": create_config_override([False, 100, False, True, -3, -2]),
    "BioCLIP GNN": create_config_override([True, 0, True, False, 100, -2]),
    # "Frozen ESM, tuned BioCLIP GNN":
    # create_config_override([ True,   0,  True,  True, 100, -2]),
    "Tuned ESM, tuned BioCLIP GNN": create_config_override(
        [True, 0, True, True, -3, -2]
    ),
    "Probe BioCLIP GNN": create_config_override([True, 100, True, False, 100, -2]),
    "Probe Random GNN": create_config_override([True, 100, False, False, 100, -2]),
    "Probe ESM, BioCLIP GNN": create_config_override([True, 100, True, True, 100, -2]),
}


def create_base_commands(task):
    """Create a list with all of the subtasks within that task. The elements of the list
    are:
    'script.py hydra.path.to.settings=value'
    """
    script, settings = {
        "PPI": ("ppi.py", [("training.data.benchmark", ["cerevisiae", "human"])]),
        "GO": ("go_terms.py", [("training.data.ontology", ["ec", "mf", "bp", "cc"])]),
        "ResPPI": ("residue_ppi.py", []),
    }[task]
    base_commands = []
    if len(settings):
        for setting, values in settings:
            for value in values:
                base_commands.append((script, f"{setting}={value}"))
    else:
        base_commands.append((script,))
    return base_commands


def main(jobs, task_priority=None, checkpoint_override=None):
    print(jobs)
    print()
    if task_priority is None:
        task_priority = list(jobs.keys())
    for task in task_priority:
        print(task)
        if task not in jobs:
            continue
        base_commands = create_base_commands(task)
        experiments = jobs[task]
        print(f"\t{experiments}")
        print()
        for exp in experiments:
            overrides = [f"{k}={v}" for k, v in experiment_types[exp].items()]
            overrides += [
                "training.tune_esm=True"
            ]  # This is always set to True, because there was previously a version of
            # the code which used pre-computed ESM embeddings
            if checkpoint_override is not None:
                overrides += [
                    f"training.checkpoints.checkpoint_dir={checkpoint_override}"
                ]
            overrides += [f"experiment_name={exp.replace(' ', '_').replace(',', '')}"]
            for script, *task_setting in base_commands:
                args = (
                    ["python", f"/app/bio-clip/scripts/finetune/{script}"]
                    + list(task_setting)
                    + overrides
                )
                print(" ".join(args))
                proc = subprocess.Popen(args, stdout=subprocess.PIPE)
                for line in proc.stdout:
                    data = line.rstrip().decode("utf-8")
                    print(data)
                    if "[Fold Done]" in data:  # just complete after the first fold...
                        proc.kill()


def standard_exps():
    return {
        "PPI": GNN_ONLY + ESM_AND_GNN,
        "GO": GNN_ONLY + ESM_AND_GNN,
        "ResPPI": GNN_ONLY + ESM_AND_GNN,
    }


def all_exps():
    return {"PPI": ALL_EXPS, "GO": ALL_EXPS, "ResPPI": ALL_EXPS}


if __name__ == "__main__":
    task_priority = ["GO", "PPI", "ResPPI"]
    jobs = standard_exps()
    checkpoint_dir = None
    main(jobs, task_priority, checkpoint_override=checkpoint_dir)
