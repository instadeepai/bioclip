import glob
import pickle

import numpy as np

pkls = glob.glob("trained_models/*_results.pckl")
collected = {}
for fname in pkls:
    name = fname.split("/")[-1].replace("_results.pckl", "")
    onto = "-".join(name.split("_")[-2:])
    with open(fname, "rb") as f:
        results = pickle.load(f)
    results["model_full_name"] = name
    collected[onto] = results

np.save("/app/bio-clip/datasets/downstream/go_ec/deepfri_baseline.npz", **collected)
