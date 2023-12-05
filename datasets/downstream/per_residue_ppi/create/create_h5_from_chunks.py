import os
import pickle
import subprocess

import h5py
import numpy as np
from create_data_chunked import config_dataset, store_dataset_items
from tqdm import tqdm

from bio_clip.data.downstream.pesto_src.data_encoding import config_encoding

BUCKET_PATH = "s3://deepchain-research/bio_clip/pesto-data"
ENDPOINT = "https://s3.kao.instadeep.io"


def download_chunks(local_path="chunks/chunk_%d.pickle", load_into_ram=True):
    ks = ["AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID"]
    for k in ks:
        assert k in os.environ and len(os.environ[k]), f"{k} not exported"
    cmd = f"aws s3 ls {BUCKET_PATH}/chunks/ --endpoint {ENDPOINT}"
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode(
        "ascii"
    )
    nums = [
        int(line.split("chunk_")[-1].split(".")[0])
        for line in out.split("\n")
        if len(line)
    ]
    next_num = max(nums) + 1
    all_chunks = {}
    for i in range(1, next_num):
        _local_path = local_path % i
        print(f"loading previous chunk_{i} to get keys...")
        if not os.path.isfile(_local_path):
            path = f"{BUCKET_PATH}/chunks/chunk_{i}.pickle"
            os.system(f"aws s3 cp {path} {_local_path} --endpoint {ENDPOINT}")
        if load_into_ram:
            with open(_local_path, "rb") as f:
                chunk = pickle.load(f)
                all_chunks = {**all_chunks, **chunk}
    return all_chunks


def write_hdf5_file(go_from=1, local_path="chunks/chunk_%d.pickle"):
    metadata_l = []

    mode = "a" if go_from > 1 else "w"
    # process structure, compute features and write dataset
    with h5py.File(config_dataset["dataset_filepath"], mode, libver="latest") as hf:
        # store dataset encoding
        for key in config_encoding:
            hf[f"metadata/{key}"] = config_encoding[key].astype(np.string_)

        # save contact type encoding
        hf["metadata/mids"] = config_dataset["molecule_ids"].astype(np.string_)

        # prepare and store all structures
        cmd = f"aws s3 ls {BUCKET_PATH}/chunks/ --endpoint {ENDPOINT}"
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            shell=True,
        ).decode("ascii")
        nums = [
            int(line.split("chunk_")[-1].split(".")[0])
            for line in out.split("\n")
            if len(line)
        ]
        for i in range(go_from, max(nums) + 1):
            _local_path = local_path % i
            print(f"loading pickle: {_local_path}")
            with open(_local_path, "rb") as f:
                chunk = pickle.load(f)
            # os.system(f"rm {local_path}")
            print("placing into HDF5")
            for values in tqdm(chunk.values()):
                # store data
                metadata = store_dataset_items(hf, **values)
                metadata_l.extend(metadata)
            del chunk

        # store metadata
        hf["metadata/keys"] = np.array([m["key"] for m in metadata_l]).astype(
            np.string_
        )
        hf["metadata/sizes"] = np.array([m["size"] for m in metadata_l])
        hf["metadata/ckeys"] = np.array([m["ckey"] for m in metadata_l]).astype(
            np.string_
        )
        hf["metadata/ctypes"] = np.stack(
            np.where(np.array([m["ctype"] for m in metadata_l])), axis=1
        ).astype(np.uint32)


if __name__ == "__main__":
    all_chunks = download_chunks(
        local_path="chunks/chunk_%d.pickle", load_into_ram=False
    )
    write_hdf5_file(go_from=1)
    name = config_dataset["dataset_filepath"]
    print(f"Uploading final data...")
    path = f"{BUCKET_PATH}/{os.path.split(name)[1]}"
    os.system(f"aws s3 cp {name} {path} --endpoint {ENDPOINT}")
