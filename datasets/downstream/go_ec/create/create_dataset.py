import argparse
import gzip
import os
from multiprocessing import Event, Process, Queue
from queue import Empty, Full
from time import sleep

import h5py
import numpy as np

#  Make sure to add DeepFRI to your python path
from deepfrier.utils import load_EC_annot, load_GO_annot
from tqdm import tqdm

from bio_clip.data.alphafold_scripts.protein import from_mmcif_string


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        default="full_dataset.h5",
        help="Path to the h5 file to write.",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="deepfri_results.npy",
        help="Path to the results file to read.",
    )
    parser.add_argument(
        "--pdb-directory", type=str, default="", help="Path to the pdb files to read."
    )
    parser.add_argument("--overwrite-h5-start-over", type=bool, help=".")
    parser.add_argument("--num-process", type=int)
    args = parser.parse_args()
    return args


def process_samples(
    samples_queue,
    failed_queue,
    remaining_queue,
    pdb_directory,
    processes_should_exit,
    max_num_consecutive_errors,
):
    num_consecutive_errors = 0
    while True:
        if processes_should_exit.is_set():
            return None

        try:
            pdb_code_chain = remaining_queue.get(timeout=0.5)
        except Empty:
            if processes_should_exit.is_set():
                return None
            continue

        try:
            code, chain = str(pdb_code_chain).split("-")
            with open(os.path.join(pdb_directory, f"{code}.cif.gz"), "rb") as f:
                cif_str = gzip.decompress(f.read()).decode("ascii")
            protein_object = from_mmcif_string(cif_str, chain)
        except Exception as e:
            num_consecutive_errors += 1
            if num_consecutive_errors >= max_num_consecutive_errors:
                raise
            failed_queue.put((pdb_code_chain, str(e)), timeout=0.5)
            print(f"{pdb_code_chain} failed: {e}")
            continue

        num_consecutive_errors = 0

        if protein_object is not None:
            while True:
                try:
                    samples_queue.put((pdb_code_chain, protein_object), timeout=0.5)
                    break
                except Full:
                    if processes_should_exit.is_set():
                        return None


def write_to_h5(samples_queue, remaining_queue, h5_path):
    count = 0
    while True:
        if samples_queue.empty():
            sleep(0.01)
            continue

        if remaining_queue.empty():
            return None

        count += 1
        if count % 100:
            print(f"num_remaining: {remaining_queue.qsize()}")

        code, protein = samples_queue.get(timeout=0.5)

        with h5py.File(h5_path, "a", libver="latest") as hf:
            hf[f"input_features/{code}/aatype"] = protein.aatype.astype(np.int32)
            hf[f"input_features/{code}/atom_positions"] = protein.atom_positions.astype(
                np.float32
            )  # np.string_)
            hf[f"input_features/{code}/atom_mask"] = protein.atom_mask.astype(np.int32)
            hf[f"input_features/{code}/residue_index"] = protein.residue_index.astype(
                np.int32
            )
            hf[f"input_features/{code}/b_factors"] = protein.b_factors.astype(
                np.float32
            )


def write_to_error_log(failed_queue, remaining_queue, error_path):
    while True:
        if failed_queue.empty():
            sleep(0.01)
            continue

        if remaining_queue.empty():
            return None

        info = failed_queue.get(timeout=0.5)

        # append to log file
        with open(error_path, "a") as f:
            f.write(f"\n#\n{info}")


def test(h5_path, wewant, test_examples):
    with h5py.File(h5_path, "r", libver="latest") as hf:
        wegot = set(hf["input_features"].keys())
    anymissing = len(wewant - wegot)
    anymissingtest = len(test_examples - wegot)
    print(f"anymissing: {anymissing}; anymissingtest: {anymissingtest}")


def continue_running(
    h5_path,
    error_path,
    pdb_directory,
    num_process,
    wewant,
    remaining,
    testexamples,
    max_num_consecutive_errors=20,
):
    with open(error_path, "w") as f:
        f.write("")

    remaining_queue = Queue()
    samples_queue = Queue()
    failed_queue = Queue()
    for r in tqdm(remaining):
        remaining_queue.put(r, timeout=0.5)
    processes_should_exit = Event()

    processes = [
        Process(
            target=process_samples,
            args=(
                samples_queue,
                failed_queue,
                remaining_queue,
                pdb_directory,
                processes_should_exit,
                max_num_consecutive_errors,
            ),
            daemon=True,
        )
        for _ in range(num_process)
    ] + [
        Process(target=write_to_h5, args=(samples_queue, remaining_queue, h5_path)),
        Process(
            target=write_to_error_log, args=(failed_queue, remaining_queue, error_path)
        ),
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print(f"Done")
    test(h5_path, wewant, testexamples)


def load_annotations():
    # get all of the codes we want inputs for...
    ec_annot_fn = "/app/DeepFRI/preprocessing/data/nrPDB-EC_2020.04_annot.tsv"
    go_annot_fn = "/app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv"
    ec = load_EC_annot(ec_annot_fn)
    go = load_GO_annot(go_annot_fn)
    return ec, go


def get_codes(h5_path, results):
    results = np.load(results, allow_pickle=True).item()
    testexamples = []
    for v in results.values():
        testexamples += list(v["proteins"])

    # open h5, read all of the keys we have input features for...
    with h5py.File(h5_path, "r", libver="latest") as hf:
        wehave = set(hf["identifiers"])
    ec, go = load_annotations()

    wewant = set(ec[0].keys()).union(set(go[0].keys()))

    remaining = set(wewant - wehave)
    return wewant, remaining, set(testexamples)


if __name__ == "__main__":
    args = parse_args()
    if args.overwrite_h5_start_over:
        ec, go = load_annotations()
        print("Overwriting h5 file.")
        # write targets
        with h5py.File(args.h5_path, "w", libver="latest") as hf:
            for name, (prot2annot, goterms, gonames, counts) in [
                ("ec", ec),
                ("go", go),
            ]:
                for onto, gt in goterms.items():
                    hf[f"{name}/goterms/{onto}"] = np.array(gt).astype(np.string_)
                for onto, gn in gonames.items():
                    hf[f"{name}/gonames/{onto}"] = np.array(gn).astype(np.string_)
                for onto, ct in counts.items():
                    hf[f"{name}/counts/{onto}"] = np.array(ct).astype(np.int32)
                identifiers = list(prot2annot.keys())
                hf[f"{name}/identifiers"] = np.array(identifiers).astype(np.string_)
                for onto in goterms.keys():
                    hf[f"{name}/{onto}"] = np.stack(
                        [prot2annot[_id][onto] for _id in identifiers]
                    ).astype(np.int32)
                    # easier format for data-loading later
                    for _id in tqdm(identifiers):
                        hf[f"targets/{name}/{onto}/{_id}"] = prot2annot[_id][
                            onto
                        ].astype(np.int32)
        print(f"Written targets to {args.h5_path}")
    else:
        print("Appending to h5 file.")

    wewant, remaining, testexamples = get_codes(args.h5_path, args.results)

    continue_running(
        args.h5_path,
        args.pdb_directory,
        args.num_process,
        wewant,
        remaining,
        testexamples,
    )
