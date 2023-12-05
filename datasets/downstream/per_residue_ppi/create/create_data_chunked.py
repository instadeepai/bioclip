import multiprocessing
import os
import pickle
import re
import subprocess
import time
from functools import partial
from glob import glob

import numpy as np
import torch as pt

from bio_clip.data.downstream.pesto_src.data_encoding import (
    encode_features,
    encode_structure,
    extract_all_contacts,
    extract_topology,
)
from bio_clip.data.downstream.pesto_src.dataset import save_data
from bio_clip.data.downstream.pesto_src.structure import (
    clean_structure,
    filter_non_atomic_subunits,
    remove_duplicate_tagged_subunits,
    split_by_chain,
    tag_hetatm_chains,
)
from bio_clip.data.downstream.pesto_src.structure_io import read_pdb

pt.multiprocessing.set_sharing_strategy("file_system")


config_dataset = {
    # parameters
    "r_thr": 5.0,  # Angstroms
    "max_num_atoms": 1024 * 8,
    "max_num_nn": 64,
    "molecule_ids": np.array(
        [
            "GLU",
            "LEU",
            "ALA",
            "ASP",
            "SER",
            "VAL",
            "GLY",
            "THR",
            "ARG",
            "PHE",
            "TYR",
            "ILE",
            "PRO",
            "ASN",
            "LYS",
            "GLN",
            "HIS",
            "TRP",
            "MET",
            "CYS",
            "A",
            "U",
            "G",
            "C",
            "DA",
            "DT",
            "DG",
            "DC",
            "MG",
            "ZN",
            "CL",
            "CA",
            "NA",
            "MN",
            "K",
            "IOD",
            "CD",
            "CU",
            "FE",
            "NI",
            "SR",
            "BR",
            "CO",
            "HG",
            "SO4",
            "NAG",
            "PO4",
            "EDO",
            "ACT",
            "MAN",
            "HEM",
            "FMT",
            "BMA",
            "ADP",
            "FAD",
            "NAD",
            "NO3",
            "GLC",
            "ATP",
            "NAP",
            "BGC",
            "GDP",
            "FUC",
            "FES",
            "FMN",
            "GAL",
            "GTP",
            "PLP",
            "MLI",
            "ANP",
            "H4B",
            "AMP",
            "NDP",
            "SAH",
            "OXY",
            "PLM",
            "CLR",
            "CDL",
            "RET",
        ]
    ),
    # input filepaths
    "pdb_filepaths": glob("data/all_biounits/*/*.pdb[0-9]*.gz"),
    # "pdb_filepaths": glob(f"/tmp/{sys.argv[-1]}/all_biounits/*/*.pdb[0-9]*.gz"),
    # output filepath
    "dataset_filepath": "data/datasets/contacts_rr5A_64nn_8192_wat.h5",
    # "dataset_filepath": f"/tmp/{sys.argv[-1]}/contacts_rr5A_64nn_8192.h5",
}


def contacts_types(s0, M0, s1, M1, ids, molecule_ids, device=pt.device("cpu")):
    # molecule types for s0 and s1
    c0 = pt.from_numpy(s0["resname"].reshape(-1, 1) == molecule_ids.reshape(1, -1)).to(
        device
    )
    c1 = pt.from_numpy(s1["resname"].reshape(-1, 1) == molecule_ids.reshape(1, -1)).to(
        device
    )

    # categorize contacts
    H = c1[ids[:, 1]].unsqueeze(1) & c0[ids[:, 0]].unsqueeze(2)

    # residue indices of contacts
    rids0 = pt.where(M0[ids[:, 0]])[1]
    rids1 = pt.where(M1[ids[:, 1]])[1]

    # create detailed contact map: automatically remove duplicated atom-atom to
    # residue-residue contacts
    Y = pt.zeros(
        (M0.shape[1], M1.shape[1], H.shape[1], H.shape[2]), device=device, dtype=pt.bool
    )
    Y[rids0, rids1] = H

    # define assembly type fingerprint matrix
    T = pt.any(pt.any(Y, dim=1), dim=0)

    return Y, T


def pack_structure_data(X, qe, qr, qn, M, ids_topk):
    return {
        "X": X.cpu().numpy().astype(np.float32),
        "ids_topk": ids_topk.cpu().numpy().astype(np.uint16),
        "qe": pt.stack(pt.where(qe > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        "qr": pt.stack(pt.where(qr > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        "qn": pt.stack(pt.where(qn > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        "M": pt.stack(pt.where(M), dim=1).cpu().numpy().astype(np.uint16),
    }, {
        "qe_shape": qe.shape,
        "qr_shape": qr.shape,
        "qn_shape": qn.shape,
        "M_shape": M.shape,
    }


def pack_contacts_data(Y, T):
    return {"Y": pt.stack(pt.where(Y), dim=1).cpu().numpy().astype(np.uint16)}, {
        "Y_shape": Y.shape,
        "ctype": T.cpu().numpy(),
    }


def pack_dataset_items(
    subunits, contacts, molecule_ids, max_num_nn, device=pt.device("cpu")
):
    # prepare storage
    structures_data = {}
    contacts_data = {}

    # extract features and contacts for all subunits with contacts
    for cid0 in contacts:
        # get subunit
        s0 = subunits[cid0]

        # extract features, encode structure and compute topology
        qe0, qr0, qn0 = encode_features(s0)
        X0, M0 = encode_structure(s0, device=device)
        ids0_topk = extract_topology(X0, max_num_nn)[0]

        # store structure data
        structures_data[cid0] = pack_structure_data(X0, qe0, qr0, qn0, M0, ids0_topk)

        # prepare storage
        if cid0 not in contacts_data:
            contacts_data[cid0] = {}

        # for all contacting subunits
        for cid1 in contacts[cid0]:
            # prepare storage for swapped interface
            if cid1 not in contacts_data:
                contacts_data[cid1] = {}

            # if contacts not already computed
            if cid1 not in contacts_data[cid0]:
                # get contacting subunit
                s1 = subunits[cid1]

                # encode structure
                X1, M1 = encode_structure(s1, device=device)

                # nonzero not supported for array with more than I_MAX elements
                if (M0.shape[1] * M1.shape[1] * (molecule_ids.shape[0] ** 2)) > 2e9:
                    # compute interface targets
                    ctc_ids = contacts[cid0][cid1]["ids"].cpu()
                    Y, T = contacts_types(
                        s0,
                        M0.cpu(),
                        s1,
                        M1.cpu(),
                        ctc_ids,
                        molecule_ids,
                        device=pt.device("cpu"),
                    )
                else:
                    # compute interface targets
                    ctc_ids = contacts[cid0][cid1]["ids"].to(device)
                    Y, T = contacts_types(
                        s0,
                        M0.to(device),
                        s1,
                        M1.to(device),
                        ctc_ids,
                        molecule_ids,
                        device=device,
                    )

                # if has contacts of compatible type
                if pt.any(Y):
                    # store contacts data
                    contacts_data[cid0][cid1] = pack_contacts_data(Y, T)
                    contacts_data[cid1][cid0] = pack_contacts_data(
                        Y.permute(1, 0, 3, 2), T.transpose(0, 1)
                    )

                # clear cuda cache
                pt.cuda.empty_cache()

    return structures_data, contacts_data


def store_dataset_items(hf, pdbid, bid, structures_data, contacts_data):
    # metadata storage
    metadata_l = []

    # for all subunits with contacts
    for cid0 in contacts_data:
        # define store key
        key = f"{pdbid.upper()[1:3]}/{pdbid.upper()}/{bid}/{cid0}"

        # save structure data
        hgrp = hf.create_group(f"data/structures/{key}")
        save_data(hgrp, attrs=structures_data[cid0][1], **structures_data[cid0][0])

        # for all contacting subunits
        for cid1 in contacts_data[cid0]:
            # define contacts store key
            ckey = f"{key}/{cid1}"

            # save contacts data
            hgrp = hf.create_group(f"data/contacts/{ckey}")
            save_data(
                hgrp, attrs=contacts_data[cid0][cid1][1], **contacts_data[cid0][cid1][0]
            )

            # store metadata
            metadata_l.append(
                {
                    "key": key,
                    "size": (np.max(structures_data[cid0][0]["M"], axis=0) + 1).astype(
                        int
                    ),
                    "ckey": ckey,
                    "ctype": contacts_data[cid0][cid1][1]["ctype"],
                }
            )

    return metadata_l


def body(pdb_filepath, output):

    # I have moved the read_pdb function from the dataloader in here
    structure = read_pdb(pdb_filepath)

    # check that structure was loaded
    if structure is None:
        return

    # parse filepath
    m = re.match(r".*/([a-z0-9]*)\.pdb([0-9]*)\.gz", pdb_filepath)
    pdbid = m[1]
    bid = m[2]

    # check size
    if structure["xyz"].shape[0] >= config_dataset["max_num_atoms"]:
        return

    # process structure
    structure = clean_structure(structure)

    # update molecules chains
    structure = tag_hetatm_chains(structure)

    # split structure
    subunits = split_by_chain(structure)

    # remove non atomic structures
    subunits = filter_non_atomic_subunits(subunits)

    # check not monomer
    if len(subunits) < 2:
        return

    # remove duplicated molecules and ions
    subunits = remove_duplicate_tagged_subunits(subunits)

    # extract all contacts from assembly
    contacts = extract_all_contacts(subunits, config_dataset["r_thr"])

    # check there are contacts
    if len(contacts) == 0:
        return

    # pack dataset items
    structures_data, contacts_data = pack_dataset_items(
        subunits,
        contacts,
        config_dataset["molecule_ids"],
        config_dataset["max_num_nn"],  # device=device
    )
    output[pdb_filepath] = {
        "pdbid": pdbid,
        "bid": bid,
        "structures_data": structures_data,
        "contacts_data": contacts_data,
    }


def launch_mp(filepaths):
    # Store the outputs in a dict suited to multiprocessing.
    manager = multiprocessing.Manager()
    output = manager.dict()
    _body = partial(body, output=output)

    # Create and start all of the processes.
    jobs = []
    for pdb_filepath in filepaths:
        p = multiprocessing.Process(target=_body, args=(pdb_filepath,))
        jobs.append(p)
        p.start()

    # Sync up all processes.
    for job in jobs:
        job.join()

    # Extract the dictionary.
    d = output.copy()
    return d


def run_chunk(
    chunk_write_path, chunk_paths, buffer_size
):  # , num_workers, prefetch_factor):
    # # with the dataloader it took about 37min for 1/92 chunks, chunk is ~1.2GB
    # # without the dataloader it took about 40min for 1/92 chunks, chunk is ~1.2GB
    # without dataloader and no loop over chunk. 30 mins
    ks = ["AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID"]
    for k in ks:
        assert k in os.environ and len(os.environ[k]), f"{k} not exported"
    t = time.time()
    chunk = launch_mp(chunk_paths)
    print(f"TOOK: {time.time() - t}s")
    # save the chunk
    with open(chunk_write_path, "wb") as f:
        pickle.dump(chunk, f)
    # upload the chunk
    dst = f"s3://deepchain-research/bio_clip/pesto-data/{chunk_write_path}"
    os.system(
        f"aws s3 cp {chunk_write_path} {dst} --endpoint https://s3.kao.instadeep.io"
    )
    # check it worked
    try:
        src = f"s3://deepchain-research/bio_clip/pesto-data/{chunk_write_path}"
        cmd = f"aws s3 ls {src} --endpoint https://s3.kao.instadeep.io"
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode(
            "ascii"
        )
        os.system(f"rm {chunk_write_path}")
        status = "SUCCESS" if chunk_write_path.split("/")[-1] in out else "FAILED"
        print(f"{status} UPLOAD {chunk_write_path}")
    except Exception as e:
        print(f"FAILED:\n\t{e}")


if __name__ == "__main__":
    # set up dataset
    num_cpus = 64
    paths = config_dataset["pdb_filepaths"]
    chunk_size = num_cpus * 50
    chunks = np.array_split(paths, len(paths) // chunk_size)
    chunk_base_path = "chunks"
    os.makedirs(chunk_base_path, exist_ok=True)
    chunk_write_path = os.path.join(chunk_base_path, "chunk_%d.pickle")

    for i, chunk in enumerate(chunks, start=1):
        print(f"RUNNING CHUNK {i} OF {len(chunks)}; size: {len(chunk)}")
        outpath = chunk_write_path % i
        run_chunk(outpath, chunk, buffer_size=num_cpus * 10)
