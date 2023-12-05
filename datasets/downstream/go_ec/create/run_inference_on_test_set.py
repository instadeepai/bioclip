# python train_DeepFRI.py \
#    -gc GAT \
#    -bs 32 \
#    -pd 1024 \
#    -ont mf \
#    -lm /app/DeepFRI/trained_models/lstm_lm.hdf5 \
#    --model_name GCN-PDB_MF \
#    --train_tfrecord_fn /app/data/PDB-GO/PDB_GO_train \
#    --valid_tfrecord_fn /app/data/PDB-GO/PDB_GO_valid
import argparse
import csv
import json
import os
import pickle

import h5py
import numpy as np

#  Make sure to add DeepFRI to your python path
from deepfrier.DeepFRI import DeepFRI
from deepfrier.utils import load_EC_annot, load_GO_annot, seq2onehot
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-lm",
        "--lm_model_name",
        type=str,
        default="lstm_lm_tf.hdf5",
        help="Path to the pretrained LSTM-Language Model.",
    )
    parser.add_argument(
        "--params-path",
        type=str,
        help="Path to the json file with the arguments used for training.",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, help="Path to the hdf5 checkpoint file."
    )
    parser.add_argument("--annot-fn", type=str, help="Path to the annotations file.")
    parser.add_argument(
        "--test-list", type=str, help="Path to a list of the test PDB examples."
    )
    args = parser.parse_args()
    print(args)
    # model_name = "GCN-PDB_MF"

    # load args
    # path = model_name + "_model_params.json"
    path = args.params_path
    with open(path) as f:
        params = json.loads(f.read())
    # override the checkpoint paths
    assert args.lm_model_name.endswith(
        params["lm_model_name"]
    ), "the name of the language model is wrong"
    params["lm_model_name"] = args.lm_model_name
    params["model_name"] = args.checkpoint_path
    params[
        "annot_fn"
    ] = args.annot_fn  # /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv
    params[
        "test_list"
    ] = args.test_list  # /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_test.csv

    # load annotations
    if params["ontology"] == "ec":
        prot2annot, goterms, gonames, counts = load_EC_annot(params["annot_fn"])
    else:
        prot2annot, goterms, gonames, counts = load_GO_annot(params["annot_fn"])

    goterms = params["goterms"]
    gonames = params["gonames"]
    output_dim = len(goterms)

    model = DeepFRI(
        output_dim=output_dim,
        n_channels=26,
        gc_dims=params["gc_dims"],
        fc_dims=params["fc_dims"],
        lr=params["lr"],
        drop=params["dropout"],
        l2_reg=params["l2_reg"],
        gc_layer=params["gc_layer"],
        lm_model_name=params["lm_model_name"],
        model_name_prefix=params["model_name"],
    )
    model.load_model()

    def save():
        pickle.dump(
            {
                "proteins": np.asarray(proteins),
                "Y_pred": np.concatenate(Y_pred, axis=0),
                "Y_true": np.concatenate(Y_true, axis=0),
                "ontology": params["ontology"],
                "goterms": goterms,
                "gonames": gonames,
            },
            open(params["model_name"] + "_subset_results.pckl", "wb"),
        )

    h5_path = "preprocessing/updated_full_dataset.h5"
    ecgo = {"MF": "GO", "BP": "GO", "CC": "GO", "EC": "EC"}[params["ontology"].upper()]
    with h5py.File(h5_path, "r", libver="latest") as hf:
        subset = set(
            map(lambda x: x.decode("ascii"), list(hf[f"{ecgo}_split/test_ids"]))
        )

    Y_pred = []
    Y_true = []
    proteins = []
    path = "/app/DeepFRI/preprocessing/data/annot_pdb_chains_npz/"
    with open(params["test_list"], mode="r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        rows = list(csv_reader)
    header, *rows = rows
    # next(csv_reader, None)  # header
    missing = []
    count = 0
    for row in tqdm(rows):
        count += 1
        if count % 100 == 0:
            save()
        prot = row[0]
        if prot in subset:
            fp = path + prot + ".npz"
            if os.path.isfile(fp):
                cmap = np.load(fp)
                sequence = str(cmap["seqres"])
                Ca_dist = cmap["C_alpha"]

                A = np.double(Ca_dist < params["cmap_thresh"])
                S = seq2onehot(sequence)

                # ##
                S = S.reshape(1, *S.shape)
                A = A.reshape(1, *A.shape)

                try:
                    pred = model.predict([A, S]).reshape(1, output_dim)
                except Exception as e:
                    print(f"{prot} failed!\n\t{e}")

                # results
                proteins.append(prot)
                Y_pred.append(pred)
                Y_true.append(
                    prot2annot[prot][params["ontology"]].reshape(1, output_dim)
                )
            else:
                missing.append(fp)
                print(fp)

    print(missing)
    with open(f"{params['ontology'].upper()}_missing.json", "w") as f:
        f.write(json.dumps(missing))
    save()
