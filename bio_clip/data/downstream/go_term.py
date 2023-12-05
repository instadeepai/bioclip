import copy
import subprocess
from functools import partial

import h5py
import numpy as np

from bio_clip.data.downstream.shared import (
    alphafold_to_legacy_bioclip,
    filter_out_sample,
    get_processing_function,
)
from bio_clip.model.esm.esm_haiku import get_pretrained_esm_model
from bio_clip.train.downstream.dataloader import BioClipDataloader
from bio_clip.types import BatchDataBioClip, FunctionPredictionBatch


def go_h5_extract_sample_fn(
    identifier,
    h5_path,
    ecgo_onto,
    bioclip_preprocess_fn,
    filter_sample=None,
    verbose=False,
):
    # Note that we match the format used in AlphaFold
    # To match the BioCLIP format, we don't use b_factor, chain_ix, but we don't have
    # resolution, pdb_cluster_size
    with h5py.File(h5_path, "r") as hf:
        # hdf5 groups
        x = hf[f"input_features/{identifier}"]
        y = hf[f"targets/{ecgo_onto}/{identifier}"]
        aatype = np.array(x["aatype"]).astype(np.int32)
        atom37_positions = np.array(x["atom_positions"]).astype(np.float32)
        atom37_gt_exists = np.array(x["atom_mask"]).astype(bool)
        residue_index = np.array(x["residue_index"]).astype(np.int32)
        target = np.array(y).astype(np.int32)

    _sample = alphafold_to_legacy_bioclip(
        identifier,
        aatype,
        residue_index,
        atom37_positions,
        atom37_gt_exists,
        bioclip_preprocess_fn,
        filter_sample,
        verbose,
    )

    sample = FunctionPredictionBatch(
        batch_data_bioclip=BatchDataBioClip(_sample.graph, _sample.tokens[None, :]),
        target=target,
    )
    return sample


def aws_download(cmd):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        msg = e.output.decode("ascii")
        rc = e.returncode
        print((rc, msg))


def prepare_dataloaders(cfg, num_devices):
    h5_path = cfg.training.data.dataset_path
    baselines_path = cfg.training.data.baselines_path

    onto = cfg.training.data.ontology
    main_term = "ec" if onto == "ec" else "go"
    ecgo_onto = f"{main_term}/{onto}"

    with h5py.File(h5_path, "r") as hf:
        identifiers = list(hf[f"targets/{ecgo_onto}"].keys())

    baselines = dict(np.load(baselines_path, allow_pickle=True).items())
    long_name = {
        "ec": "enzyme-commission",
        "bp": "biological-process",
        "cc": "cellular-component",
        "mf": "molecular-function",
    }[onto]
    baseline = baselines[long_name].item()
    test_identifiers = baseline["proteins"].tolist()
    train_identifiers = list(set(identifiers) - set(test_identifiers))

    fixes_sizes = cfg.training.data.fixed_sizes
    # Max residue has -2 to account for beginning and end-of-sequence token,
    filter_out_sample_fn = partial(
        filter_out_sample,
        min_number_valid_residues=fixes_sizes.graph_max_neighbor,
        max_number_residues=fixes_sizes.maximum_padding - 2,
    )
    number_of_transformer_blocks = int(
        (cfg.model.plm.esm_model_name).split("_")[1].replace("t", "")
    )
    embeddings_layer_to_save = int(
        cfg.training.proportion_esm_layer * number_of_transformer_blocks
    )
    esm_parameters, forward_esm_fn, tokenizer = get_pretrained_esm_model(
        cfg.model.plm.esm_model_name,
        embeddings_layers_to_save=[embeddings_layer_to_save],
    )
    bioclip_legacy_save_to_bioclip_training_data = get_processing_function(
        cfg, tokenizer, fixes_sizes.maximum_padding, fixes_sizes.graph_max_neighbor
    )
    processing_fn = partial(
        go_h5_extract_sample_fn,
        h5_path=h5_path,
        ecgo_onto=ecgo_onto,
        bioclip_preprocess_fn=bioclip_legacy_save_to_bioclip_training_data,
        filter_sample=filter_out_sample_fn,
    )
    b_cfg = cfg.training.batch
    dl_cfg = copy.copy(cfg.training.data.dataloader)
    trn_bds = (num_devices, b_cfg.num_per_device_update)

    def get_train_dataloader():
        return BioClipDataloader(
            dl_cfg,
            train_identifiers,
            trn_bds,
            processing_fn,
        )

    inf_bds = (num_devices, b_cfg.num_per_device_inference)

    def get_test_dataloader():
        return BioClipDataloader(
            dl_cfg,
            test_identifiers,
            inf_bds,
            processing_fn,
        )

    return get_train_dataloader, get_test_dataloader, baseline
