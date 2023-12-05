import argparse
import os
from pathlib import Path

from data_processing import BioClipDataProcessingConfig, build_dataset_bioclip


def get_parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="cath",
    )
    parser.add_argument("--data-dir", type=str, help="data directory")
    parser.add_argument("--cache-path", type=str, help="directory to save graphs")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=96,
        required=False,
        help="Number of workers for data preprocessing",
    )
    parser.add_argument(
        "--graph-max-neighbor",
        type=int,
        default=10,
        required=False,
        help="Only for data caching and inference.",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        required=False,
        help="Proportion of the training dataset that we preprocess.",
    )
    parser.add_argument(
        "--graph-residue-loc-is-alphac",
        default=True,
        action="store_true",
        help="whether to use coordinates of alphaC or avg of \
            atom locations as the representative residue location."
        "Only for data caching and inference.",
    )
    parser.add_argument("--pocket_cutoff", type=float, default=8.0, required=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_parser_args()
    if args.data_dir:
        data_dir = args.data_dir
    else:
        if args.data == "cath":
            data_dir = "data/bio_clip/CATH/non-redundant-data-sets/dompdb"
        elif args.data == "pdb":
            data_dir = "data/bio_clip/PDB"
        elif args.data == "alphafold":
            data_dir = "data/bio_clip/alphafold"
        else:
            raise NotImplementedError

    if args.cache_path:
        cache_path = [args.cache_path]
    else:
        cache_path = [
            os.path.join(
                "data",
                "bio_clip",
                f"{args.data}/",
            )
        ]

    data_config = BioClipDataProcessingConfig(
        cache_path=cache_path,
        n_jobs=args.n_jobs,
        pocket_cutoff=args.pocket_cutoff,
        graph_max_neighbor=args.graph_max_neighbor,
        graph_residue_loc_is_alphac=args.graph_residue_loc_is_alphac,
    )

    # Directory may exist!
    os.makedirs(cache_path[0], exist_ok=True)

    # Build dataset
    build_dataset_bioclip(
        data_config=data_config,
        raw_data_path=Path(data_dir),
    )
