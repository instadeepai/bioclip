import dataclasses
import os
import time
import traceback
from multiprocessing import Event, Process, Queue
from multiprocessing.synchronize import Event as EventClass
from queue import Empty, Full
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

import jax
import numpy as np
from cloudpathlib import AnyPath, CloudPath, GSPath, S3Path

from bio_clip.data.parsers import parse_fasta
from bio_clip.data.protein_datastructure import ProteinStructureSample
from bio_clip.types import BatchDataWithTokensBioClip
from bio_clip.utils.utils import get_logger, tmpdir_manager

logger = get_logger(__name__)


@dataclasses.dataclass
class BioClipDataloaderParams:
    filepaths: List[str]  # List of filepaths to ProteinStructureSample stored as npy
    # files. The filepaths can be GCP bucket filepaths (in which case they should start
    # with gs://)
    batch_dims: Tuple[int, ...]  # Batches of training samples will be reshaped to this
    # shape before being outputted (the effective batch size is np.prod(batch_dims)).
    # This is useful for preparing the batches in a shape that matches what jax.pmap and
    # jax.vmap expect.
    shuffle: bool = True  # If True, shuffle the filepaths at the beginning of all
    # epochs.
    prefetch_factor: int = 2  # Number of batches to prefetch in the background.
    max_num_epochs: Optional[int] = None  # The dataset iterator will stop yielding
    # samples after this number of epochs over the filepaths. If None, the iterator
    # never stops yielding examples.
    num_process: int = 1  # Number of parallel processes used to generate individual
    # samples in parallel. If 0, samples are generated sequentially from the main
    # process.
    max_num_consecutive_errors: int = 20  # If processing samples fails for this number
    # of consecutive samples, an exception is raised.
    use_weighted_sampling: bool = False  # whether or not to do weighted sampling.
    weights: Optional[
        np.ndarray
    ] = None  # weights to sample the filepaths in proportion to.

    def __post_init__(self) -> None:
        assert self.filepaths, self.filepaths
        assert np.prod(self.batch_dims) > 0, self.batch_dims
        assert self.prefetch_factor > 0, self.prefetch_factor
        if self.max_num_epochs is not None:
            assert self.max_num_epochs > 0, self.max_num_epochs
        assert self.num_process >= 0, self.num_process
        assert self.max_num_consecutive_errors >= 0, self.max_num_consecutive_errors


def get_filepaths_to_samples(
    fasta_filepath: AnyPath,
    are_pdb_samples: bool,
    s3_sess,
) -> List[str]:

    with tmpdir_manager(base_dir="/tmp") as tmp_dir:
        prefix = "/".join(str(fasta_filepath).split("/")[:3]) + "/"
        if isinstance(fasta_filepath, GSPath):
            local_filepath = os.path.join(tmp_dir, "input.fa")
            logger.info(f"Downloading {str(fasta_filepath)} to {local_filepath}")
            fasta_filepath.download_to(local_filepath)
        elif isinstance(fasta_filepath, S3Path):
            local_filepath = os.path.join(tmp_dir, "input.fa")
            fasta_files3path = s3_sess.S3Path(fasta_filepath)
            fasta_files3path.download_to(local_filepath)
        else:
            prefix = "/app/bio-clip/datasets/pretraining/"
            local_filepath = str(fasta_filepath)

        def makepath(x):
            return f"{prefix}samples/{x}/cif_raw_data.npy"

        logger.info(f"Reading {local_filepath}")
        with open(local_filepath, "r") as file_to_read:
            file_content = file_to_read.read()

        logger.info(f"Parsing {local_filepath}")
        success, error_msg, all_sequence_w_ids = parse_fasta(file_content)
        if not success:
            raise RuntimeError(error_msg)

        logger.info(f"Done parsing {fasta_filepath}")

    if are_pdb_samples:
        return [makepath(seq.id) for seq in all_sequence_w_ids]
    return [makepath(seq.id.split("UniRef50_")[-1]) for seq in all_sequence_w_ids]


class BioClipDataloader(Iterable[BatchDataWithTokensBioClip]):
    """
    This dataloader needs to be used within a context manager to properly shut down
        background processes that parse protein structure samples and generate
        BatchDataWithTokensBioClip objects. Example of use:

    ```
    with BioClipDataloader(
        bioclip_dataloader_params,
        preprocess_sample_fn,
    ) as bioclip_dataloader:
        for data_batch in bioclip_dataloader:
            ...
    ```
    """

    def __init__(
        self,
        params,
        processing_fn: Callable[[ProteinStructureSample], BatchDataWithTokensBioClip],
        filter_out_fn: Optional[Callable[[ProteinStructureSample], bool]] = None,
    ):
        """
        Args:
            params: parametrization of the dataloader.
            processing_fn: function that maps a single protein structure sample parsed
                from one of the files specified in params to a
                BatchDataWithTokensBioClip object.
            filter_out_fn: (optional) a function that takes as an input a protein
                structure sample parsed from one of the files specified in params and
                returns True if and only if the corresponding sample should be
                discarded (e.g. if the sequence is too short or not enough atom
                coordinates are determined, etc...). If not specified, no sample is
                discarded.
        """
        self._params = params
        self.s3_sess = None
        self._processing_fn = processing_fn
        self._filter_out_fn = (
            filter_out_fn if filter_out_fn is not None else lambda _: False
        )

        self._processes_should_exit: EventClass = Event()
        self._filepath_queue: Queue = Queue()
        self._sample_queue: Queue = Queue()
        self._batch_queue: Queue = Queue()
        self._background_processes: List[Process] = []

    def __enter__(self) -> "BioClipDataloader":
        self._filepath_queue = Queue(maxsize=max(2, self._params.num_process))
        self._sample_queue = Queue(maxsize=np.prod(self._params.batch_dims))
        self._batch_queue = Queue(maxsize=self._params.prefetch_factor)

        if self._params.conditional_cluster_sampling_args is not None:
            print("Using conditional cluster sampling")
            self._background_processes = [
                Process(
                    target=_feed_filepath_conditional_cluster_sampling,
                    args=(
                        self._filepath_queue,
                        self._params.filepaths,
                        self._params.conditional_cluster_sampling_args,
                        self._processes_should_exit,
                    ),
                    daemon=True,
                )
            ]
        elif self._params.use_weighted_sampling:
            print("Using weighted sampling")
            self._background_processes = [
                Process(
                    target=_feed_filepath_weighted,
                    args=(
                        self._filepath_queue,
                        self._params.filepaths,
                        self._params.weights,
                        self._processes_should_exit,
                    ),
                    daemon=True,
                )
            ]
        else:
            print("Using uniform sampling")
            self._background_processes = [
                Process(
                    target=_feed_filepath,
                    args=(
                        self._filepath_queue,
                        self._params.filepaths,
                        self._params.shuffle,
                        self._processes_should_exit,
                    ),
                    daemon=True,
                )
            ]

        if self._params.num_process > 0:
            self._background_processes.extend(
                [
                    Process(
                        target=_generate_samples,
                        args=(
                            self._filepath_queue,
                            self._sample_queue,
                            self._processing_fn,
                            self._filter_out_fn,
                            self._processes_should_exit,
                            self._params.max_num_consecutive_errors,
                            self.s3_sess,
                        ),
                        daemon=True,
                    )
                    for _ in range(self._params.num_process)
                ]
            )
            self._background_processes.append(
                Process(
                    target=_batch_samples,
                    args=(
                        self._sample_queue,
                        self._batch_queue,
                        self._params.batch_dims,
                        self._processes_should_exit,
                    ),
                    daemon=True,
                ),
            )

        for process in self._background_processes:
            process.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        self._processes_should_exit.set()
        start_time = time.perf_counter()
        while any([process.is_alive() for process in self._background_processes]):
            # queues need to be flushed for all processes to exit
            all_queues: List[Queue] = [
                self._filepath_queue,
                self._sample_queue,
                self._batch_queue,
            ]
            for my_queue in all_queues:
                try:
                    my_queue.get_nowait()
                except Empty:
                    pass

            if time.perf_counter() - start_time > 20.0:
                logger.error("Failed to join all processes, expect zombie processes")
                for process in self._background_processes:
                    process.terminate()
                break

        self._processes_should_exit.clear()

    def __iter__(self) -> Iterator[BatchDataWithTokensBioClip]:
        if not bool(self._background_processes):
            raise RuntimeError("Must be used within a context manager")

        if self._processes_should_exit.is_set():
            raise RuntimeError("Can be used only once within a context manager")

        batch_size = np.prod(self._params.batch_dims)
        max_num_epochs = self._params.max_num_epochs
        max_num_consecutive_errors = self._params.max_num_consecutive_errors
        num_samples_in_one_epoch = len(self._params.filepaths)

        num_collected_samples = 0
        num_consecutive_errors = 0
        while (
            max_num_epochs is None
            or num_collected_samples < num_samples_in_one_epoch * max_num_epochs
        ):
            if self._params.num_process > 0:
                yield self._batch_queue.get()
            else:
                batch_of_samples: List = []
                filenames_in_batch = []
                while len(batch_of_samples) < batch_size:
                    original_filepath = self._filepath_queue.get()
                    try:
                        processed_sample = _process_sample(
                            AnyPath(original_filepath),
                            self._processing_fn,
                            self._filter_out_fn,
                        )
                    except Exception:
                        num_consecutive_errors += 1
                        if num_consecutive_errors >= max_num_consecutive_errors:
                            raise
                        logger.warning(
                            f"Failed to process sample {original_filepath}. "
                            f"Exact error: {str(traceback.format_exc())}."
                        )
                        continue

                    num_consecutive_errors = 0

                    if processed_sample is not None:
                        filenames_in_batch.append(original_filepath)
                        batch_of_samples.append(processed_sample)

                yield (
                    jax.tree_map(
                        lambda *x: np.stack(x).reshape(
                            (*self._params.batch_dims, *x[0].shape)
                        ),
                        *batch_of_samples,
                    ),
                    filenames_in_batch,
                )

            num_collected_samples += batch_size

        self._processes_should_exit.set()


def _process_sample(
    original_filepath: AnyPath,
    processing_fn: Callable[[ProteinStructureSample], BatchDataWithTokensBioClip],
    filter_out_fn: Callable[[ProteinStructureSample], bool],
    s3_sess,
) -> Optional[BatchDataWithTokensBioClip]:
    with tmpdir_manager(base_dir="/tmp") as tmp_dir:
        if isinstance(original_filepath, CloudPath):
            filepath = os.path.join(tmp_dir, "sample.npy")
            original_filepath.download_to(filepath)

        elif isinstance(original_filepath, S3Path):
            if s3_sess is None:
                raise Exception("s3_sess cannot be None")
            fasta_files3path = s3_sess.S3Path(original_filepath)
            fasta_files3path.download_to(filepath)
        else:
            filepath = str(original_filepath)

        if not os.path.isfile(filepath):
            logger.warning(f"File not found {str(original_filepath)}")
            return None

        sample = ProteinStructureSample.from_file(filepath)

        if filter_out_fn(sample):
            # logger.info(f"Sample {sample.chain_id} has been filtered out.")
            return None

        processed_sample = jax.tree_map(lambda x: np.array(x), processing_fn(sample))
        return processed_sample


def _generate_samples(
    filepath_queue: Queue,
    sample_queue: Queue,
    processing_fn: Callable[[ProteinStructureSample], BatchDataWithTokensBioClip],
    filter_out_fn: Callable[[ProteinStructureSample], bool],
    processes_should_exit: EventClass,
    max_num_consecutive_errors: int,
    s3_sess,
) -> None:
    num_consecutive_errors = 0
    while True:
        if processes_should_exit.is_set():
            return None

        try:
            original_filepath = filepath_queue.get(timeout=0.5)
        except Empty:
            if processes_should_exit.is_set():
                return None
            continue

        try:
            processed_sample = _process_sample(
                AnyPath(original_filepath), processing_fn, filter_out_fn, s3_sess
            )
        except Exception:
            num_consecutive_errors += 1
            if num_consecutive_errors >= max_num_consecutive_errors:
                raise
            logger.warning(
                f"Failed to process sample {original_filepath}. "
                f"Exact error: {str(traceback.format_exc())}."
            )
            continue

        num_consecutive_errors = 0

        if processed_sample is not None:
            while True:
                try:
                    sample_queue.put(processed_sample, timeout=0.5)
                    break
                except Full:
                    if processes_should_exit.is_set():
                        return None


def _batch_samples(
    sample_queue: Queue,
    batch_queue: Queue,
    batch_dims: Tuple[int, ...],
    processes_should_exit: EventClass,
) -> None:
    batch_size = np.prod(batch_dims)

    while True:
        if processes_should_exit.is_set():
            return None

        batch_of_samples: List[BatchDataWithTokensBioClip] = []
        while len(batch_of_samples) < batch_size:
            if processes_should_exit.is_set():
                return None

            try:
                batch_of_samples.append(sample_queue.get(timeout=0.5))
            except Empty:
                if processes_should_exit.is_set():
                    return None
                continue

        batch = jax.tree_map(
            lambda *x: np.stack(x).reshape((*batch_dims, *x[0].shape)),
            *batch_of_samples,
        )
        while True:
            try:
                batch_queue.put(batch, timeout=0.5)
                break
            except Full:
                if processes_should_exit.is_set():
                    return None


def _feed_filepath(
    filepath_queue: Queue,
    all_filepaths: List[str],
    shuffle_filepath: bool,
    processes_should_exit: EventClass,
) -> None:
    num_filepaths = len(all_filepaths)
    while True:
        for filepath_idx in (
            np.random.permutation(num_filepaths)
            if shuffle_filepath
            else range(num_filepaths)
        ):
            while True:
                if processes_should_exit.is_set():
                    return None

                try:
                    filepath_queue.put(all_filepaths[filepath_idx], timeout=0.5)
                    break
                except Full:
                    if processes_should_exit.is_set():
                        return None


def _feed_filepath_weighted(
    filepath_queue: Queue,
    all_filepaths: List[str],
    all_weights: List[float],
    processes_should_exit: EventClass,
) -> None:
    num_filepaths = len(all_filepaths)
    probabilities = np.array(all_weights)
    probabilities /= probabilities.sum()
    while True:
        for filepath_idx in np.random.choice(
            np.arange(num_filepaths),
            p=probabilities,
            size=num_filepaths,
        ):
            while True:
                if processes_should_exit.is_set():
                    return None

                try:
                    filepath_queue.put(all_filepaths[filepath_idx], timeout=0.5)
                    break
                except Full:
                    if processes_should_exit.is_set():
                        return None


def _feed_filepath_conditional_cluster_sampling(
    filepath_queue: Queue,
    all_filepaths: List[str],
    cond_sampling_args,
    processes_should_exit: EventClass,
):
    # num_filepaths = len(all_filepaths) I need:
    # - (clusters, cluster_weights) list of cluster names and list of cluster weights
    # - per_cluster_weights dict[cluster name str --> (filename indices list,
    #  corresponding weights)] --> actually the within cluster weights can be uniform
    (
        clusters,
        cluster_weights,
        cluster2filepaths,
        take_n_seq_per_cluster,
    ) = cond_sampling_args
    cluster_probabilities = np.array(cluster_weights)
    cluster_probabilities /= cluster_probabilities.sum()
    while True:
        for cluster_idx in np.random.choice(
            np.arange(len(cluster_weights)),
            p=cluster_probabilities,
            size=len(cluster_weights),
        ):
            cluster_name = clusters[cluster_idx]
            within_cluster_filepaths = cluster2filepaths[cluster_name]

            for filepath in np.random.choice(
                within_cluster_filepaths,
                size=min(take_n_seq_per_cluster, len(within_cluster_filepaths)),
                replace=False,
            ):
                while True:
                    if processes_should_exit.is_set():
                        return None

                    try:
                        filepath_queue.put(filepath, timeout=0.5)
                        break
                    except Full:
                        if processes_should_exit.is_set():
                            return None
