import time
import traceback
from multiprocessing import Event, Process, Queue
from multiprocessing.synchronize import Event as EventClass
from queue import Empty, Full
from typing import Callable, Iterable, Iterator, List, Tuple

import jax
import numpy as np

from bio_clip.data.protein_datastructure import ProteinStructureSample
from bio_clip.types import BatchDataWithTokensBioClip
from bio_clip.utils.utils import get_logger

logger = get_logger(__name__)


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
        identifiers: List,
        batch_dims: Tuple,
        processing_fn: Callable,
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
        if self._params.num_process == 0:
            raise NotImplementedError
        self._processing_fn = processing_fn
        self._identifiers = identifiers
        self._batch_dims = batch_dims

        self._processes_should_exit: EventClass = Event()
        self._identifier_queue: Queue = Queue()
        self._sample_queue: Queue = Queue()
        self._batch_queue: Queue = Queue()
        self._background_processes: List[Process] = []

    def __enter__(self) -> "BioClipDataloader":
        self._identifier_queue = Queue()
        self._sample_queue = Queue(maxsize=np.prod(self._batch_dims) * 4)
        self._batch_queue = Queue(maxsize=self._params.prefetch_factor)
        self.exceptions = []

        for _id in self._identifiers:
            self._identifier_queue.put(_id, timeout=0.5)

        self._background_processes = []

        self._background_processes.extend(
            [
                Process(
                    target=_generate_samples,
                    args=(
                        self._identifier_queue,
                        self._sample_queue,
                        self._processing_fn,
                        self._processes_should_exit,
                        self._params.max_num_consecutive_errors,
                        self.exceptions,
                    ),
                    daemon=True,
                )
                for _ in range(self._params.num_process)
            ]
        )
        self.launch_batch_process()

        for process in self._background_processes:
            process.start()

        return self

    def launch_batch_process(self):
        self._background_processes.append(
            Process(
                target=_batch_samples,
                args=(
                    self._identifier_queue,
                    self._sample_queue,
                    self._batch_queue,
                    self._batch_dims,
                    self._processes_should_exit,
                    self.exceptions,
                ),
                daemon=True,
            ),
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        self._processes_should_exit.set()
        start_time = time.perf_counter()
        while any([process.is_alive() for process in self._background_processes]):
            # queues need to be flushed for all processes to exit
            all_queues: List[Queue] = [
                self._identifier_queue,
                self._sample_queue,
                self._batch_queue,
            ]
            for my_queue in all_queues:
                try:
                    my_queue.get_nowait()
                except Empty:
                    self.exceptions.append("failed on my_queue.get_nowait()")

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

        consective_errors = 0
        while True:
            try:
                out = self._batch_queue.get(timeout=1.0)
                if out is None:
                    break
                consective_errors = 0
                yield out
            except Empty:
                self.exceptions.append(
                    "failed on: out = self._batch_queue.get(timeout=5.0)"
                )
                consective_errors += 1
                if consective_errors > 10:
                    break
                print(">> Tried to yield when queue was empty.")
        self._processes_should_exit.set()

    def __len__(self):
        return len(self._identifiers) // np.prod(self._batch_dims)


def _batch_samples(
    identifier_queue: Queue,
    sample_queue: Queue,
    batch_queue: Queue,
    batch_dims: Tuple[int, ...],
    processes_should_exit: EventClass,
    exceptions: List,
) -> None:
    """Adds to the batch queue from the sample queue."""
    print(f"\n\nBATCHDIMS: {batch_dims}\n\n")
    batch_size = np.prod(batch_dims)

    while True:
        if processes_should_exit.is_set():
            return None

        ids_of_batch = []
        batch_of_samples: List[BatchDataWithTokensBioClip] = []
        pad = False
        while len(batch_of_samples) < batch_size:

            if processes_should_exit.is_set():
                return None

            try:
                identifier, sample = sample_queue.get(timeout=0.5)
                ids_of_batch.append(identifier)
                batch_of_samples.append(sample)
            except Empty:
                exceptions.append(
                    "failed on: identifier, sample = sample_queue.get(timeout=0.5)"
                )
                print("_batch_samples: sample queue empty!")
                pad = True
                break
        if len(batch_of_samples) == 0:
            batch_queue.put(None, timeout=0.5)
            return None

        if pad and len(batch_of_samples) < batch_size:
            mask = (
                (np.arange(batch_size) < len(batch_of_samples))
                .astype(bool)
                .reshape(*batch_dims)
            )
            batch_of_samples += [batch_of_samples[-1]] * (
                batch_size - len(batch_of_samples)
            )
        else:
            mask = np.ones(batch_dims, dtype=bool)

        batch = jax.tree_map(
            lambda *x: np.stack(x).reshape((*batch_dims, *x[0].shape)),
            *batch_of_samples,
        )
        while True:
            try:
                batch_queue.put((ids_of_batch, batch, mask), timeout=0.5)
                break
            except Full:
                exceptions.append(
                    "failed on: batch_queue.put((ids_of_batch, batch, mask), "
                    "timeout=0.5)"
                )
                if processes_should_exit.is_set():
                    return None


def _generate_samples(
    id_queue: Queue,
    sample_queue: Queue,
    processing_fn: Callable[[ProteinStructureSample], BatchDataWithTokensBioClip],
    processes_should_exit: EventClass,
    max_num_consecutive_errors: int,
    exceptions: List,
    verbose: bool = False,
) -> None:
    """Adds to the sample queue"""
    num_consecutive_errors = 0
    while True:
        if processes_should_exit.is_set():
            return None

        try:
            identifier = id_queue.get(timeout=0.5)
        except Empty:
            exceptions.append("failed on: identifier = id_queue.get(timeout=0.5)")
            if processes_should_exit.is_set():
                return None
            continue

        # this must be wrapped in a try statement, because each error will kill a
        # different process.
        try:
            sample = processing_fn(identifier)

            # filtering is now done within the processing_fn
            if sample is None:
                continue

            processed_sample = jax.tree_map(lambda x: np.array(x), sample)
        except Exception as e:
            exceptions.append(f"failed on:sample = processing_fn(identifier) with {e}")
            num_consecutive_errors += 1
            if num_consecutive_errors >= max_num_consecutive_errors:
                raise
            if verbose:
                logger.warning(
                    f"Failed to process sample {identifier}. "
                    f"Exact error: {str(traceback.format_exc())}."
                )
            continue

        while True:
            try:
                sample_queue.put((identifier, processed_sample), timeout=0.5)
                break
            except Full:
                exceptions.append(
                    "failed on: sample_queue.put((identifier, processed_sample), "
                    "timeout=0.5)"
                )
                if processes_should_exit.is_set():
                    return None
