import contextlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Iterator, List, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict, config_dict
from omegaconf import DictConfig, ListConfig

root_loggers: List[str] = []
T = TypeVar("T")


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None) -> Iterator[str]:
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_logger(name: str) -> logging.Logger:
    """Configuring logging.

    The level is configured thanks to the environment variable 'LOG_LEVEL'
        (default 'INFO').
    """
    global root_loggers

    name_root_logger = name.split(".")[0]

    if name_root_logger not in root_loggers:
        logger = logging.getLogger(name_root_logger)
        logger.propagate = False
        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | %(process)d | %(levelname)s | %(module)s:%(funcName)s:"
                "%(lineno)d | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        default_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(default_log_level if default_log_level else "INFO")

        root_loggers.append(name_root_logger)

    return logging.getLogger(name)


logger = get_logger(__name__)


class ThreadedIterable(Iterable[T]):
    """
    This starts a thread in the background to fetch the next element in the wrapped
    iterable in an asynchronous fashion and cache it until next() is called.
    """

    def __init__(self, iterable: Iterable[T]):
        self._iterable = iterable

        self._stop_iteration = False
        self._exception: Optional[Exception] = None
        self._next_element: Optional[T] = None
        self._need_new_element_event = threading.Event()
        self._need_new_element_event.set()
        self._element_is_ready_event = threading.Event()
        self._element_is_ready_event.clear()
        self._thread: Optional[threading.Thread] = None

    def __iter__(self) -> Iterator[T]:
        if self._thread is not None:
            raise RuntimeError("A single iterator can be created from this iterable.")

        self._thread = threading.Thread(
            target=self._background_thread, args=(), daemon=True
        )
        self._thread.start()

        while True:
            self._element_is_ready_event.wait()
            if self._exception is not None:
                raise self._exception

            if self._stop_iteration:
                return

            self._element_is_ready_event.clear()
            next_element = self._next_element
            self._need_new_element_event.set()
            yield next_element  # type: ignore

    def _background_thread(self) -> None:
        data_iterator = iter(self._iterable)

        while True:
            self._need_new_element_event.wait()
            try:
                start_time = time.perf_counter()
                next_element = next(data_iterator)
                logger.info(
                    f"Took {(time.perf_counter() - start_time):.3f} secs to fetch an "
                    "element in the background."
                )
            except StopIteration:
                self._element_is_ready_event.set()
                self._stop_iteration = True
                return
            except Exception as exception:
                self._element_is_ready_event.set()
                self._exception = exception
                return

            self._need_new_element_event.clear()
            self._next_element = next_element
            self._element_is_ready_event.set()


def convert_to_ml_dict(dct: Union[DictConfig, Any]) -> Union[ConfigDict, Any]:
    """
    This function converts the DictConfig returned by Hydra
    into a ConfigDict. The recusion allows to convert
    all the nested DictConfig elements of the config. The recursion stops
    once the reached element is not a DictConfig.
    """
    if not type(dct) is DictConfig:
        if type(dct) is ListConfig:
            return list(dct)
        return dct
    dct_ml = config_dict.ConfigDict()
    for k in list(dct.keys()):
        dct_ml[k] = convert_to_ml_dict(dct[k])
    return dct_ml


def show_git():
    cmd = (
        "cd /app/bio-clip; git config --global --add safe.directory /app/bio-clip; "
        "git log"
    )
    out = subprocess.run(cmd, capture_output=True, shell=True)
    info = "\n".join(out.stdout.decode("utf-8").split("\n")[:5])
    print(info)


def pad(data, desired_size):
    retrieved_batch_size = jax.tree_util.tree_leaves(data)[0].shape[0]
    need = desired_size - retrieved_batch_size
    mask = np.ones(desired_size)
    if need > 0:
        mask[retrieved_batch_size:] = 0
        data = jax.tree_map(
            lambda x: jnp.concatenate(
                [x, jnp.zeros((need,) + x.shape[1:], x.dtype)], axis=0
            ),
            data,
        )
    return data, mask
