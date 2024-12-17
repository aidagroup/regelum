from contextlib import contextmanager
import threading

from regelum.utils.logger import logger

_SYMBOLIC_INFERENCE_ACTIVE = threading.local()
_SYMBOLIC_INFERENCE_ACTIVE.value = False


@contextmanager
def symbolic_inference():
    """A context manager that enables symbolic inference for State values."""
    _SYMBOLIC_INFERENCE_ACTIVE.value = True
    try:
        yield
    finally:
        _SYMBOLIC_INFERENCE_ACTIVE.value = False
