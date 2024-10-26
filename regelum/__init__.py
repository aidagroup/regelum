from contextlib import contextmanager
import threading

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
