"""Global state for regelum."""

import threading

_SYMBOLIC_INFERENCE_ACTIVE = threading.local()
_SYMBOLIC_INFERENCE_ACTIVE.value = False
