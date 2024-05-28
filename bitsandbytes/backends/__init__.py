from typing import Dict

from bitsandbytes.backends.base import Backend

backends: Dict[str, Backend] = {}


def register_backend(backend_name: str, backend_instance: Backend):
    backends[backend_name.lower()] = backend_instance


def ensure_backend_is_available(device_type: str):
    """Check if a backend is available for the given device type."""
    if device_type.lower() not in backends:
        raise NotImplementedError(f"Device backend for {device_type} is currently not supported.")
