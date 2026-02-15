"""Registry for diarization backends."""

from ..backends.diarization import DiarizationBackend

DIARIZATION_BACKENDS: dict[str, type[DiarizationBackend]] = {}


def register_diarization_backend(name: str, backend_class: type[DiarizationBackend]):
    """Register a diarization backend."""
    DIARIZATION_BACKENDS[name] = backend_class


def get_diarization_backend(name: str) -> type[DiarizationBackend]:
    """Get a diarization backend class by name."""
    if name not in DIARIZATION_BACKENDS:
        available = list(DIARIZATION_BACKENDS.keys())
        raise KeyError(f"Unknown diarization backend '{name}'. Available: {available}")
    return DIARIZATION_BACKENDS[name]
