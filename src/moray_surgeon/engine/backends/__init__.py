"""Language backends: the seam between the generic dispatcher and per-language tools.

Importing this package registers the built-in :class:`PythonBackend`; the
dispatcher resolves backends per target file through :func:`backend_for`.
Adding a language = one subclass of :class:`LanguageBackend` + one
:func:`register_backend` call. Nothing else in the engine changes.
"""

from __future__ import annotations

from moray_surgeon.engine.backends.base import (
    LanguageBackend,
    backend_for,
    register_backend,
    registered_suffixes,
)
from moray_surgeon.engine.backends.python_backend import PythonBackend

__all__ = [
    "LanguageBackend",
    "PythonBackend",
    "backend_for",
    "register_backend",
    "registered_suffixes",
]

# Claim the Python suffixes as soon as the backend layer is imported, so the
# dispatcher (and any embedding host) sees a working registry with no setup.
register_backend(PythonBackend())
