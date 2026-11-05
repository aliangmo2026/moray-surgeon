"""Language backends: the seam between the generic dispatcher and per-language tools.

Importing this package registers the built-in :class:`PythonBackend` and
:class:`TypeScriptBackend`; the dispatcher resolves backends per target file
through :func:`backend_for`. Adding a language = one subclass of
:class:`LanguageBackend` + one :func:`register_backend` call. Nothing else in the
engine changes.

The TypeScript backend is registered eagerly but only reaches Node.js when an
atom actually runs, so importing this package works on machines without Node.
"""

from __future__ import annotations

from moray_surgeon.engine.backends.base import (
    LanguageBackend,
    backend_for,
    register_backend,
    registered_suffixes,
)
from moray_surgeon.engine.backends.python_backend import PythonBackend
from moray_surgeon.engine.backends.typescript_backend import TypeScriptBackend

__all__ = [
    "LanguageBackend",
    "PythonBackend",
    "TypeScriptBackend",
    "backend_for",
    "register_backend",
    "registered_suffixes",
]

# Claim the built-in suffixes as soon as the backend layer is imported, so the
# dispatcher (and any embedding host) sees a working registry with no setup.
register_backend(PythonBackend())
register_backend(TypeScriptBackend())
