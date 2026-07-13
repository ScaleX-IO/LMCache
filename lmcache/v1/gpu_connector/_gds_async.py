# SPDX-License-Identifier: Apache-2.0
"""Platform dispatch for the GDS async backend.

Selects the GDSContext-facing backend from cuFile, hipFile, or uGDS. Automatic
selection chooses cuFile on NVIDIA and hipFile on AMD ROCm; GDS L1 config can
explicitly select uGDS for a raw ``/dev/ugds_drvX`` slab. All implementations
expose an identical API -- :class:`AsyncHandle`, :class:`Submission`, and the
``register_*`` / ``deregister_*`` / ``close_driver`` functions -- so
:mod:`lmcache.v1.gpu_connector.gds_context` remains backend-agnostic.

Selection is by ``torch.version.hip``: a ROCm torch build reports a non-None
HIP version. Importing this shim does not dlopen any GPU IO driver; both
backends bind ``libcufile``/``libhipfile`` lazily on first use.
"""

# Standard
from types import ModuleType
from typing import TYPE_CHECKING, Literal

# Third Party
import torch

BackendName = Literal["auto", "cufile", "hipfile", "ugds"]

# The backend surface re-exported under stable module-level names so callers
# (and test monkeypatches) target this module.
_EXPORTED_NAMES = (
    "AsyncHandle",
    "Submission",
    "close_driver",
    "register_handle",
    "deregister_handle",
    "register_buffer",
    "deregister_buffer",
    "register_stream",
    "deregister_stream",
)


def _load_backend(name: BackendName) -> tuple[str, ModuleType]:
    selected = name
    if selected == "auto":
        selected = "hipfile" if torch.version.hip is not None else "cufile"
    if selected == "cufile":
        # First Party
        from lmcache.v1.gpu_connector import _cufile_async as backend
    elif selected == "hipfile":
        # First Party
        from lmcache.v1.gpu_connector import _hipfile_async as backend
    elif selected == "ugds":
        # First Party
        from lmcache.v1.gpu_connector import _ugds_async as backend
    else:
        raise ValueError(f"unsupported GDS L1 backend: {name}")
    return selected, backend


def _bind_backend_surface(backend: ModuleType) -> None:
    """Rebind every exported name to the given backend module."""
    for name in _EXPORTED_NAMES:
        globals()[name] = getattr(backend, name)


if TYPE_CHECKING:
    # Static surface for type checkers; every backend exposes the same names.
    # First Party
    from lmcache.v1.gpu_connector import _cufile_async as _backend
    from lmcache.v1.gpu_connector._cufile_async import (
        AsyncHandle as AsyncHandle,
        Submission as Submission,
        close_driver as close_driver,
        deregister_buffer as deregister_buffer,
        deregister_handle as deregister_handle,
        deregister_stream as deregister_stream,
        register_buffer as register_buffer,
        register_handle as register_handle,
        register_stream as register_stream,
    )
else:
    _selected_backend, _backend = _load_backend("auto")
    _bind_backend_surface(_backend)


def select_backend(name: BackendName) -> str:
    """Select the process-global GDS L1 implementation.

    Args:
        name: Explicit backend name, or ``auto`` for platform selection.

    Returns:
        The resolved backend name.
    """
    global _backend
    global _selected_backend

    _selected_backend, _backend = _load_backend(name)
    _bind_backend_surface(_backend)
    return _selected_backend
