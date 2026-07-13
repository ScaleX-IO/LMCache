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

# A static type checker analyzes the TYPE_CHECKING branch only (one ``_backend``
# binding, so no ``no-redef``); at runtime the ``elif``/``else`` pick the real
# backend by platform.
BackendName = Literal["auto", "cufile", "hipfile", "ugds"]


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


if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector import _cufile_async as _backend
else:
    _selected_backend, _backend = _load_backend("auto")


def select_backend(name: BackendName) -> str:
    """Select the process-global GDS L1 implementation.

    Args:
        name: Explicit backend name, or ``auto`` for platform selection.

    Returns:
        The resolved backend name.
    """
    global _backend
    global _selected_backend
    global AsyncHandle
    global Submission
    global close_driver
    global register_handle
    global deregister_handle
    global register_buffer
    global deregister_buffer
    global register_stream
    global deregister_stream

    _selected_backend, _backend = _load_backend(name)
    AsyncHandle = _backend.AsyncHandle
    Submission = _backend.Submission
    close_driver = _backend.close_driver
    register_handle = _backend.register_handle
    deregister_handle = _backend.deregister_handle
    register_buffer = _backend.register_buffer
    deregister_buffer = _backend.deregister_buffer
    register_stream = _backend.register_stream
    deregister_stream = _backend.deregister_stream
    return _selected_backend


# Re-export the selected backend's surface under stable names so callers
# (and test monkeypatches) target this module.
AsyncHandle = _backend.AsyncHandle
Submission = _backend.Submission
close_driver = _backend.close_driver
register_handle = _backend.register_handle
deregister_handle = _backend.deregister_handle
register_buffer = _backend.register_buffer
deregister_buffer = _backend.deregister_buffer
register_stream = _backend.register_stream
deregister_stream = _backend.deregister_stream
