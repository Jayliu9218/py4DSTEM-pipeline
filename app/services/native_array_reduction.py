from __future__ import annotations

from app.services.array_reduction import PythonReductionBackend


class NativeReductionBackend(PythonReductionBackend):
    """Placeholder native backend hook.

    This class intentionally delegates to the Python implementation until a
    compiled extension is added. Keeping the module importable lets packaging
    and backend-selection paths stabilize before introducing C++ build risk.
    """

    name = "native-python-stub"


def create_backend() -> NativeReductionBackend:
    return NativeReductionBackend()
