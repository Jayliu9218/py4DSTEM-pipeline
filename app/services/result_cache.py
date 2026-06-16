from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResultCacheKey:
    operation: str
    source_id: int
    source_shape: tuple[int, ...]
    parameters: tuple[tuple[str, object], ...]


class ResultCache:
    """Small LRU cache for derived display results."""

    def __init__(self, limit: int = 8) -> None:
        self.limit = max(int(limit), 1)
        self._items: OrderedDict[ResultCacheKey, Any] = OrderedDict()

    def get(self, key: ResultCacheKey) -> Any | None:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: ResultCacheKey, value: Any) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.limit:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()


def make_result_cache_key(
    operation: str,
    source: Any,
    parameters: dict[str, object],
) -> ResultCacheKey:
    data = source if hasattr(source, "shape") else getattr(source, "data", source)
    shape = tuple(int(value) for value in getattr(data, "shape", ()))
    normalized = tuple(sorted((str(key), _normalize_value(value)) for key, value in parameters.items()))
    return ResultCacheKey(operation, id(source), shape, normalized)


def _normalize_value(value: object) -> object:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_normalize_value(item) for item in value)
    return str(value)
