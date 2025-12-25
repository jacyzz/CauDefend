from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def iter_jsonl(path: Path, *, limit: int = 0) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj
                n += 1
                if limit and n >= limit:
                    break


def count_jsonl(path: Path, *, limit: int = 0) -> int:
    c = 0
    for _ in iter_jsonl(path, limit=limit):
        c += 1
    return c


class JsonlAtomicWriter:
    """Stream JSONL to a temp file then atomically replace the target."""

    def __init__(self, path: Path):
        self.path = path
        self._tmp_path = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
        self._f: Optional[Any] = None

    def __enter__(self) -> "JsonlAtomicWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self._tmp_path.open("w", encoding="utf-8")
        return self

    def write_obj(self, obj: Dict[str, Any]) -> None:
        if self._f is None:
            raise RuntimeError("JsonlAtomicWriter not opened")
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._f is not None:
                try:
                    self._f.flush()
                except Exception:
                    pass
                try:
                    self._f.close()
                except Exception:
                    pass
        finally:
            self._f = None

        if exc_type is not None:
            try:
                if self._tmp_path.exists():
                    os.remove(self._tmp_path)
            except Exception:
                pass
            return

        os.replace(self._tmp_path, self.path)
