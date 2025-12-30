from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


_TASKS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def get_task(task_id: str) -> Dict[str, Any]:
    with _LOCK:
        t = _TASKS.get(task_id)
        return dict(t) if isinstance(t, dict) else {}


def list_tasks(*, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    with _LOCK:
        items: List[Dict[str, Any]] = []
        for _tid, t in _TASKS.items():
            if not isinstance(t, dict):
                continue
            if kind and t.get("kind") != kind:
                continue
            items.append(dict(t))
        items.sort(key=lambda x: float(x.get("created_at", 0.0) or 0.0), reverse=True)
        return items


def update_task(task_id: str, updates: Dict[str, Any]):
    now = time.time()
    with _LOCK:
        if task_id not in _TASKS:
            _TASKS[task_id] = {}
        _TASKS[task_id].update(dict(updates))
        _TASKS[task_id].setdefault("created_at", now)
        _TASKS[task_id]["updated_at"] = now


def create_task(
    task_id: str,
    *,
    kind: str,
    provider: str,
    input_path: str,
    output_path: str,
    model: str,
    total: int = 0,
):
    now = time.time()
    with _LOCK:
        _TASKS[task_id] = {
            "task_id": task_id,
            "kind": kind,
            "provider": provider,
            "model": model,
            "input_path": input_path,
            "output_path": output_path,
            "status": "pending",
            "current": 0,
            "total": total,
            "error": None,
            "result": None,
            "cancel_requested": False,
            "created_at": now,
            "updated_at": now,
        }


def request_cancel(task_id: str) -> bool:
    with _LOCK:
        t = _TASKS.get(task_id)
        if not isinstance(t, dict):
            return False
        t["cancel_requested"] = True
        t["updated_at"] = time.time()
        return True

