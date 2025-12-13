import threading
from typing import Dict, Any

_TASKS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

def get_task(task_id: str) -> Dict[str, Any]:
    with _LOCK:
        return _TASKS.get(task_id, {})

def update_task(task_id: str, updates: Dict[str, Any]):
    with _LOCK:
        if task_id not in _TASKS:
            _TASKS[task_id] = {}
        _TASKS[task_id].update(updates)

def create_task(task_id: str, total: int = 0):
    with _LOCK:
        _TASKS[task_id] = {
            "status": "pending",
            "current": 0,
            "total": total,
            "error": None,
            "result": None
        }

