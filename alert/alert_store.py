import json
from datetime import datetime, timedelta
from pathlib import Path
import threading

_alert_file = Path("evidence/alerts.json")
_alert_file.parent.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()
_MAX_ALERTS = 500


def _load_alerts():
    if not _alert_file.exists():
        return []
    try:
        with open(_alert_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_alerts(alerts):
    with open(_alert_file, "w") as f:
        json.dump(alerts[-_MAX_ALERTS:], f, indent=2)


def add_alert(report_id: str, timestamp: str, score: float, weapons: list,
              persons_count: int, scene: str, camera: str = "", frame_path: str = ""):
    with _lock:
        alerts = _load_alerts()
        alerts.append({
            "id": report_id,
            "time": timestamp,
            "score": round(score, 3),
            "weapons": len(weapons),
            "weapon_types": list(set(w.get("class", "senjata") for w in weapons)),
            "persons": persons_count,
            "scene": scene,
            "camera": camera,
            "frame_path": frame_path,
        })
        _save_alerts(alerts)


def get_alerts(limit=50):
    with _lock:
        alerts = _load_alerts()
        return alerts[-limit:][::-1]


def clear_alerts():
    with _lock:
        if _alert_file.exists():
            _alert_file.unlink()
