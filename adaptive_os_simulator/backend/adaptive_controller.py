from __future__ import annotations

from typing import List, Dict, Any
import statistics

from .utils import Process
from .simulator import Scheduler
from .ml_model import extract_features_from_processes, load_model


def rule_based_decider(current_time: float, procs: List[Process]) -> Dict[str, Any]:
    if not procs:
        return {"policy": Scheduler.RR, "reason": "no processes"}

    remaining_times = [p.remaining_time for p in procs]
    priorities = [p.priority for p in procs]
    mean_rt = sum(remaining_times) / len(remaining_times)
    short_jobs = len([t for t in remaining_times if t <= max(1.0, 0.5 * mean_rt)])
    prio_var = statistics.pvariance(priorities) if len(priorities) > 1 else 0.0

    if prio_var > 2.0:
        return {"policy": Scheduler.PRIORITY, "reason": "high priority variance"}
    if short_jobs >= max(2, len(procs) // 2):
        return {"policy": Scheduler.SJF, "reason": "many short jobs"}
    # Small queue, low variance and longer jobs -> FCFS is fine
    if len(procs) <= 3 and mean_rt >= 3.0 and prio_var < 0.5:
        return {"policy": Scheduler.FCFS, "reason": "small queue with similar priorities"}
    if len(procs) >= 6:
        return {"policy": Scheduler.RR, "reason": "many jobs, fair sharing"}
    # default to SRTF for responsiveness on mixed loads
    return {"policy": Scheduler.SRTF, "reason": "default responsiveness"}


def ml_driven_decider(current_time: float, procs: List[Process]) -> Dict[str, Any]:
    # Backward-compatible default: heuristic if no model is provided
    return heuristic_ml_decider(current_time, procs)


_MODEL_CACHE = {}


def get_model_decider(model_path: str):
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = load_model(model_path)

    model = _MODEL_CACHE[model_path]

    def _decider(current_time: float, procs: List[Process]) -> Dict[str, Any]:
        if not procs:
            return {"policy": Scheduler.RR, "reason": "no processes"}
        feats = extract_features_from_processes(procs)
        X = [[feats[k] for k in sorted(feats.keys())]]
        # Ensure feature order matches training
        # We sort both at train-time and here by FEATURE_NAMES order; training script uses FEATURE_NAMES
        X = [[feats[k] for k in FEATURE_NAMES]]
        pred = model.predict(X)[0]
        return {"policy": str(pred), "reason": "ml: model prediction"}

    return _decider


def heuristic_ml_decider(current_time: float, procs: List[Process]) -> Dict[str, Any]:
    num_procs = len(procs)
    if num_procs == 0:
        return {"policy": Scheduler.RR, "reason": "no processes"}
    mean_remaining = sum(p.remaining_time for p in procs) / num_procs
    if mean_remaining < 3.0:
        return {"policy": Scheduler.SJF, "reason": "ml: short mean remaining"}
    if any(p.priority <= 1 for p in procs):
        return {"policy": Scheduler.PRIORITY, "reason": "ml: critical priorities present"}
    if num_procs > 8:
        return {"policy": Scheduler.RR, "reason": "ml: high load fair sharing"}
    return {"policy": Scheduler.SRTF, "reason": "ml: default"}


