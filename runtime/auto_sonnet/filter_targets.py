from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .filter_metrics import FilterMetrics


@dataclass(slots=True)
class MetricToleranceSpec:
    target: float
    tolerance: float = 0.0


@dataclass(slots=True)
class FilterTargetSpec:
    name: str | None = None
    center_freq_hz: MetricToleranceSpec | None = None
    bandwidth_hz: MetricToleranceSpec | None = None
    high_side_zero_freq_hz: MetricToleranceSpec | None = None
    max_insertion_loss_db: float | None = None
    min_return_loss_db: float | None = None
    min_high_side_zero_depth_db: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MetricErrorVector:
    center_freq_error_hz: float | None
    bandwidth_error_hz: float | None
    high_side_zero_freq_error_hz: float | None
    insertion_loss_error_db: float | None
    return_loss_error_db: float | None
    high_side_zero_depth_error_db: float | None
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_metric_tolerance(raw: Any, context: str) -> MetricToleranceSpec | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        if "target" not in raw:
            raise ValueError(f"Metric tolerance object '{context}' must contain 'target'")
        return MetricToleranceSpec(
            target=float(raw["target"]),
            tolerance=float(raw.get("tolerance", 0.0)),
        )
    return MetricToleranceSpec(target=float(raw), tolerance=0.0)


def load_filter_target(path: str | Path) -> FilterTargetSpec:
    target_path = Path(path).resolve()
    data = json.loads(target_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Filter target spec must contain a top-level JSON object")

    return FilterTargetSpec(
        name=data.get("name"),
        center_freq_hz=_parse_metric_tolerance(data.get("center_freq_hz"), "center_freq_hz"),
        bandwidth_hz=_parse_metric_tolerance(data.get("bandwidth_hz"), "bandwidth_hz"),
        high_side_zero_freq_hz=_parse_metric_tolerance(data.get("high_side_zero_freq_hz"), "high_side_zero_freq_hz"),
        max_insertion_loss_db=None if data.get("max_insertion_loss_db") is None else float(data["max_insertion_loss_db"]),
        min_return_loss_db=None if data.get("min_return_loss_db") is None else float(data["min_return_loss_db"]),
        min_high_side_zero_depth_db=None
        if data.get("min_high_side_zero_depth_db") is None
        else float(data["min_high_side_zero_depth_db"]),
    )


def _signed_error(actual: float | None, spec: MetricToleranceSpec | None) -> float | None:
    if actual is None or spec is None:
        return None
    return actual - spec.target


def compute_metric_errors(metrics: FilterMetrics, target: FilterTargetSpec) -> MetricErrorVector:
    insertion_loss_db = -metrics.main_peak_s21_db
    return_loss_db = -metrics.best_s11_db
    high_side_zero_depth_db = None if metrics.high_side_zero_s21_db is None else -metrics.high_side_zero_s21_db

    center_freq_error_hz = _signed_error(metrics.main_peak_freq_hz, target.center_freq_hz)
    bandwidth_error_hz = _signed_error(metrics.bandwidth_3db_hz, target.bandwidth_hz)
    high_side_zero_freq_error_hz = _signed_error(metrics.high_side_zero_freq_hz, target.high_side_zero_freq_hz)
    insertion_loss_error_db = (
        None if target.max_insertion_loss_db is None else insertion_loss_db - target.max_insertion_loss_db
    )
    return_loss_error_db = None if target.min_return_loss_db is None else target.min_return_loss_db - return_loss_db
    high_side_zero_depth_error_db = (
        None
        if target.min_high_side_zero_depth_db is None or high_side_zero_depth_db is None
        else target.min_high_side_zero_depth_db - high_side_zero_depth_db
    )

    success = True

    if target.center_freq_hz is not None:
        success = success and center_freq_error_hz is not None and abs(center_freq_error_hz) <= target.center_freq_hz.tolerance
    if target.bandwidth_hz is not None:
        success = success and bandwidth_error_hz is not None and abs(bandwidth_error_hz) <= target.bandwidth_hz.tolerance
    if target.high_side_zero_freq_hz is not None:
        success = success and high_side_zero_freq_error_hz is not None and abs(high_side_zero_freq_error_hz) <= target.high_side_zero_freq_hz.tolerance
    if target.max_insertion_loss_db is not None:
        success = success and insertion_loss_error_db is not None and insertion_loss_error_db <= 0.0
    if target.min_return_loss_db is not None:
        success = success and return_loss_error_db is not None and return_loss_error_db <= 0.0
    if target.min_high_side_zero_depth_db is not None:
        success = success and high_side_zero_depth_error_db is not None and high_side_zero_depth_error_db <= 0.0

    return MetricErrorVector(
        center_freq_error_hz=center_freq_error_hz,
        bandwidth_error_hz=bandwidth_error_hz,
        high_side_zero_freq_error_hz=high_side_zero_freq_error_hz,
        insertion_loss_error_db=insertion_loss_error_db,
        return_loss_error_db=return_loss_error_db,
        high_side_zero_depth_error_db=high_side_zero_depth_error_db,
        success=success,
    )


def is_target_satisfied(metrics: FilterMetrics, target: FilterTargetSpec) -> bool:
    return compute_metric_errors(metrics, target).success
