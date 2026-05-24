from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .filter_metrics import FilterMetricConfig, FilterMetrics, extract_filter_metrics
from .filter_targets import (
    FilterTargetSpec,
    MetricErrorVector,
    compute_metric_errors,
    is_target_satisfied,
    load_filter_target,
)
from .touchstone import load_touchstone


def _serialize_metrics(metrics: FilterMetrics) -> dict[str, float | None]:
    raw = metrics.to_dict()
    converted: dict[str, float | None] = {}
    for key, value in raw.items():
        converted[key] = value
        if value is None:
            continue
        if key.endswith("_freq_hz"):
            converted[key.removesuffix("_hz") + "_ghz"] = value / 1e9
        elif key.endswith("_3db_hz"):
            converted[key.removesuffix("_hz") + "_mhz"] = value / 1e6
    return converted


@dataclass(slots=True)
class TouchstoneAnalysis:
    source_path: Path
    metrics: FilterMetrics
    target: FilterTargetSpec | None = None
    errors: MetricErrorVector | None = None
    target_satisfied: bool | None = None
    output_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_path": str(self.source_path),
            "metrics": _serialize_metrics(self.metrics),
        }
        if self.target is not None:
            payload["target"] = self.target.to_dict()
        if self.errors is not None:
            payload["errors"] = self.errors.to_dict()
        if self.target_satisfied is not None:
            payload["target_satisfied"] = self.target_satisfied
        if self.output_path is not None:
            payload["output_path"] = str(self.output_path)
        return payload


def analyze_touchstone(
    s2p_path: str | Path,
    *,
    target_path: str | Path | None = None,
    config: FilterMetricConfig | None = None,
    output_path: str | Path | None = None,
) -> TouchstoneAnalysis:
    source_path = Path(s2p_path).resolve()
    data = load_touchstone(source_path)
    metrics = extract_filter_metrics(data, config)

    target: FilterTargetSpec | None = None
    errors: MetricErrorVector | None = None
    target_satisfied: bool | None = None
    if target_path is not None:
        target = load_filter_target(target_path)
        errors = compute_metric_errors(metrics, target)
        target_satisfied = is_target_satisfied(metrics, target)

    resolved_output_path = None if output_path is None else Path(output_path).resolve()
    analysis = TouchstoneAnalysis(
        source_path=source_path,
        metrics=metrics,
        target=target,
        errors=errors,
        target_satisfied=target_satisfied,
        output_path=resolved_output_path,
    )
    if resolved_output_path is not None:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return analysis
