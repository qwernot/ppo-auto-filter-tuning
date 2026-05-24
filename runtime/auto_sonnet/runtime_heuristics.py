from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable


@dataclass(slots=True)
class RuntimeAnomalyReport:
    elapsed_seconds: float
    reference_seconds: float | None
    ratio_to_reference: float | None
    warning_ratio: float
    severe_ratio: float
    level: str
    message: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "reference_seconds": self.reference_seconds,
            "ratio_to_reference": self.ratio_to_reference,
            "warning_ratio": self.warning_ratio,
            "severe_ratio": self.severe_ratio,
            "level": self.level,
            "message": self.message,
        }


def build_runtime_reference(samples: Iterable[float]) -> float | None:
    positives = [float(item) for item in samples if float(item) > 0.0]
    if not positives:
        return None
    return float(median(positives))


def assess_runtime_anomaly(
    elapsed_seconds: float,
    reference_seconds: float | None,
    *,
    warning_ratio: float = 2.0,
    severe_ratio: float = 3.0,
) -> RuntimeAnomalyReport:
    elapsed = max(float(elapsed_seconds), 0.0)
    if reference_seconds is None or float(reference_seconds) <= 0.0:
        return RuntimeAnomalyReport(
            elapsed_seconds=elapsed,
            reference_seconds=None,
            ratio_to_reference=None,
            warning_ratio=warning_ratio,
            severe_ratio=severe_ratio,
            level="insufficient_reference",
            message=None,
        )

    reference = float(reference_seconds)
    ratio = elapsed / reference if reference > 0.0 else None
    if ratio is None:
        level = "insufficient_reference"
        message = None
    elif ratio >= severe_ratio:
        level = "severe"
        message = (
            f"Simulation runtime is {ratio:.2f}x the reference. "
            "This can indicate a convergence problem, a mesh issue, or an unusually difficult parameter point."
        )
    elif ratio >= warning_ratio:
        level = "warning"
        message = (
            f"Simulation runtime is {ratio:.2f}x the reference. "
            "This may indicate a convergence slowdown or a problematic parameter combination."
        )
    else:
        level = "ok"
        message = None

    return RuntimeAnomalyReport(
        elapsed_seconds=elapsed,
        reference_seconds=reference,
        ratio_to_reference=ratio,
        warning_ratio=warning_ratio,
        severe_ratio=severe_ratio,
        level=level,
        message=message,
    )
