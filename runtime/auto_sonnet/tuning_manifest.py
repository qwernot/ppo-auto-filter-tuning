from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .filter_metrics import FilterMetricConfig


def _resolve_optional_path(raw: str | None, *, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base_dir / expanded).resolve()


@dataclass(slots=True)
class TuningVariableSpec:
    name: str
    sonnet_names: list[str]
    unit: str = "MM"
    min_value: float | None = None
    max_value: float | None = None
    default_value: float | None = None
    max_delta: float | None = None
    snap_step: float | None = None
    snap_origin: float | None = None
    description: str | None = None
    shared_group: str | None = None

    def normalize(self, value: float) -> float:
        if self.snap_step is None or self.snap_step <= 0.0:
            return value
        origin = self.snap_origin
        if origin is None:
            origin = self.default_value if self.default_value is not None else 0.0
        snapped = float(origin) + round((value - float(origin)) / self.snap_step) * self.snap_step
        if self.min_value is not None:
            snapped = max(snapped, self.min_value)
        if self.max_value is not None:
            snapped = min(snapped, self.max_value)
        return snapped

    def validate(self, value: float) -> None:
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"Variable '{self.name}' value {value} is below min_value {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"Variable '{self.name}' value {value} is above max_value {self.max_value}")


@dataclass(slots=True)
class TuningManifest:
    name: str
    template_path: Path
    save_path: str
    server: str | None = None
    target_path: Path | None = None
    analysis: FilterMetricConfig = field(default_factory=FilterMetricConfig)
    variables: list[TuningVariableSpec] = field(default_factory=list)
    source_path: Path | None = None

    @property
    def variable_map(self) -> dict[str, TuningVariableSpec]:
        return {variable.name: variable for variable in self.variables}

    def resolve_values(self, overrides: dict[str, float] | None = None) -> dict[str, float]:
        values: dict[str, float] = {}
        overrides = overrides or {}
        for variable in self.variables:
            if variable.name in overrides:
                value = float(overrides[variable.name])
            elif variable.default_value is not None:
                value = float(variable.default_value)
            else:
                raise ValueError(f"Missing required tuning variable value: {variable.name}")
            value = variable.normalize(value)
            variable.validate(value)
            values[variable.name] = value
        extra = sorted(set(overrides) - {variable.name for variable in self.variables})
        if extra:
            raise ValueError(f"Unknown tuning variable(s): {', '.join(extra)}")
        return values

    def to_template_updates(self, overrides: dict[str, float] | None = None) -> dict[str, float]:
        resolved = self.resolve_values(overrides)
        updates: dict[str, float] = {}
        for variable in self.variables:
            value = resolved[variable.name]
            for sonnet_name in variable.sonnet_names:
                updates[sonnet_name] = value
        return updates

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "template_path": str(self.template_path),
            "save_path": self.save_path,
            "server": self.server,
            "target_path": None if self.target_path is None else str(self.target_path),
            "analysis": {
                "peak_search_start_hz": self.analysis.peak_search_start_hz,
                "peak_search_stop_hz": self.analysis.peak_search_stop_hz,
                "bandwidth_drop_db": self.analysis.bandwidth_drop_db,
                "high_side_search_start_hz": self.analysis.high_side_search_start_hz,
                "high_side_search_stop_hz": self.analysis.high_side_search_stop_hz,
            },
            "variables": [
                {
                    "name": variable.name,
                    "sonnet_names": variable.sonnet_names,
                    "unit": variable.unit,
                    "min_value": variable.min_value,
                    "max_value": variable.max_value,
                    "default_value": variable.default_value,
                    "max_delta": variable.max_delta,
                    "snap_step": variable.snap_step,
                    "snap_origin": variable.snap_origin,
                    "description": variable.description,
                    "shared_group": variable.shared_group,
                }
                for variable in self.variables
            ],
        }


def load_tuning_manifest(path: str | Path) -> TuningManifest:
    manifest_path = Path(path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Tuning manifest must contain a top-level JSON object")
    base_dir = manifest_path.parent
    raw_analysis = data.get("analysis") or {}

    variables: list[TuningVariableSpec] = []
    for raw_variable in data.get("variables", []):
        if not isinstance(raw_variable, dict):
            raise ValueError("Tuning manifest variables must be objects")
        sonnet_names = raw_variable.get("sonnet_names")
        if sonnet_names is None:
            single_name = raw_variable.get("sonnet_name")
            if single_name is None:
                raise ValueError("Each tuning variable requires 'sonnet_names' or 'sonnet_name'")
            sonnet_names = [single_name]
        variables.append(
            TuningVariableSpec(
                name=str(raw_variable["name"]),
                sonnet_names=[str(name) for name in sonnet_names],
                unit=str(raw_variable.get("unit", "MM")),
                min_value=None if raw_variable.get("min_value") is None else float(raw_variable["min_value"]),
                max_value=None if raw_variable.get("max_value") is None else float(raw_variable["max_value"]),
                default_value=None if raw_variable.get("default_value") is None else float(raw_variable["default_value"]),
                max_delta=None if raw_variable.get("max_delta") is None else float(raw_variable["max_delta"]),
                snap_step=None if raw_variable.get("snap_step") is None else float(raw_variable["snap_step"]),
                snap_origin=None if raw_variable.get("snap_origin") is None else float(raw_variable["snap_origin"]),
                description=None if raw_variable.get("description") is None else str(raw_variable["description"]),
                shared_group=None if raw_variable.get("shared_group") is None else str(raw_variable["shared_group"]),
            )
        )

    template_path = _resolve_optional_path(str(data["template_path"]), base_dir=base_dir)
    if template_path is None:
        raise ValueError("Tuning manifest requires template_path")

    return TuningManifest(
        name=str(data.get("name", manifest_path.stem)),
        template_path=template_path,
        save_path=str(data.get("save_path", "SonnetProject/tuning_eval.sonx")),
        server=None if data.get("server") is None else str(data.get("server")),
        target_path=_resolve_optional_path(data.get("target_path"), base_dir=base_dir),
        analysis=FilterMetricConfig(
            peak_search_start_hz=None
            if raw_analysis.get("peak_search_start_hz") is None
            else float(raw_analysis["peak_search_start_hz"]),
            peak_search_stop_hz=None
            if raw_analysis.get("peak_search_stop_hz") is None
            else float(raw_analysis["peak_search_stop_hz"]),
            bandwidth_drop_db=float(raw_analysis.get("bandwidth_drop_db", 3.0)),
            high_side_search_start_hz=None
            if raw_analysis.get("high_side_search_start_hz") is None
            else float(raw_analysis["high_side_search_start_hz"]),
            high_side_search_stop_hz=None
            if raw_analysis.get("high_side_search_stop_hz") is None
            else float(raw_analysis["high_side_search_stop_hz"]),
        ),
        variables=variables,
        source_path=manifest_path,
    )
