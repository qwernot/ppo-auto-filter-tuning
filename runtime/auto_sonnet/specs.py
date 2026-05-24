from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None

from .microwave_specs import compile_microwave_planar_spec, expand_microwave_planar_variants


NumberLike = str | int | float
PointList = str | list[tuple[float, float]]


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required field '{key}' in {context}")
    return mapping[key]


def _optional_key(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _normalize_points(raw_points: Any) -> PointList:
    if isinstance(raw_points, str):
        return raw_points
    if not isinstance(raw_points, list):
        raise ValueError("points must be a semicolon string or a list of [x, y] pairs")
    return [tuple(point) for point in raw_points]


def _resolve_input_path(path: str | None, *, base_dir: Path | None = None) -> str | None:
    if path is None:
        return None
    expanded = Path(os.path.expandvars(path)).expanduser()
    if expanded.is_absolute() or base_dir is None:
        return expanded.as_posix()
    return (base_dir / expanded).resolve().as_posix()


@dataclass(slots=True)
class UnitsSpec:
    length: str = "MM"
    roughness: str = "micron"
    sheetres: str = "OHSQ"
    resistance: str = "OH"
    inductance: str = "NH"
    capacitance: str = "PF"
    frequency: str = "GHZ"
    conductivity: str = "SM"
    resistivity: str = "OHCM"


@dataclass(slots=True)
class BoxSpec:
    size_x: NumberLike
    size_y: NumberLike
    cells_x: int
    cells_y: int
    local_origin: tuple[NumberLike, NumberLike] | None = None


@dataclass(slots=True)
class VariableSpec:
    key: str
    name: str
    value: NumberLike


@dataclass(slots=True)
class DielectricSpec:
    key: str
    name: str
    eps: NumberLike
    tan: NumberLike = 0.0


@dataclass(slots=True)
class DielectricLayerSpec:
    key: str
    number: int
    thickness: NumberLike
    material_name: str | None = None


@dataclass(slots=True)
class ConductorSpec:
    key: str
    name: str
    conductivity: NumberLike


@dataclass(slots=True)
class PlanarTechLayerSpec:
    key: str
    diel_layer: str
    thickness: NumberLike
    model_type: str
    material_name: str
    name: str


@dataclass(slots=True)
class PolygonSpec:
    key: str
    tech_layer: str
    points: PointList


@dataclass(slots=True)
class PortSpec:
    key: str
    port_number: int
    polygon: str
    edge: int


@dataclass(slots=True)
class FrequencySweepSpec:
    key: str
    sweep_type: str
    start: NumberLike | None = None
    stop: NumberLike | None = None
    step: NumberLike | None = None
    num_points: int | None = None
    freqs: list[NumberLike] | None = None
    freq: NumberLike | None = None


@dataclass(slots=True)
class ParameterSweepSpec:
    key: str
    variable: str
    sweep_type: str
    start: NumberLike | None = None
    stop: NumberLike | None = None
    step: NumberLike | None = None
    num_points: int | None = None
    values: list[NumberLike] | None = None
    value: NumberLike | None = None
    min_value: NumberLike | None = None
    max_value: NumberLike | None = None


@dataclass(slots=True)
class SweepSetSpec:
    key: str
    frequency_sweeps: list[FrequencySweepSpec] = field(default_factory=list)
    parameter_sweeps: list[ParameterSweepSpec] = field(default_factory=list)


@dataclass(slots=True)
class OutputFileSpec:
    key: str
    path: str
    param: str = "S"
    file_format: str = "TOUCH"
    format: str = "MA"


@dataclass(slots=True)
class ProjectSettings:
    save_path: str
    template: str | None = None
    verify_before_analyze: bool = True
    analyze: bool = True
    clean_data_before_analyze: bool = False
    server: str | None = None
    launch_monitor: bool = False
    queries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AutomationSpec:
    project: ProjectSettings
    units: UnitsSpec | None = None
    box: BoxSpec | None = None
    variables: list[VariableSpec] = field(default_factory=list)
    dielectrics: list[DielectricSpec] = field(default_factory=list)
    dielectric_layers: list[DielectricLayerSpec] = field(default_factory=list)
    conductors: list[ConductorSpec] = field(default_factory=list)
    planar_tech_layers: list[PlanarTechLayerSpec] = field(default_factory=list)
    polygons: list[PolygonSpec] = field(default_factory=list)
    ports: list[PortSpec] = field(default_factory=list)
    sweep_sets: list[SweepSetSpec] = field(default_factory=list)
    output_files: list[OutputFileSpec] = field(default_factory=list)
    preamble: list[str] = field(default_factory=list)
    postamble: list[str] = field(default_factory=list)
    source_path: Path | None = None


@dataclass(slots=True)
class AutomationJob:
    key: str
    spec: AutomationSpec


def _parse_units(raw: dict[str, Any] | None) -> UnitsSpec | None:
    if raw is None:
        return None
    return UnitsSpec(
        length=raw.get("length", "MM"),
        roughness=raw.get("roughness", "micron"),
        sheetres=raw.get("sheetres", "OHSQ"),
        resistance=raw.get("resistance", "OH"),
        inductance=raw.get("inductance", "NH"),
        capacitance=raw.get("capacitance", "PF"),
        frequency=raw.get("frequency", "GHZ"),
        conductivity=raw.get("conductivity", "SM"),
        resistivity=raw.get("resistivity", "OHCM"),
    )


def _parse_box(raw: dict[str, Any] | None) -> BoxSpec | None:
    if raw is None:
        return None
    local_origin = raw.get("local_origin")
    return BoxSpec(
        size_x=_require(raw, "size_x", "box"),
        size_y=_require(raw, "size_y", "box"),
        cells_x=_require(raw, "cells_x", "box"),
        cells_y=_require(raw, "cells_y", "box"),
        local_origin=tuple(local_origin) if local_origin is not None else None,
    )


def _parse_variables(raw_items: list[dict[str, Any]]) -> list[VariableSpec]:
    return [
        VariableSpec(
            key=_optional_key(item, "key", "id") or _require(item, "name", "variable"),
            name=_require(item, "name", "variable"),
            value=_require(item, "value", "variable"),
        )
        for item in raw_items
    ]


def _parse_dielectrics(raw_items: list[dict[str, Any]]) -> list[DielectricSpec]:
    return [
        DielectricSpec(
            key=_optional_key(item, "key", "id") or _require(item, "name", "dielectric"),
            name=_require(item, "name", "dielectric"),
            eps=_require(item, "eps", "dielectric"),
            tan=item.get("tan", 0.0),
        )
        for item in raw_items
    ]


def _parse_dielectric_layers(raw_items: list[dict[str, Any]]) -> list[DielectricLayerSpec]:
    return [
        DielectricLayerSpec(
            key=_optional_key(item, "key", "id") or f"layer_{_require(item, 'number', 'dielectric layer')}",
            number=_require(item, "number", "dielectric layer"),
            thickness=_require(item, "thickness", "dielectric layer"),
            material_name=_optional_key(item, "material_name", "material"),
        )
        for item in raw_items
    ]


def _parse_conductors(raw_items: list[dict[str, Any]]) -> list[ConductorSpec]:
    return [
        ConductorSpec(
            key=_optional_key(item, "key", "id") or _require(item, "name", "conductor"),
            name=_require(item, "name", "conductor"),
            conductivity=_require(item, "conductivity", "conductor"),
        )
        for item in raw_items
    ]


def _parse_planar_tech_layers(raw_items: list[dict[str, Any]]) -> list[PlanarTechLayerSpec]:
    layers: list[PlanarTechLayerSpec] = []
    for item in raw_items:
        material_name = _optional_key(item, "material_name", "material")
        if material_name is None:
            raise ValueError("planar tech layer requires 'material_name' or 'material'")
        layers.append(
            PlanarTechLayerSpec(
                key=_optional_key(item, "key", "id") or _require(item, "name", "planar tech layer"),
                diel_layer=_require(item, "diel_layer", "planar tech layer"),
                thickness=_require(item, "thickness", "planar tech layer"),
                model_type=_optional_key(item, "model_type", "model") or "Thin Metal",
                material_name=material_name,
                name=_require(item, "name", "planar tech layer"),
            )
        )
    return layers


def _parse_polygons(raw_items: list[dict[str, Any]]) -> list[PolygonSpec]:
    return [
        PolygonSpec(
            key=_optional_key(item, "key", "id") or f"polygon_{index}",
            tech_layer=_require(item, "tech_layer", "polygon"),
            points=_normalize_points(_require(item, "points", "polygon")),
        )
        for index, item in enumerate(raw_items, start=1)
    ]


def _parse_ports(raw_items: list[dict[str, Any]]) -> list[PortSpec]:
    ports: list[PortSpec] = []
    for index, item in enumerate(raw_items, start=1):
        polygon = _optional_key(item, "polygon", "poly")
        if polygon is None:
            raise ValueError("port requires 'polygon' or 'poly'")
        ports.append(
            PortSpec(
                key=_optional_key(item, "key", "id") or f"port_{index}",
                port_number=_require(item, "port_number", "port"),
                polygon=polygon,
                edge=_require(item, "edge", "port"),
            )
        )
    return ports


def _parse_frequency_sweeps(raw_items: list[dict[str, Any]]) -> list[FrequencySweepSpec]:
    sweeps: list[FrequencySweepSpec] = []
    for index, item in enumerate(raw_items, start=1):
        sweeps.append(
            FrequencySweepSpec(
                key=_optional_key(item, "key", "id") or f"freq_sweep_{index}",
                sweep_type=_optional_key(item, "sweep_type", "type") or "adaptive",
                start=item.get("start"),
                stop=item.get("stop"),
                step=item.get("step"),
                num_points=item.get("num_points"),
                freqs=item.get("freqs"),
                freq=item.get("freq"),
            )
        )
    return sweeps


def _parse_parameter_sweeps(raw_items: list[dict[str, Any]]) -> list[ParameterSweepSpec]:
    sweeps: list[ParameterSweepSpec] = []
    for index, item in enumerate(raw_items, start=1):
        sweeps.append(
            ParameterSweepSpec(
                key=_optional_key(item, "key", "id") or f"param_sweep_{index}",
                variable=_require(item, "variable", "parameter sweep"),
                sweep_type=_optional_key(item, "sweep_type", "type") or "linear",
                start=item.get("start"),
                stop=item.get("stop"),
                step=item.get("step"),
                num_points=item.get("num_points"),
                values=item.get("values"),
                value=item.get("value"),
                min_value=_optional_key(item, "min_value", "min"),
                max_value=_optional_key(item, "max_value", "max"),
            )
        )
    return sweeps


def _parse_sweep_sets(raw: dict[str, Any]) -> list[SweepSetSpec]:
    raw_sets = raw.get("sweep_sets", [])
    if not raw_sets:
        frequency = raw.get("frequency_sweeps", [])
        parameter = raw.get("parameter_sweeps", [])
        if frequency or parameter:
            raw_sets = [{"key": "default_sweepset", "frequency_sweeps": frequency, "parameter_sweeps": parameter}]

    sweep_sets: list[SweepSetSpec] = []
    for index, item in enumerate(raw_sets, start=1):
        sweep_sets.append(
            SweepSetSpec(
                key=_optional_key(item, "key", "id") or f"sweepset_{index}",
                frequency_sweeps=_parse_frequency_sweeps(item.get("frequency_sweeps", [])),
                parameter_sweeps=_parse_parameter_sweeps(item.get("parameter_sweeps", [])),
            )
        )
    return sweep_sets


def _parse_output_files(raw_items: list[dict[str, Any]]) -> list[OutputFileSpec]:
    return [
        OutputFileSpec(
            key=_optional_key(item, "key", "id") or f"output_{index}",
            path=_require(item, "path", "output file"),
            param=item.get("param", "S"),
            file_format=_optional_key(item, "file_format", "type") or "TOUCH",
            format=item.get("format", "MA"),
        )
        for index, item in enumerate(raw_items, start=1)
    ]


def _automation_spec_from_standard_mapping(data: dict[str, Any], *, source_path: Path | None = None) -> AutomationSpec:
    project_raw = _require(data, "project", "root")
    base_dir = source_path.parent if source_path is not None else None
    project = ProjectSettings(
        save_path=_require(project_raw, "save_path", "project"),
        template=_resolve_input_path(project_raw.get("template"), base_dir=base_dir),
        verify_before_analyze=project_raw.get("verify_before_analyze", True),
        analyze=project_raw.get("analyze", True),
        clean_data_before_analyze=project_raw.get("clean_data_before_analyze", False),
        server=project_raw.get("server"),
        launch_monitor=project_raw.get("launch_monitor", False),
        queries=list(project_raw.get("queries", [])),
    )
    return AutomationSpec(
        project=project,
        units=_parse_units(data.get("units")),
        box=_parse_box(data.get("box")),
        variables=_parse_variables(data.get("variables", [])),
        dielectrics=_parse_dielectrics(data.get("dielectrics", [])),
        dielectric_layers=_parse_dielectric_layers(_optional_key(data, "dielectric_layers", "layers") or []),
        conductors=_parse_conductors(data.get("conductors", [])),
        planar_tech_layers=_parse_planar_tech_layers(_optional_key(data, "planar_tech_layers", "tech_layers") or []),
        polygons=_parse_polygons(data.get("polygons", [])),
        ports=_parse_ports(data.get("ports", [])),
        sweep_sets=_parse_sweep_sets(data),
        output_files=_parse_output_files(_optional_key(data, "output_files", "outputs") or []),
        preamble=list(data.get("preamble", [])),
        postamble=list(data.get("postamble", [])),
        source_path=source_path,
    )


def automation_spec_from_mapping(data: dict[str, Any], *, source_path: Path | None = None) -> AutomationSpec:
    schema_version = data.get("schema_version")
    if schema_version == "microwave_planar_v0":
        compiled = compile_microwave_planar_spec(data, source_path=source_path)
        return _automation_spec_from_standard_mapping(compiled, source_path=source_path)
    return _automation_spec_from_standard_mapping(data, source_path=source_path)


def automation_jobs_from_mapping(data: dict[str, Any], *, source_path: Path | None = None) -> list[AutomationJob]:
    schema_version = data.get("schema_version")
    if schema_version == "microwave_planar_v0":
        return [
            AutomationJob(
                key=job_key,
                spec=_automation_spec_from_standard_mapping(compiled, source_path=source_path),
            )
            for job_key, compiled in expand_microwave_planar_variants(data, source_path=source_path)
        ]
    return [AutomationJob(key="default", spec=_automation_spec_from_standard_mapping(data, source_path=source_path))]


def _load_mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    spec_path = Path(path).resolve()
    suffix = spec_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(spec_path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML support is unavailable in this interpreter")
        data = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Unsupported spec file type. Use .json or .toml")
    if not isinstance(data, dict):
        raise ValueError("Specification file must contain a top-level object")
    return spec_path, data


def load_automation_spec(path: str | Path) -> AutomationSpec:
    spec_path, data = _load_mapping(path)
    return automation_spec_from_mapping(data, source_path=spec_path)


def load_automation_jobs(path: str | Path) -> list[AutomationJob]:
    spec_path, data = _load_mapping(path)
    return automation_jobs_from_mapping(data, source_path=spec_path)
