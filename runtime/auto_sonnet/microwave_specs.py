from __future__ import annotations

import ast
import math
import re
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Any


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required field '{key}' in {context}")
    return mapping[key]


def _safe_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not normalized:
        raise ValueError(f"Unable to derive a safe identifier from '{name}'")
    if normalized[0].isdigit():
        normalized = f"v_{normalized}"
    return normalized


def _ensure_identifier(name: str, context: str) -> str:
    if not _IDENTIFIER.fullmatch(name):
        raise ValueError(f"{context} '{name}' is not a valid identifier")
    return name


def _ensure_object_list(raw_items: Any, context: str) -> list[dict[str, Any]]:
    if raw_items is None:
        return []
    if not isinstance(raw_items, list):
        raise ValueError(f"{context} must be an array of objects")
    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{context} item #{index} must be an object")
        items.append(dict(item))
    return items


def _collect_variable_table(raw_variables: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if raw_variables is None:
        return [], {}
    if isinstance(raw_variables, dict):
        variable_items: list[dict[str, Any]] = []
        default_context: dict[str, Any] = {}
        for key, raw_value in raw_variables.items():
            name = _ensure_identifier(str(key), "variable name")
            entry: dict[str, Any] = {"key": name, "name": name}
            if isinstance(raw_value, dict):
                if "value" not in raw_value:
                    raise ValueError(f"Variable '{name}' mapping must include 'value'")
                entry["value"] = raw_value["value"]
                for extra_key, extra_value in raw_value.items():
                    if extra_key != "value":
                        entry[extra_key] = extra_value
            else:
                entry["value"] = raw_value
            variable_items.append(entry)
            default_context[name] = entry["value"]
        return variable_items, default_context
    if isinstance(raw_variables, list):
        variable_items = []
        default_context = {}
        for item in raw_variables:
            if not isinstance(item, dict):
                raise ValueError("variables list items must be objects")
            name = _ensure_identifier(str(_require(item, "name", "variable")), "variable name")
            key = _safe_name(str(item.get("key", name)))
            entry = {"key": key, "name": name, "value": _require(item, "value", "variable")}
            variable_items.append(entry)
            default_context[name] = entry["value"]
        return variable_items, default_context
    raise ValueError("variables must be either a table/object or a list of variable objects")


def _eval_expr(value: Any, context: dict[str, float]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"Unsupported expression value: {value!r}")
    text = value.strip()
    if not text:
        raise ValueError("Expression must not be empty")

    node = ast.parse(text, mode="eval")

    def visit(expr: ast.AST) -> float:
        if isinstance(expr, ast.Expression):
            return visit(expr.body)
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
            return float(expr.value)
        if isinstance(expr, ast.Name):
            if expr.id not in context:
                raise ValueError(f"Unknown variable in expression: {expr.id}")
            return float(context[expr.id])
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            operand = visit(expr.operand)
            return operand if isinstance(expr.op, ast.UAdd) else -operand
        if isinstance(expr, ast.BinOp):
            left = visit(expr.left)
            right = visit(expr.right)
            if isinstance(expr.op, ast.Add):
                return left + right
            if isinstance(expr.op, ast.Sub):
                return left - right
            if isinstance(expr.op, ast.Mult):
                return left * right
            if isinstance(expr.op, ast.Div):
                return left / right
        raise ValueError(f"Unsupported expression syntax: {text}")

    return visit(node)


def _numeric_context(default_context: dict[str, Any]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    pending = dict(default_context)
    while pending:
        progressed = False
        for key, raw_value in list(pending.items()):
            try:
                resolved[key] = _eval_expr(raw_value, resolved)
            except ValueError:
                continue
            pending.pop(key)
            progressed = True
        if not progressed:
            unresolved = ", ".join(sorted(pending))
            raise ValueError(f"Unable to resolve numeric defaults for variables: {unresolved}")
    return resolved


def _format_variant_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(Decimal(str(value)).normalize(), "f").rstrip("0").rstrip(".")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug or "variant"


def _variant_key(variable: str, value: float) -> str:
    token = _format_variant_number(value).replace("-", "neg").replace(".", "p")
    return f"{_slugify(variable)}_{token}"


def _resolve_number(value: Any, context: dict[str, float]) -> float:
    return _eval_expr(value, context)


def _normalized_sweep_sets(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_sets = _ensure_object_list(data.get("sweep_sets"), "sweep_sets")
    if raw_sets:
        normalized: list[dict[str, Any]] = []
        for index, sweep_set in enumerate(raw_sets, start=1):
            item = dict(sweep_set)
            item.setdefault("key", f"sweepset_{index}")
            item["frequency_sweeps"] = _ensure_object_list(item.get("frequency_sweeps"), "frequency_sweeps")
            item["parameter_sweeps"] = _ensure_object_list(item.get("parameter_sweeps"), "parameter_sweeps")
            normalized.append(item)
        return normalized

    frequency_sweeps = _ensure_object_list(data.get("frequency_sweeps"), "frequency_sweeps")
    parameter_sweeps = _ensure_object_list(data.get("parameter_sweeps"), "parameter_sweeps")
    if frequency_sweeps or parameter_sweeps:
        return [
            {
                "key": "default_sweepset",
                "frequency_sweeps": frequency_sweeps,
                "parameter_sweeps": parameter_sweeps,
            }
        ]
    return []


def _frequency_only_sweep_sets(data: dict[str, Any]) -> list[dict[str, Any]]:
    frequency_sets: list[dict[str, Any]] = []
    for sweep_set in _normalized_sweep_sets(data):
        frequency_only = {
            "key": sweep_set.get("key"),
            "frequency_sweeps": list(sweep_set.get("frequency_sweeps", [])),
            "parameter_sweeps": [],
        }
        if frequency_only["frequency_sweeps"]:
            frequency_sets.append(frequency_only)
    return frequency_sets


def _expand_linear_values(start: float, stop: float, step: float) -> list[float]:
    if step == 0:
        raise ValueError("linear sweep step must not be zero")
    if stop > start and step < 0:
        raise ValueError("linear sweep step must be positive when stop > start")
    if stop < start and step > 0:
        raise ValueError("linear sweep step must be negative when stop < start")

    values: list[float] = []
    current = Decimal(str(start))
    target = Decimal(str(stop))
    delta = Decimal(str(step))
    guard = 0
    while True:
        values.append(float(current))
        if current == target:
            break
        current += delta
        guard += 1
        if guard > 100000:
            raise ValueError("linear sweep generated too many points")
        if delta > 0 and current > target:
            if values[-1] != stop:
                values.append(stop)
            break
        if delta < 0 and current < target:
            if values[-1] != stop:
                values.append(stop)
            break
    return values


def _expand_parameter_sweep_values(sweep: dict[str, Any], context: dict[str, float]) -> list[float]:
    sweep_type = str(sweep.get("type", sweep.get("sweep_type", "linear"))).lower()
    if sweep_type == "linear":
        return _expand_linear_values(
            _resolve_number(_require(sweep, "start", "parameter sweep"), context),
            _resolve_number(_require(sweep, "stop", "parameter sweep"), context),
            _resolve_number(_require(sweep, "step", "parameter sweep"), context),
        )
    if sweep_type == "list":
        values = sweep.get("values", [])
        if not values:
            raise ValueError("list parameter sweep requires at least one value")
        return [_resolve_number(value, context) for value in values]
    if sweep_type == "single_value":
        return [_resolve_number(_require(sweep, "value", "parameter sweep"), context)]
    if sweep_type == "exponential":
        start = _resolve_number(_require(sweep, "start", "parameter sweep"), context)
        stop = _resolve_number(_require(sweep, "stop", "parameter sweep"), context)
        num_points = int(_require(sweep, "num_points", "parameter sweep"))
        if num_points < 2:
            raise ValueError("exponential parameter sweep requires num_points >= 2")
        ratio = (stop / start) ** (1 / (num_points - 1))
        return [start * (ratio**index) for index in range(num_points)]
    if sweep_type in {"corner", "sensitivity"}:
        min_value = sweep.get("min_value", sweep.get("min"))
        max_value = sweep.get("max_value", sweep.get("max"))
        if min_value is None or max_value is None:
            raise ValueError(f"{sweep_type} parameter sweep requires min and max")
        return [_resolve_number(min_value, context), _resolve_number(max_value, context)]
    raise ValueError(f"Unsupported high-level parameter sweep type: {sweep_type}")


def _port_edge_for_side(side: str) -> int:
    lowered = side.lower()
    if lowered in {"west", "left"}:
        return 1
    if lowered in {"east", "right"}:
        return 3
    raise ValueError(f"Unsupported port side '{side}'. Only west/left and east/right are supported in v0.")


def compile_microwave_planar_spec(
    data: dict[str, Any],
    *,
    source_path: Path | None = None,
    variable_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema_version = str(data.get("schema_version", "")).strip()
    if schema_version != "microwave_planar_v0":
        raise ValueError(f"Unsupported microwave schema_version: {schema_version!r}")

    project = dict(_require(data, "project", "microwave planar spec"))
    units = dict(data.get("units", {}))
    stackup = dict(_require(data, "stackup", "microwave planar spec"))
    layout = dict(_require(data, "layout", "microwave planar spec"))
    segments = _ensure_object_list(_require(data, "segments", "microwave planar spec"), "segments")
    ports = _ensure_object_list(_require(data, "ports", "microwave planar spec"), "ports")
    sweep_sets = _frequency_only_sweep_sets(data)
    output_files = _ensure_object_list(data.get("output_files"), "output_files")

    layout_kind = str(_require(layout, "kind", "layout"))
    if layout_kind != "microstrip_chain":
        raise ValueError(f"Unsupported layout kind '{layout_kind}'. Only 'microstrip_chain' is supported in v0.")

    variable_items, default_context = _collect_variable_table(data.get("variables"))
    merged_context = dict(default_context)
    if variable_overrides:
        merged_context.update(variable_overrides)
    numeric_context = _numeric_context(merged_context)

    substrate_name = str(stackup.get("substrate_name", "Substrate"))
    metal_name = str(stackup.get("metal_name", "Copper"))
    trace_name = str(stackup.get("trace_name", "Trace"))
    air_thickness = _resolve_number(stackup.get("air_thickness", 10.0), numeric_context)
    substrate_thickness = _resolve_number(_require(stackup, "substrate_thickness", "stackup"), numeric_context)
    substrate_eps = _resolve_number(_require(stackup, "substrate_eps", "stackup"), numeric_context)
    substrate_tan = _resolve_number(stackup.get("substrate_tan", 0.0), numeric_context)
    metal_thickness = _resolve_number(_require(stackup, "metal_thickness", "stackup"), numeric_context)
    metal_model = str(stackup.get("metal_model", "Thick Metal"))
    metal_conductivity = _resolve_number(_require(stackup, "metal_conductivity", "stackup"), numeric_context)

    margin_x = _resolve_number(layout.get("box_margin_x", 1.0), numeric_context)
    margin_y = _resolve_number(layout.get("box_margin_y", 4.0), numeric_context)
    cells_per_unit = _resolve_number(layout.get("cells_per_length_unit", 10), numeric_context)
    start_x = _resolve_number(layout.get("start_x", 0.0), numeric_context)

    if not segments:
        raise ValueError("At least one segment is required")

    compiled_variables = [
        {
            "key": item["key"],
            "name": item["name"],
            "value": numeric_context[item["name"]],
        }
        for item in variable_items
    ]

    max_width = 0.0
    for segment in segments:
        width_expr = _require(segment, "width", f"segment {segment}")
        max_width = max(max_width, _eval_expr(width_expr, numeric_context))

    centerline_y_value = layout.get("centerline_y")
    if centerline_y_value is None:
        centerline_y_value = margin_y + max_width / 2.0
    centerline_y = _resolve_number(centerline_y_value, numeric_context)

    polygons: list[dict[str, Any]] = []
    port_segment_map: dict[str, str] = {}
    segment_ids: set[str] = set()
    current_x = start_x
    total_length = 0.0

    for index, segment in enumerate(segments, start=1):
        segment_id = _safe_name(str(segment.get("id", f"section_{index}")))
        if segment_id in segment_ids:
            raise ValueError(f"Duplicate segment id after normalization: '{segment_id}'")
        segment_ids.add(segment_id)
        length_expr = _require(segment, "length", f"segment {segment_id}")
        width_expr = _require(segment, "width", f"segment {segment_id}")
        length = _resolve_number(length_expr, numeric_context)
        width = _resolve_number(width_expr, numeric_context)
        total_length += length

        x0 = current_x
        x1 = x0 + length
        y_top = centerline_y + width / 2.0
        y_bot = centerline_y - width / 2.0

        points = [(x1, y_top), (x0, y_top), (x0, y_bot), (x1, y_bot)]
        polygons.append({"key": segment_id, "tech_layer": "trace", "points": points})
        port_segment_map[segment_id] = segment_id
        current_x = x1

    min_x = min(point[0] for polygon in polygons for point in polygon["points"])
    max_x = max(point[0] for polygon in polygons for point in polygon["points"])
    min_y = min(point[1] for polygon in polygons for point in polygon["points"])
    max_y = max(point[1] for polygon in polygons for point in polygon["points"])

    shift_x = max(margin_x - min_x, 0.0)
    shift_y = max(margin_y - min_y, 0.0)
    if shift_x or shift_y:
        shifted_polygons: list[dict[str, Any]] = []
        for polygon in polygons:
            shifted_polygons.append(
                {
                    **polygon,
                    "points": [(x + shift_x, y + shift_y) for x, y in polygon["points"]],
                }
            )
        polygons = shifted_polygons
        min_x += shift_x
        max_x += shift_x
        min_y += shift_y
        max_y += shift_y

    box_size_x = max_x + margin_x
    box_size_y = max_y + margin_y
    cells_x = max(1, math.ceil(box_size_x * cells_per_unit))
    cells_y = max(1, math.ceil(box_size_y * cells_per_unit))

    compiled_ports: list[dict[str, Any]] = []
    for port in ports:
        segment_ref = _safe_name(str(_require(port, "segment", "port")))
        if segment_ref not in port_segment_map:
            raise ValueError(f"Port references unknown segment '{segment_ref}'")
        compiled_ports.append(
            {
                "key": str(port.get("key", f"port_{_require(port, 'port_number', 'port')}")),
                "port_number": _require(port, "port_number", "port"),
                "polygon": port_segment_map[segment_ref],
                "edge": _port_edge_for_side(str(_require(port, "side", "port"))),
            }
        )

    low_level: dict[str, Any] = {
        "project": project,
        "units": {
            "length": str(units.get("length", "MM")).upper(),
            "roughness": str(units.get("roughness", "micron")),
            "sheetres": str(units.get("sheetres", "OHSQ")),
            "resistance": str(units.get("resistance", "OH")),
            "inductance": str(units.get("inductance", "NH")),
            "capacitance": str(units.get("capacitance", "PF")),
            "frequency": str(units.get("frequency", "GHZ")).upper(),
            "conductivity": str(units.get("conductivity", "SM")),
            "resistivity": str(units.get("resistivity", "OHCM")),
        },
        "box": {
            "size_x": box_size_x,
            "size_y": box_size_y,
            "cells_x": cells_x,
            "cells_y": cells_y,
        },
        "variables": compiled_variables,
        "dielectrics": [
            {"key": "substrate", "name": substrate_name, "eps": substrate_eps, "tan": substrate_tan}
        ],
        "dielectric_layers": [
            {"key": "air_layer", "number": 0, "thickness": air_thickness},
            {
                "key": "substrate_layer",
                "number": 1,
                "thickness": substrate_thickness,
                "material_name": substrate_name,
            },
        ],
        "conductors": [
            {"key": "copper", "name": metal_name, "conductivity": metal_conductivity}
        ],
        "planar_tech_layers": [
            {
                "key": "trace",
                "diel_layer": "air_layer",
                "thickness": metal_thickness,
                "model_type": metal_model,
                "material_name": metal_name,
                "name": trace_name,
            }
        ],
        "polygons": polygons,
        "ports": compiled_ports,
        "sweep_sets": sweep_sets,
        "output_files": output_files or [{"key": "touchstone", "path": "$BASENAME.s2p"}],
    }

    for optional_key in ("preamble", "postamble"):
        if optional_key in data:
            low_level[optional_key] = data[optional_key]

    return low_level


def expand_microwave_planar_variants(
    data: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    variable_items, default_context = _collect_variable_table(data.get("variables"))
    variable_names = {item["name"] for item in variable_items}
    numeric_context = _numeric_context(default_context)

    parameter_sweeps: list[dict[str, Any]] = []
    for sweep_set in _normalized_sweep_sets(data):
        parameter_sweeps.extend(list(sweep_set.get("parameter_sweeps", [])))

    if not parameter_sweeps:
        return [("default", compile_microwave_planar_spec(data, source_path=source_path))]

    sweep_dimensions: list[tuple[str, list[float]]] = []
    seen_variables: set[str] = set()
    for sweep in parameter_sweeps:
        variable_name = str(_require(sweep, "variable", "parameter sweep"))
        if variable_name not in variable_names:
            raise ValueError(f"High-level parameter sweep references unknown variable '{variable_name}'")
        if variable_name in seen_variables:
            raise ValueError(f"Duplicate high-level parameter sweep for variable '{variable_name}' is not supported")
        seen_variables.add(variable_name)
        sweep_dimensions.append((variable_name, _expand_parameter_sweep_values(sweep, numeric_context)))

    variants: list[tuple[str, dict[str, Any]]] = []
    for combination in product(*(values for _, values in sweep_dimensions)):
        overrides = {variable_name: value for (variable_name, _), value in zip(sweep_dimensions, combination)}
        label = "__".join(
            _variant_key(variable_name, value)
            for (variable_name, _), value in zip(sweep_dimensions, combination)
        )
        compiled = compile_microwave_planar_spec(
            data,
            source_path=source_path,
            variable_overrides=overrides,
        )
        preamble = list(compiled.get("preamble", []))
        preamble.append(
            "# sweep_variant " + ", ".join(
                f"{variable_name}={_format_variant_number(value)}"
                for (variable_name, _), value in zip(sweep_dimensions, combination)
            )
        )
        compiled["preamble"] = preamble
        variants.append((label, compiled))
    return variants
