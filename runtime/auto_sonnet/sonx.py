from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import xml.etree.ElementTree as ET


_PROJECT_NS = "https://www.sonnetsoftware.com/schema/project"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_NS = {"son": _PROJECT_NS}

ET.register_namespace("", _PROJECT_NS)
ET.register_namespace("xsi", _XSI_NS)


@dataclass(slots=True)
class ProjectGrid:
    size_x: float
    size_y: float
    cells_x: int
    cells_y: int

    @property
    def cell_x(self) -> float:
        return self.size_x / self.cells_x

    @property
    def cell_y(self) -> float:
        return self.size_y / self.cells_y

    def to_dict(self) -> dict[str, float | int]:
        return {
            "size_x": self.size_x,
            "size_y": self.size_y,
            "cells_x": self.cells_x,
            "cells_y": self.cells_y,
            "cell_x": self.cell_x,
            "cell_y": self.cell_y,
        }


@dataclass(slots=True)
class FixedRefPlaneSpec:
    side: str
    ref_length: float


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer():
        return str(int(value))
    text = format(Decimal(repr(value)), "f").rstrip("0").rstrip(".")
    return text or "0"


def read_project_variables(path: str | Path) -> dict[str, str]:
    project_path = Path(path).resolve()
    tree = ET.parse(project_path)
    root = tree.getroot()
    variables: dict[str, str] = {}
    for variable in root.findall(".//son:Geometry/son:Variable", _NS):
        name = variable.get("Name")
        value = variable.get("Value")
        if name is not None and value is not None:
            variables[name] = value
    return variables


def read_project_grid(path: str | Path) -> ProjectGrid:
    project_path = Path(path).resolve()
    tree = ET.parse(project_path)
    root = tree.getroot()

    size = root.find(".//son:Geometry/son:Box/son:Size", _NS)
    num_cells = root.find(".//son:Geometry/son:Box/son:NumCells", _NS)
    if size is None or num_cells is None:
        raise ValueError("Sonnet project does not contain Box/Size and Box/NumCells information")

    size_x = size.get("X")
    size_y = size.get("Y")
    cells_x = num_cells.get("X")
    cells_y = num_cells.get("Y")
    if size_x is None or size_y is None or cells_x is None or cells_y is None:
        raise ValueError("Incomplete Box/Size or Box/NumCells information in Sonnet project")

    return ProjectGrid(
        size_x=float(size_x),
        size_y=float(size_y),
        cells_x=int(cells_x),
        cells_y=int(cells_y),
    )


def _polygon_signature(polygon: ET.Element) -> list[int]:
    signature: list[int] = []
    for points in polygon.findall("son:Points", _NS):
        text = (points.text or "").strip()
        signature.append(text.count("("))
    return signature


def write_project_with_variables(
    template_path: str | Path,
    output_path: str | Path,
    variable_updates: dict[str, float | int | str],
    *,
    disable_variable_sweeps: bool = True,
    clear_output_files: bool = False,
) -> Path:
    source_path = Path(template_path).resolve()
    target_path = Path(output_path).resolve()
    tree = ET.parse(source_path)
    root = tree.getroot()

    variables = {
        variable.get("Name"): variable
        for variable in root.findall(".//son:Geometry/son:Variable", _NS)
        if variable.get("Name") is not None
    }
    missing = [name for name in variable_updates if name not in variables]
    if missing:
        raise ValueError(f"Variables not found in Sonnet project template: {', '.join(sorted(missing))}")

    for name, value in variable_updates.items():
        variables[name].set("Value", str(value) if isinstance(value, str) else _format_number(float(value)))

    if disable_variable_sweeps:
        control = root.find(".//son:Control", _NS)
        if control is not None:
            sweep_type = control.find("son:SweepType", _NS)
            if sweep_type is not None:
                sweep_type.text = "ABS_ENTRY"

        sweeps = root.find(".//son:Sweeps", _NS)
        if sweeps is not None:
            sweeps.set("SweepVariables", "FALSE")
            for swept_var in sweeps.findall(".//son:SweptVar", _NS):
                swept_var.set("On", "FALSE")

    if clear_output_files:
        output_files = root.find(".//son:OutputFiles", _NS)
        if output_files is not None:
            root.remove(output_files)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="    ")
    tree.write(target_path, encoding="utf-8", xml_declaration=True)
    return target_path


def write_project_with_swept_ranges(
    template_path: str | Path,
    output_path: str | Path,
    sweep_updates: dict[str, dict[str, float | int | bool]],
    *,
    sweep_variables: bool | None = None,
) -> Path:
    source_path = Path(template_path).resolve()
    target_path = Path(output_path).resolve()
    tree = ET.parse(source_path)
    root = tree.getroot()

    sweep_nodes = {
        swept_var.get("Name"): swept_var
        for swept_var in root.findall(".//son:Sweeps//son:SweptVar", _NS)
        if swept_var.get("Name") is not None
    }
    missing = [name for name in sweep_updates if name not in sweep_nodes]
    if missing:
        raise ValueError(f"Swept variables not found in Sonnet project template: {', '.join(sorted(missing))}")

    for name, update in sweep_updates.items():
        swept_var = sweep_nodes[name]
        if "start" in update:
            swept_var.set("Start", _format_number(float(update["start"])))
        if "stop" in update:
            swept_var.set("Stop", _format_number(float(update["stop"])))
        if "step" in update:
            swept_var.set("Step", _format_number(float(update["step"])))
        if "on" in update:
            swept_var.set("On", "TRUE" if bool(update["on"]) else "FALSE")

    if sweep_variables is not None:
        sweeps = root.find(".//son:Sweeps", _NS)
        if sweeps is not None:
            sweeps.set("SweepVariables", "TRUE" if sweep_variables else "FALSE")

        control = root.find(".//son:Control", _NS)
        if control is not None:
            sweep_type = control.find("son:SweepType", _NS)
            if sweep_type is not None:
                sweep_type.text = "VARSWP" if sweep_variables else "ABS_ENTRY"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="    ")
    tree.write(target_path, encoding="utf-8", xml_declaration=True)
    return target_path


def write_project_with_box_setup(
    template_path: str | Path,
    output_path: str | Path,
    *,
    top_cover_material: str | None = None,
    bottom_cover_material: str | None = None,
    fixed_ref_planes: list[FixedRefPlaneSpec] | None = None,
    clear_ref_planes: bool = True,
) -> Path:
    source_path = Path(template_path).resolve()
    target_path = Path(output_path).resolve()
    tree = ET.parse(source_path)
    root = tree.getroot()

    box = root.find(".//son:Geometry/son:Box", _NS)
    if box is None:
        raise ValueError("Sonnet project does not contain a Geometry/Box section")

    cover_updates = (
        ("TOP", top_cover_material),
        ("BOTTOM", bottom_cover_material),
    )
    for cover_type, material in cover_updates:
        if material is None:
            continue
        cover = box.find(f"son:BoxCover[@Type='{cover_type}']", _NS)
        if cover is None:
            raise ValueError(f"Sonnet project does not contain a {cover_type} BoxCover")
        cover.set("MaterialType", material)

    if clear_ref_planes:
        for ref_plane in list(box.findall("son:RefPlane", _NS)):
            box.remove(ref_plane)

    if fixed_ref_planes:
        for spec in fixed_ref_planes:
            side = spec.side.strip().upper()
            if side not in {"LEFT", "RIGHT", "TOP", "BOTTOM"}:
                raise ValueError(f"Unsupported RefPlane side: {spec.side}")
            ET.SubElement(
                box,
                f"{{{_PROJECT_NS}}}RefPlane",
                attrib={
                    "Type": "FIXED",
                    "Side": side,
                    "RefLength": _format_number(float(spec.ref_length)),
                },
            )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="    ")
    tree.write(target_path, encoding="utf-8", xml_declaration=True)
    return target_path


def clone_project_parameterization(
    source_template_path: str | Path,
    target_project_path: str | Path,
    output_path: str | Path,
    *,
    copy_sweeps: bool = True,
) -> Path:
    source_path = Path(source_template_path).resolve()
    target_path = Path(target_project_path).resolve()
    output_path = Path(output_path).resolve()

    source_tree = ET.parse(source_path)
    target_tree = ET.parse(target_path)
    source_root = source_tree.getroot()
    target_root = target_tree.getroot()

    source_geometry = source_root.find(".//son:Geometry", _NS)
    target_geometry = target_root.find(".//son:Geometry", _NS)
    if source_geometry is None or target_geometry is None:
        raise ValueError("Both source and target Sonnet projects must contain a Geometry section")

    source_polygons = {
        polygon.get("MacroID"): polygon
        for polygon in source_root.findall(".//son:PlanarPolygon", _NS)
        if polygon.get("MacroID") is not None
    }
    target_polygons = {
        polygon.get("MacroID"): polygon
        for polygon in target_root.findall(".//son:PlanarPolygon", _NS)
        if polygon.get("MacroID") is not None
    }
    missing_polygons = [macro_id for macro_id in source_polygons if macro_id not in target_polygons]
    if missing_polygons:
        raise ValueError(
            "Target Sonnet project is missing polygons required for parameterization: "
            + ", ".join(sorted(missing_polygons))
        )

    for macro_id, source_polygon in source_polygons.items():
        target_polygon = target_polygons[macro_id]
        source_id = source_polygon.get("Id")
        if source_id is not None:
            target_polygon.set("Id", source_id)
        source_signature = _polygon_signature(source_polygon)
        target_signature = _polygon_signature(target_polygon)
        if source_signature != target_signature:
            raise ValueError(
                f"Polygon '{macro_id}' does not match between source and target projects: "
                f"{source_signature} != {target_signature}"
            )

    for existing_variable in list(target_geometry.findall("son:Variable", _NS)):
        target_geometry.remove(existing_variable)
    for source_variable in source_geometry.findall("son:Variable", _NS):
        target_geometry.insert(0, copy.deepcopy(source_variable))

    if copy_sweeps:
        source_control = source_root.find(".//son:Control", _NS)
        target_control = target_root.find(".//son:Control", _NS)
        if source_control is not None and target_control is not None:
            source_sweep_type = source_control.find("son:SweepType", _NS)
            target_sweep_type = target_control.find("son:SweepType", _NS)
            if source_sweep_type is not None and target_sweep_type is not None:
                target_sweep_type.text = source_sweep_type.text

        source_sweeps = source_root.find(".//son:Sweeps", _NS)
        target_sweeps = target_root.find(".//son:Sweeps", _NS)
        if source_sweeps is not None and target_sweeps is not None:
            if source_sweeps.get("SweepVariables") is not None:
                target_sweeps.set("SweepVariables", source_sweeps.get("SweepVariables", "FALSE"))

            source_sets = {
                sweep_set.get("MacroID"): sweep_set
                for sweep_set in source_sweeps.findall("son:Set", _NS)
                if sweep_set.get("MacroID") is not None
            }
            target_sets = {
                sweep_set.get("MacroID"): sweep_set
                for sweep_set in target_sweeps.findall("son:Set", _NS)
                if sweep_set.get("MacroID") is not None
            }
            for macro_id, source_set in source_sets.items():
                target_set = target_sets.get(macro_id)
                if target_set is None:
                    continue
                for existing_variables in list(target_set.findall("son:Variables", _NS)):
                    target_set.remove(existing_variables)
                source_variables = source_set.find("son:Variables", _NS)
                if source_variables is not None:
                    target_set.append(copy.deepcopy(source_variables))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(target_tree, space="    ")
    target_tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
