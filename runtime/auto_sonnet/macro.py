from __future__ import annotations

from decimal import Decimal
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _base26(index: int) -> str:
    if index < 1:
        raise ValueError("Macro id index must be >= 1")
    result = ""
    value = index
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer():
        return str(int(value))
    text = format(Decimal(repr(value)), "f").rstrip("0").rstrip(".")
    return text or "0"


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "/").replace('"', '\\"') + '"'


@dataclass(frozen=True)
class RawValue:
    value: str


def _format_value(value: object) -> str:
    if isinstance(value, RawValue):
        return value.value
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, Path):
        return _quote(value.as_posix())
    if isinstance(value, tuple):
        return ",".join(_format_value(item) for item in value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_.$:+/\-]+", text):
        return text
    return _quote(text)


def _format_path(path: str | Path) -> str:
    if isinstance(path, Path):
        return _quote(path.as_posix())
    return _quote(str(path).replace("\\", "/"))


def _format_points(points: str | Iterable[tuple[float, float]]) -> RawValue:
    if isinstance(points, str):
        return RawValue(points)
    rendered = ";".join(f"{_format_number(x)},{_format_number(y)}" for x, y in points)
    return RawValue(rendered)


class MacroBuilder:
    _PREFIXES = {
        "project": "project",
        "dielectric": "dielectric",
        "diel_layer": "diel_layer",
        "conductor": "conductor",
        "planar_tech_layer": "planarTL",
        "polygon": "polygon",
        "port": "port",
        "variable": "variable",
        "sweepset": "sweepset",
        "freq_sweep": "freq_sweep",
        "param_sweep": "param_sweep",
        "output_file": "output_file",
    }

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._counters = {kind: 0 for kind in self._PREFIXES}
        self._aliases: dict[str, str] = {}
        self.active_project: str | None = None

    def _next_id(self, kind: str) -> str:
        self._counters[kind] += 1
        return f"{self._PREFIXES[kind]}{_base26(self._counters[kind])}"

    def _remember(self, alias: str | None, macro_id: str) -> str:
        if alias:
            self._aliases[alias] = macro_id
        return macro_id

    def resolve(self, alias_or_macro_id: str) -> str:
        return self._aliases.get(alias_or_macro_id, alias_or_macro_id)

    def _write(self, command: str) -> None:
        self._lines.append(command.rstrip())
        self._lines.append("")

    def _assignments(self, **assignments: object) -> str:
        return " ".join(f"{key}={_format_value(value)}" for key, value in assignments.items() if value is not None)

    def comment(self, text: str) -> "MacroBuilder":
        for line in text.splitlines():
            self._lines.append(f"# {line}".rstrip())
        self._lines.append("")
        return self

    def raw(self, command: str) -> "MacroBuilder":
        for line in command.splitlines():
            self._lines.append(line.rstrip())
        self._lines.append("")
        return self

    def add_project(self, *, alias: str | None = "project", template: str | Path | None = None, macro_id: str | None = None) -> str:
        project_id = self._remember(alias, macro_id or self._next_id("project"))
        command = f"add project id={project_id}"
        if template is not None:
            command += f" template={_format_path(template)}"
        self._write(command)
        self.active_project = project_id
        return project_id

    def using(self, alias_or_macro_id: str) -> "MacroBuilder":
        self.active_project = self.resolve(alias_or_macro_id)
        self._write(f"using {self.active_project}")
        return self

    def modify(self, alias_or_macro_id: str, **assignments: object) -> "MacroBuilder":
        target = self.resolve(alias_or_macro_id)
        payload = self._assignments(**assignments)
        self._write(f"modify {target} {payload}".rstrip())
        return self

    def add_variable(self, *, alias: str | None, var_name: str, value: object, macro_id: str | None = None) -> str:
        variable_id = self._remember(alias, macro_id or self._next_id("variable"))
        self._write(f"add variable id={variable_id} var_name={_format_value(var_name)} value={_format_value(value)}")
        return variable_id

    def add_dielectric(self, *, alias: str | None, name: str, eps: object, tan: object, macro_id: str | None = None) -> str:
        dielectric_id = self._remember(alias, macro_id or self._next_id("dielectric"))
        self._write(
            f"add dielectric id={dielectric_id} Name={_format_value(name)} "
            f"Eps:Value={_format_value(eps)} Tan:Value={_format_value(tan)}"
        )
        return dielectric_id

    def add_dielectric_layer(self, *, alias: str | None, number: int, thickness: object, macro_id: str | None = None) -> str:
        layer_id = self._remember(alias, macro_id or self._next_id("diel_layer"))
        self._write(f"add diel_layer id={layer_id} num={_format_value(number)} Thickness={_format_value(thickness)}")
        return layer_id

    def add_conductor(self, *, alias: str | None, name: str, conductivity: object, macro_id: str | None = None) -> str:
        conductor_id = self._remember(alias, macro_id or self._next_id("conductor"))
        self._write(f"add conductor id={conductor_id} Name={_format_value(name)} Conductivity={_format_value(conductivity)}")
        return conductor_id

    def add_planar_tech_layer(
        self,
        *,
        alias: str | None,
        diel_layer: str,
        thickness: object,
        model_type: str,
        material_name: str,
        name: str,
        macro_id: str | None = None,
    ) -> str:
        tech_layer_id = self._remember(alias, macro_id or self._next_id("planar_tech_layer"))
        self._write(
            f"add tech_layer_planar id={tech_layer_id} diel_layer={self.resolve(diel_layer)} "
            f"Thickness={_format_value(thickness)} ModelType={_format_value(model_type)} "
            f"MaterialName={_format_value(material_name)} Name={_format_value(name)}"
        )
        return tech_layer_id

    def add_polygon(
        self,
        *,
        alias: str | None,
        tech_layer: str,
        points: str | Iterable[tuple[float, float]],
        macro_id: str | None = None,
    ) -> str:
        polygon_id = self._remember(alias, macro_id or self._next_id("polygon"))
        self._write(
            f"add polygon id={polygon_id} tech_layer={self.resolve(tech_layer)} points={_format_value(_format_points(points))}"
        )
        return polygon_id

    def add_port(self, *, alias: str | None, port_number: int, polygon: str, edge: int, macro_id: str | None = None) -> str:
        port_id = self._remember(alias, macro_id or self._next_id("port"))
        self._write(
            f"add port id={port_id} port_number={_format_value(port_number)} poly={self.resolve(polygon)} edge={_format_value(edge)}"
        )
        return port_id

    def add_sweepset(self, *, alias: str | None, macro_id: str | None = None) -> str:
        sweepset_id = self._remember(alias, macro_id or self._next_id("sweepset"))
        self._write(f"add sweepset id={sweepset_id}")
        return sweepset_id

    def add_freq_sweep(
        self,
        *,
        alias: str | None,
        sweep_set: str,
        sweep_type: str,
        start: object | None = None,
        stop: object | None = None,
        step: object | None = None,
        num_points: int | None = None,
        freqs: Iterable[object] | None = None,
        freq: object | None = None,
        macro_id: str | None = None,
    ) -> str:
        freq_sweep_id = self._remember(alias, macro_id or self._next_id("freq_sweep"))
        lowered = sweep_type.strip().lower()
        params: list[str] = []
        if lowered in {"adaptive", "linear", "exponential"}:
            if start is None or stop is None:
                raise ValueError(f"{lowered} sweep requires start and stop")
            params.extend([f"start={_format_value(start)}", f"stop={_format_value(stop)}"])
            if lowered == "linear":
                if step is None:
                    raise ValueError("linear sweep requires step")
                params.append(f"step={_format_value(step)}")
            if lowered == "exponential":
                if num_points is None:
                    raise ValueError("exponential sweep requires num_points")
                params.append(f"num_points={_format_value(num_points)}")
        elif lowered == "list":
            if not freqs:
                raise ValueError("list sweep requires freqs")
            params.append(f"freqs={','.join(_format_value(item) for item in freqs)}")
        elif lowered == "single_freq":
            if freq is None:
                raise ValueError("single_freq sweep requires freq")
            params.append(f"freq={_format_value(freq)}")
        elif lowered != "dc_freq":
            raise ValueError(f"Unsupported frequency sweep type: {sweep_type}")
        self._write(
            f"add freq_sweep id={freq_sweep_id} set={self.resolve(sweep_set)} {lowered} {' '.join(params)}".rstrip()
        )
        return freq_sweep_id

    def add_param_sweep(
        self,
        *,
        alias: str | None,
        sweep_set: str,
        variable: str,
        sweep_type: str,
        start: object | None = None,
        stop: object | None = None,
        step: object | None = None,
        num_points: int | None = None,
        values: Iterable[object] | None = None,
        value: object | None = None,
        min_value: object | None = None,
        max_value: object | None = None,
        macro_id: str | None = None,
    ) -> str:
        param_sweep_id = self._remember(alias, macro_id or self._next_id("param_sweep"))
        lowered = sweep_type.strip().lower()
        params: list[str] = []
        if lowered in {"linear", "exponential"}:
            if start is None or stop is None:
                raise ValueError(f"{lowered} parameter sweep requires start and stop")
            params.extend([f"start={_format_value(start)}", f"stop={_format_value(stop)}"])
            if lowered == "linear":
                if step is None:
                    raise ValueError("linear parameter sweep requires step")
                params.append(f"step={_format_value(step)}")
            if lowered == "exponential":
                if num_points is None:
                    raise ValueError("exponential parameter sweep requires num_points")
                params.append(f"num_points={_format_value(num_points)}")
        elif lowered == "list":
            if not values:
                raise ValueError("list parameter sweep requires values")
            params.append(f"values={','.join(_format_value(item) for item in values)}")
        elif lowered == "single_value":
            if value is None:
                raise ValueError("single_value parameter sweep requires value")
            params.append(f"value={_format_value(value)}")
        elif lowered in {"corner", "sensitivity"}:
            if min_value is None or max_value is None:
                raise ValueError(f"{lowered} parameter sweep requires min and max")
            params.extend([f"min={_format_value(min_value)}", f"max={_format_value(max_value)}"])
        else:
            raise ValueError(f"Unsupported parameter sweep type: {sweep_type}")
        self._write(
            f"add param_sweep id={param_sweep_id} set={self.resolve(sweep_set)} var={self.resolve(variable)} {lowered} {' '.join(params)}".rstrip()
        )
        return param_sweep_id

    def add_output_file(
        self,
        *,
        alias: str | None,
        path: str | Path,
        param: str = "S",
        file_format: str = "TOUCH",
        format: str = "MA",
        macro_id: str | None = None,
    ) -> str:
        output_id = self._remember(alias, macro_id or self._next_id("output_file"))
        self._write(
            f"add output_file id={output_id} path={_format_path(path)} "
            f"param={_format_value(param)} file_format={_format_value(file_format)} format={_format_value(format)}"
        )
        return output_id

    def delete_output_file(self, alias_or_macro_id: str) -> "MacroBuilder":
        self._write(f"delete output_file {self.resolve(alias_or_macro_id)}")
        return self

    def move_polygon(self, alias_or_macro_id: str, *, x: object, y: object) -> "MacroBuilder":
        self._write(f"move_polygon {self.resolve(alias_or_macro_id)} by {_format_value((x, y))}")
        return self

    def move_vertex(self, alias_or_macro_id: str, *, vertex: int, mode: str, x: object, y: object) -> "MacroBuilder":
        lowered = mode.lower()
        if lowered not in {"to", "by"}:
            raise ValueError("move_vertex mode must be 'to' or 'by'")
        self._write(f"move_vertex {self.resolve(alias_or_macro_id)} vertex={vertex} {lowered} {_format_value((x, y))}")
        return self

    def save(self, path: str | Path) -> "MacroBuilder":
        self._write(f"save path={_format_path(path)}")
        return self

    def verify_project(self) -> "MacroBuilder":
        self._write("verify_project")
        return self

    def analyze(self, *, monitor: bool = False, server: str | None = None) -> "MacroBuilder":
        command = "analyze"
        if monitor:
            command += " monitor"
        if server:
            command += f" server={_format_value(server)}"
        self._write(command)
        return self

    def clean_data(self) -> "MacroBuilder":
        self._write("clean_data")
        return self

    def get_num_ports(self) -> "MacroBuilder":
        self._write("get num_ports")
        return self

    def get_variables(self) -> "MacroBuilder":
        self._write("get variables")
        return self

    def get_optimization(self) -> "MacroBuilder":
        self._write("get optimization")
        return self

    def render(self) -> str:
        return "\n".join(self._lines).rstrip() + "\n"

    def write(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.render(), encoding="utf-8")
        return output_path


class MacroScript(MacroBuilder):
    def add_project(self, template: str | Path | None = None, project_id: str = "project1") -> str:  # type: ignore[override]
        return super().add_project(alias=project_id, template=template, macro_id=project_id)

    def add_polygon(self, tech_layer: str, points: str | Iterable[tuple[float, float]], polygon_id: str = "polygon1") -> str:  # type: ignore[override]
        return super().add_polygon(alias=polygon_id, tech_layer=tech_layer, points=points, macro_id=polygon_id)
