from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


_TOUCHSTONE_SUFFIX = re.compile(r"\.s(\d+)p$", re.IGNORECASE)
_FREQUENCY_SCALE = {
    "HZ": 1.0,
    "KHZ": 1e3,
    "MHZ": 1e6,
    "GHZ": 1e9,
    "THZ": 1e12,
}


@dataclass(slots=True)
class TouchstoneData:
    path: Path | None
    frequency_hz: list[float]
    s11: list[complex]
    s21: list[complex] = field(default_factory=list)
    s12: list[complex] = field(default_factory=list)
    s22: list[complex] = field(default_factory=list)
    frequency_unit: str = "GHZ"
    parameter: str = "S"
    data_format: str = "MA"
    reference_resistance: float = 50.0
    port_count: int = 2

    def __len__(self) -> int:
        return len(self.frequency_hz)


def _infer_port_count(path: Path) -> int:
    match = _TOUCHSTONE_SUFFIX.search(path.name)
    if match is None:
        raise ValueError(f"Unable to infer Touchstone port count from filename: {path.name}")
    return int(match.group(1))


def _parse_option_line(line: str) -> tuple[str, str, str, float]:
    tokens = line[1:].split()
    if len(tokens) < 3:
        raise ValueError(f"Invalid Touchstone option line: {line}")

    frequency_unit = tokens[0].upper()
    parameter = tokens[1].upper()
    data_format = tokens[2].upper()
    if frequency_unit not in _FREQUENCY_SCALE:
        raise ValueError(f"Unsupported Touchstone frequency unit: {frequency_unit}")
    if parameter != "S":
        raise ValueError(f"Unsupported Touchstone parameter type: {parameter}")
    if data_format not in {"MA", "DB", "RI"}:
        raise ValueError(f"Unsupported Touchstone data format: {data_format}")

    reference_resistance = 50.0
    if len(tokens) > 3:
        if len(tokens) < 5 or tokens[3].upper() != "R":
            raise ValueError(f"Unsupported Touchstone option line tail: {line}")
        reference_resistance = float(tokens[4])
    return frequency_unit, parameter, data_format, reference_resistance


def _complex_from_pair(first: float, second: float, data_format: str) -> complex:
    if data_format == "RI":
        return complex(first, second)
    if data_format == "MA":
        radians = math.radians(second)
        return complex(first * math.cos(radians), first * math.sin(radians))
    if data_format == "DB":
        magnitude = 10 ** (first / 20.0)
        radians = math.radians(second)
        return complex(magnitude * math.cos(radians), magnitude * math.sin(radians))
    raise ValueError(f"Unsupported Touchstone data format: {data_format}")


def get_sparameter(data: TouchstoneData, key: str) -> list[complex]:
    normalized = key.strip().lower()
    mapping = {
        "s11": data.s11,
        "s21": data.s21,
        "s12": data.s12,
        "s22": data.s22,
    }
    if normalized in mapping:
        values = mapping[normalized]
        if normalized == "s11" or values:
            return values
        raise KeyError(f"S-parameter {key} is not available for a {data.port_count}-port Touchstone file")
    raise KeyError(f"Unsupported S-parameter key: {key}")


def to_db(values: Iterable[complex | float]) -> list[float]:
    converted: list[float] = []
    for value in values:
        magnitude = abs(value)
        if magnitude == 0.0:
            converted.append(float("-inf"))
        else:
            converted.append(20.0 * math.log10(magnitude))
    return converted


def load_touchstone(path: str | Path) -> TouchstoneData:
    touchstone_path = Path(path).resolve()
    port_count = _infer_port_count(touchstone_path)
    if port_count not in {1, 2}:
        raise NotImplementedError("Only 1-port and 2-port Touchstone files are currently supported")

    frequency_unit = "GHZ"
    parameter = "S"
    data_format = "MA"
    reference_resistance = 50.0
    option_seen = False

    frequency_hz: list[float] = []
    s11: list[complex] = []
    s21: list[complex] = []
    s12: list[complex] = []
    s22: list[complex] = []

    expected_tokens = 1 + 2 * port_count * port_count
    pending_tokens: list[str] = []

    for line_number, raw_line in enumerate(touchstone_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            frequency_unit, parameter, data_format, reference_resistance = _parse_option_line(line)
            option_seen = True
            continue

        pending_tokens.extend(line.split())
        while len(pending_tokens) >= expected_tokens:
            chunk = pending_tokens[:expected_tokens]
            pending_tokens = pending_tokens[expected_tokens:]
            try:
                numeric = [float(token) for token in chunk]
            except ValueError as exc:
                raise ValueError(f"Invalid numeric data in Touchstone file at line {line_number}") from exc

            scale = _FREQUENCY_SCALE[frequency_unit]
            frequency_hz.append(numeric[0] * scale)
            s11.append(_complex_from_pair(numeric[1], numeric[2], data_format))
            if port_count == 2:
                s21.append(_complex_from_pair(numeric[3], numeric[4], data_format))
                s12.append(_complex_from_pair(numeric[5], numeric[6], data_format))
                s22.append(_complex_from_pair(numeric[7], numeric[8], data_format))

    if not option_seen:
        raise ValueError(f"Touchstone option line not found in file: {touchstone_path}")
    if pending_tokens:
        raise ValueError(f"Incomplete Touchstone data record in file: {touchstone_path}")
    if not frequency_hz:
        raise ValueError(f"No Touchstone data points found in file: {touchstone_path}")

    return TouchstoneData(
        path=touchstone_path,
        frequency_hz=frequency_hz,
        s11=s11,
        s21=s21,
        s12=s12,
        s22=s22,
        frequency_unit=frequency_unit,
        parameter=parameter,
        data_format=data_format,
        reference_resistance=reference_resistance,
        port_count=port_count,
    )
