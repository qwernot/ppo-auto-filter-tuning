from __future__ import annotations

from pathlib import Path

from .discovery import discover_installation


def find_sonnet_dir(sonnet_dir: str | Path | None = None) -> Path:
    return discover_installation(sonnet_dir=sonnet_dir).sonnet_dir


def find_runmacro_path(sonnet_dir: str | Path | None = None) -> Path:
    if sonnet_dir is None:
        return discover_installation().runmacro_path
    return discover_installation(sonnet_dir=sonnet_dir).runmacro_path

