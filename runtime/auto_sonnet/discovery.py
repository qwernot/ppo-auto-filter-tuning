from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import winreg
except ImportError:  # pragma: no cover - non-Windows fallback
    winreg = None


def _normalize_path(path_like: str | Path) -> Path:
    return Path(os.path.expandvars(str(path_like))).expanduser().resolve()


def _version_key(version: str) -> tuple[int | str, ...]:
    parts = re.split(r"[.\-_]", version)
    key: list[int | str] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def _runmacro_name() -> str:
    return "runmacro.exe" if os.name == "nt" else "runmacro"


def _runmacro_from_dir(sonnet_dir: Path) -> Path:
    return sonnet_dir / "bin" / _runmacro_name()


@dataclass(frozen=True)
class SonnetInstallation:
    sonnet_dir: Path
    runmacro_path: Path
    source: str
    version: str | None = None

    def validate(self) -> "SonnetInstallation":
        if not self.sonnet_dir.exists():
            raise FileNotFoundError(f"Sonnet directory not found: {self.sonnet_dir}")
        if not self.runmacro_path.exists():
            raise FileNotFoundError(f"runmacro executable not found: {self.runmacro_path}")
        return self


def _candidate_from_runmacro(path_like: str | Path, source: str, version: str | None = None) -> SonnetInstallation | None:
    runmacro_path = _normalize_path(path_like)
    if runmacro_path.is_dir():
        runmacro_path = runmacro_path / _runmacro_name()
    if not runmacro_path.exists():
        return None
    return SonnetInstallation(
        sonnet_dir=runmacro_path.parent.parent,
        runmacro_path=runmacro_path,
        source=source,
        version=version,
    )


def _candidate_from_sonnet_dir(path_like: str | Path, source: str, version: str | None = None) -> SonnetInstallation | None:
    sonnet_dir = _normalize_path(path_like)
    if sonnet_dir.name.lower() == "bin" and (sonnet_dir / _runmacro_name()).exists():
        sonnet_dir = sonnet_dir.parent
    runmacro_path = _runmacro_from_dir(sonnet_dir)
    if not runmacro_path.exists():
        return None
    return SonnetInstallation(
        sonnet_dir=sonnet_dir,
        runmacro_path=runmacro_path,
        source=source,
        version=version,
    )


def _iter_registry_installations() -> Iterable[SonnetInstallation]:
    if winreg is None:
        return []

    registry_locations = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Sonnet Software\sonnet"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Sonnet Software\sonnet"),
    ]
    installations: list[SonnetInstallation] = []
    for hive, registry_key in registry_locations:
        try:
            with winreg.OpenKey(hive, registry_key) as base_key:
                index = 0
                while True:
                    try:
                        version = winreg.EnumKey(base_key, index)
                    except OSError:
                        break
                    index += 1
                    try:
                        with winreg.OpenKey(base_key, version) as version_key:
                            sonnet_dir, _ = winreg.QueryValueEx(version_key, "SONNET_DIR")
                    except OSError:
                        continue
                    candidate = _candidate_from_sonnet_dir(sonnet_dir, source="registry", version=version)
                    if candidate is not None:
                        installations.append(candidate)
        except OSError:
            continue
    return installations


def discover_installations(
    *,
    sonnet_dir: str | Path | None = None,
    runmacro_path: str | Path | None = None,
) -> list[SonnetInstallation]:
    candidates: list[SonnetInstallation] = []
    seen: set[Path] = set()

    def add(candidate: SonnetInstallation | None) -> None:
        if candidate is None:
            return
        resolved = candidate.runmacro_path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(candidate)

    add(_candidate_from_runmacro(runmacro_path, source="explicit-runmacro") if runmacro_path else None)
    add(_candidate_from_sonnet_dir(sonnet_dir, source="explicit-sonnet-dir") if sonnet_dir else None)

    env_dir = os.environ.get("SONNET_DIR")
    if env_dir:
        add(_candidate_from_sonnet_dir(env_dir, source="environment"))

    registry_items = sorted(
        _iter_registry_installations(),
        key=lambda item: _version_key(item.version or "0"),
        reverse=True,
    )
    for candidate in registry_items:
        add(candidate)

    for root in [Path("C:/Program Files/Sonnet Software"), Path("D:/Program Files/Sonnet Software")]:
        if not root.exists():
            continue
        children = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda item: _version_key(item.name), reverse=True)
        for child in children:
            add(_candidate_from_sonnet_dir(child, source="common-install-root", version=child.name))

    return candidates


def discover_installation(
    *,
    sonnet_dir: str | Path | None = None,
    runmacro_path: str | Path | None = None,
) -> SonnetInstallation:
    installations = discover_installations(sonnet_dir=sonnet_dir, runmacro_path=runmacro_path)
    if not installations:
        raise FileNotFoundError(
            "Unable to locate a Sonnet 19 installation. "
            "Set SONNET_DIR, or pass --sonnet-dir / --runmacro-path."
        )
    return installations[0].validate()

