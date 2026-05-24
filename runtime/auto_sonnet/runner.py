from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .discovery import SonnetInstallation, discover_installation


def _parse_json_payloads(stdout: str) -> list[object]:
    payloads: list[object] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] not in "[{":
            continue
        try:
            payloads.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return payloads


@dataclass
class ExecutionResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    cwd: Path
    elapsed_seconds: float
    json_payloads: list[object] = field(default_factory=list)

    def assert_success(self) -> "ExecutionResult":
        if self.returncode != 0:
            raise RuntimeError(
                f"runmacro execution failed with return code {self.returncode}\nSTDOUT:\n{self.stdout}\nSTDERR:\n{self.stderr}"
            )
        return self


class SonnetRunner:
    def __init__(self, installation: SonnetInstallation | None = None) -> None:
        self.installation = installation or discover_installation()

    @classmethod
    def from_discovery(
        cls,
        *,
        sonnet_dir: str | Path | None = None,
        runmacro_path: str | Path | None = None,
    ) -> "SonnetRunner":
        return cls(discover_installation(sonnet_dir=sonnet_dir, runmacro_path=runmacro_path))

    def version(self) -> str:
        completed = subprocess.run(
            [str(self.installation.runmacro_path), "-ver"],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
        )
        return (completed.stdout or completed.stderr).strip()

    def run_macro(
        self,
        macro_path: str | Path,
        *,
        cwd: str | Path | None = None,
        verbose: bool = False,
        timeout: int | None = None,
    ) -> ExecutionResult:
        macro_file = Path(macro_path).resolve()
        working_directory = Path(cwd).resolve() if cwd is not None else macro_file.parent
        command = [str(self.installation.runmacro_path)]
        if verbose:
            command.append("-v")
        command.append(str(macro_file))

        env = os.environ.copy()
        env.setdefault("SONNET_DIR", str(self.installation.sonnet_dir))

        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=working_directory,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            encoding="utf-8",
            env=env,
        )
        elapsed = time.perf_counter() - started
        return ExecutionResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            cwd=working_directory,
            elapsed_seconds=elapsed,
            json_payloads=_parse_json_payloads(completed.stdout),
        )


def run_macro(
    macro_path: str | Path,
    *,
    sonnet_dir: str | Path | None = None,
    runmacro_path: str | Path | None = None,
    cwd: str | Path | None = None,
    verbose: bool = False,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    installation = discover_installation(sonnet_dir=sonnet_dir, runmacro_path=runmacro_path)
    macro_file = Path(macro_path).resolve()
    working_directory = Path(cwd).resolve() if cwd is not None else macro_file.parent
    command = [str(installation.runmacro_path)]
    if verbose:
        command.append("-v")
    command.append(str(macro_file))
    env = os.environ.copy()
    env.setdefault("SONNET_DIR", str(installation.sonnet_dir))
    try:
        return subprocess.run(
            command,
            cwd=working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            encoding="utf-8",
            env=env,
        )
    except TypeError:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            encoding="utf-8",
        )
