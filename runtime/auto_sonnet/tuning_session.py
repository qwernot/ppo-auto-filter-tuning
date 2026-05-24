from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .analysis import TouchstoneAnalysis
from .runtime_heuristics import RuntimeAnomalyReport, assess_runtime_anomaly
from .runner import SonnetRunner
from .sonx import write_project_with_variables
from .specs import AutomationSpec, OutputFileSpec, ProjectSettings
from .tuning_manifest import TuningManifest, load_tuning_manifest
from .workflow import GeneratedArtifacts, RunArtifacts, RunAndAnalyzeArtifacts, SonnetAutomation


@dataclass(slots=True)
class TuningEvaluationResult:
    manifest: TuningManifest
    variable_values: dict[str, float]
    edited_template_path: Path
    run: RunArtifacts
    analysis: TouchstoneAnalysis | None
    runtime_report: RuntimeAnomalyReport | None
    result_path: Path | None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_name": self.manifest.name,
            "template_path": str(self.manifest.template_path),
            "edited_template_path": str(self.edited_template_path),
            "variable_values": self.variable_values,
            "run": {
                "workspace": str(self.run.workspace),
                "macro_path": str(self.run.macro_path),
                "project_path": str(self.run.project_path),
                "command": self.run.execution.command,
                "returncode": self.run.execution.returncode,
                "stdout": self.run.execution.stdout,
                "stderr": self.run.execution.stderr,
                "elapsed_seconds": self.run.execution.elapsed_seconds,
            },
        }
        if self.analysis is not None:
            payload["analysis"] = self.analysis.to_dict()
        if self.runtime_report is not None:
            payload["runtime_report"] = self.runtime_report.to_dict()
        if self.result_path is not None:
            payload["result_path"] = str(self.result_path)
        return payload


@dataclass(slots=True)
class PreparedTuningStepArtifacts:
    manifest: TuningManifest
    variable_values: dict[str, float]
    edited_template_path: Path
    generated: GeneratedArtifacts
    expected_s2p_path: Path
    job_path: Path | None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_name": self.manifest.name,
            "template_path": str(self.manifest.template_path),
            "edited_template_path": str(self.edited_template_path),
            "variable_values": self.variable_values,
            "workspace": str(self.generated.workspace),
            "macro_path": str(self.generated.macro_path),
            "project_path": str(self.generated.project_path),
            "expected_s2p_path": str(self.expected_s2p_path),
        }
        if self.manifest.target_path is not None:
            payload["target_path"] = str(self.manifest.target_path)
        payload["analysis"] = {
            "peak_search_start_hz": self.manifest.analysis.peak_search_start_hz,
            "peak_search_stop_hz": self.manifest.analysis.peak_search_stop_hz,
            "bandwidth_drop_db": self.manifest.analysis.bandwidth_drop_db,
            "high_side_search_start_hz": self.manifest.analysis.high_side_search_start_hz,
            "high_side_search_stop_hz": self.manifest.analysis.high_side_search_stop_hz,
        }
        if self.job_path is not None:
            payload["job_path"] = str(self.job_path)
        return payload


def _load_manifest(manifest: str | Path | TuningManifest) -> TuningManifest:
    if isinstance(manifest, TuningManifest):
        return manifest
    return load_tuning_manifest(manifest)


def _build_template_eval_spec(
    manifest: TuningManifest,
    template_reference: str | Path,
    *,
    server: str | None = None,
) -> AutomationSpec:
    return AutomationSpec(
        project=ProjectSettings(
            save_path=manifest.save_path,
            template=str(template_reference),
            verify_before_analyze=True,
            analyze=True,
            server=server if server is not None else manifest.server,
        ),
        output_files=[OutputFileSpec(key="touchstone", path="$BASENAME.s2p", param="S", file_format="TOUCH", format="MA")],
    )


def _clear_workspace(workspace_path: Path) -> None:
    if workspace_path.exists():
        for child in workspace_path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    workspace_path.mkdir(parents=True, exist_ok=True)


def _resolve_expected_output_path(project_path: Path, raw_path: str) -> Path:
    substituted = raw_path.replace("$BASENAME", project_path.stem)
    target = Path(substituted)
    if target.is_absolute():
        return target
    return project_path.parent / target


def prepare_tuning_step(
    manifest: str | Path | TuningManifest,
    variable_values: dict[str, float],
    workspace: str | Path,
    *,
    server: str | None = None,
    job_filename: str = "step_job.json",
) -> PreparedTuningStepArtifacts:
    tuning_manifest = _load_manifest(manifest)
    resolved_values = tuning_manifest.resolve_values(variable_values)
    template_updates = tuning_manifest.to_template_updates(resolved_values)

    workspace_path = Path(workspace).resolve()
    _clear_workspace(workspace_path)
    edited_template_path = workspace_path / "Template" / tuning_manifest.template_path.name
    write_project_with_variables(
        tuning_manifest.template_path,
        edited_template_path,
        template_updates,
        disable_variable_sweeps=True,
        clear_output_files=True,
    )

    template_reference = Path("Template") / tuning_manifest.template_path.name
    spec = _build_template_eval_spec(tuning_manifest, template_reference, server=server)
    generated = SonnetAutomation().generate(spec, workspace_path)
    expected_s2p_path = _resolve_expected_output_path(generated.project_path, spec.output_files[0].path)

    job_path = workspace_path / job_filename if job_filename else None
    prepared = PreparedTuningStepArtifacts(
        manifest=tuning_manifest,
        variable_values=resolved_values,
        edited_template_path=edited_template_path,
        generated=generated,
        expected_s2p_path=expected_s2p_path,
        job_path=job_path,
    )
    if job_path is not None:
        job_path.write_text(json.dumps(prepared.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return prepared


def evaluate_tuning_step(
    manifest: str | Path | TuningManifest,
    variable_values: dict[str, float],
    workspace: str | Path,
    *,
    runner: SonnetRunner | None = None,
    server: str | None = None,
    result_filename: str = "step_result.json",
    verbose: bool = False,
    timeout: int | None = None,
    runtime_reference_seconds: float | None = None,
    runtime_warning_ratio: float = 2.0,
    runtime_severe_ratio: float = 3.0,
) -> TuningEvaluationResult:
    prepared = prepare_tuning_step(manifest, variable_values, workspace, server=server, job_filename="")
    tuning_manifest = prepared.manifest
    resolved_values = prepared.variable_values
    workspace_path = prepared.generated.workspace
    edited_template_path = prepared.edited_template_path
    template_reference = Path("Template") / tuning_manifest.template_path.name
    spec = _build_template_eval_spec(tuning_manifest, template_reference, server=server)
    automation = SonnetAutomation(runner=runner or SonnetRunner())
    analyzed: RunAndAnalyzeArtifacts = automation.run_and_analyze(
        spec,
        workspace_path,
        target_path=tuning_manifest.target_path,
        config=tuning_manifest.analysis,
        analysis_filename="metrics.json",
        verbose=verbose,
        timeout=timeout,
    )

    result_path = workspace_path / result_filename
    runtime_report = assess_runtime_anomaly(
        analyzed.run.execution.elapsed_seconds,
        runtime_reference_seconds,
        warning_ratio=runtime_warning_ratio,
        severe_ratio=runtime_severe_ratio,
    )
    result = TuningEvaluationResult(
        manifest=tuning_manifest,
        variable_values=resolved_values,
        edited_template_path=edited_template_path,
        run=analyzed.run,
        analysis=analyzed.analysis,
        runtime_report=runtime_report,
        result_path=result_path,
    )
    result_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return result
