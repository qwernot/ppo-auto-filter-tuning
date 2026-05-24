from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable

from .rl_env import ObjectiveSummary, RewardConfig, summarize_objective
from .runtime_heuristics import RuntimeAnomalyReport, build_runtime_reference
from .runner import SonnetRunner
from .tuning_manifest import TuningManifest, load_tuning_manifest
from .tuning_session import evaluate_tuning_step


def _load_manifest(manifest: str | Path | TuningManifest) -> TuningManifest:
    if isinstance(manifest, TuningManifest):
        return manifest
    return load_tuning_manifest(manifest)


@dataclass(slots=True)
class BaselineSearchEvaluation:
    tag: str
    variable_values: dict[str, float]
    workspace: Path
    objective: ObjectiveSummary | None
    target_satisfied: bool
    analysis_available: bool
    elapsed_seconds: float | None = None
    runtime_report: RuntimeAnomalyReport | None = None
    accepted: bool = False

    @property
    def sort_loss(self) -> float:
        if self.objective is None:
            return float("inf")
        return float(self.objective.total_loss)

    def to_dict(self) -> dict[str, object]:
        return {
            "tag": self.tag,
            "variable_values": self.variable_values,
            "workspace": str(self.workspace),
            "analysis_available": self.analysis_available,
            "target_satisfied": self.target_satisfied,
            "accepted": self.accepted,
            "elapsed_seconds": self.elapsed_seconds,
            "sort_loss": self.sort_loss,
            "objective": None if self.objective is None else self.objective.to_dict(),
            "runtime_report": None if self.runtime_report is None else self.runtime_report.to_dict(),
        }


@dataclass(slots=True)
class BaselineSearchResult:
    mode: str
    initial: BaselineSearchEvaluation
    best: BaselineSearchEvaluation
    evaluations: list[BaselineSearchEvaluation]
    accepted_tags: list[str]
    stopped_reason: str

    def summary_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "total_evaluations": len(self.evaluations),
            "accepted_tags": self.accepted_tags,
            "stopped_reason": self.stopped_reason,
            "success": self.best.target_satisfied,
            "initial_loss": self.initial.sort_loss,
            "best_loss": self.best.sort_loss,
            "best_tag": self.best.tag,
            "best_variable_values": self.best.variable_values,
            "best_objective": None if self.best.objective is None else self.best.objective.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "initial": self.initial.to_dict(),
            "best": self.best.to_dict(),
            "accepted_tags": self.accepted_tags,
            "stopped_reason": self.stopped_reason,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "summary": self.summary_dict(),
        }


def _evaluation_key(manifest: TuningManifest, values: dict[str, float]) -> tuple[float, ...]:
    return tuple(float(values[variable.name]) for variable in manifest.variables)


def _evaluate_candidate(
    manifest: TuningManifest,
    values: dict[str, float],
    workspace: Path,
    *,
    tag: str,
    runner: SonnetRunner | None = None,
    server: str | None = None,
    evaluate_fn: Callable[..., Any] = evaluate_tuning_step,
    reward_config: RewardConfig | None = None,
    verbose: bool = False,
    timeout: int | None = None,
    runtime_reference_seconds: float | None = None,
) -> BaselineSearchEvaluation:
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate_fn(
            manifest,
            values,
            workspace,
            runner=runner,
            server=server,
            verbose=verbose,
            timeout=timeout,
            runtime_reference_seconds=runtime_reference_seconds,
        )
    except TypeError:
        result = evaluate_fn(manifest, values, workspace)

    analysis_available = getattr(result, "analysis", None) is not None
    objective = summarize_objective(result, reward_config) if analysis_available else None
    return BaselineSearchEvaluation(
        tag=tag,
        variable_values=dict(result.variable_values),
        workspace=workspace,
        objective=objective,
        target_satisfied=bool(analysis_available and result.analysis.target_satisfied),
        analysis_available=analysis_available,
        elapsed_seconds=float(result.run.execution.elapsed_seconds),
        runtime_report=getattr(result, "runtime_report", None),
    )


def coordinate_search_tuning(
    manifest: str | Path | TuningManifest,
    workspace: str | Path,
    *,
    initial_values: dict[str, float] | None = None,
    runner: SonnetRunner | None = None,
    server: str | None = None,
    evaluate_fn: Callable[..., Any] = evaluate_tuning_step,
    reward_config: RewardConfig | None = None,
    max_rounds: int = 3,
    step_scale: float = 1.0,
    improvement_tolerance: float = 1e-9,
    verbose: bool = False,
    timeout: int | None = None,
) -> BaselineSearchResult:
    tuning_manifest = _load_manifest(manifest)
    workspace_path = Path(workspace).resolve()
    current_values = tuning_manifest.resolve_values(initial_values)
    initial = _evaluate_candidate(
        tuning_manifest,
        current_values,
        workspace_path / "initial",
        tag="initial",
        runner=runner,
        server=server,
        evaluate_fn=evaluate_fn,
        reward_config=reward_config,
        verbose=verbose,
        timeout=timeout,
    )
    evaluations = [initial]
    accepted_tags: list[str] = []
    best = initial
    runtime_samples = [initial.elapsed_seconds] if initial.elapsed_seconds is not None else []

    if best.target_satisfied:
        return BaselineSearchResult(
            mode="coordinate",
            initial=initial,
            best=best,
            evaluations=evaluations,
            accepted_tags=accepted_tags,
            stopped_reason="initial_target_satisfied",
        )

    for round_index in range(1, max_rounds + 1):
        current_key = _evaluation_key(tuning_manifest, best.variable_values)
        seen = {current_key}
        round_best: BaselineSearchEvaluation | None = None
        round_best_loss = best.sort_loss

        for variable in tuning_manifest.variables:
            if variable.max_delta is None:
                continue
            for direction_label, direction_sign in (("minus", -1.0), ("plus", 1.0)):
                raw_candidate = dict(best.variable_values)
                raw_candidate[variable.name] = float(best.variable_values[variable.name]) + direction_sign * step_scale * float(
                    variable.max_delta
                )
                candidate_values = tuning_manifest.resolve_values(raw_candidate)
                candidate_key = _evaluation_key(tuning_manifest, candidate_values)
                if candidate_key in seen:
                    continue
                seen.add(candidate_key)

                candidate = _evaluate_candidate(
                    tuning_manifest,
                    candidate_values,
                    workspace_path / f"round_{round_index:02d}" / f"{variable.name}_{direction_label}",
                    tag=f"round_{round_index:02d}/{variable.name}_{direction_label}",
                    runner=runner,
                    server=server,
                    evaluate_fn=evaluate_fn,
                    reward_config=reward_config,
                    verbose=verbose,
                    timeout=timeout,
                    runtime_reference_seconds=build_runtime_reference(runtime_samples),
                )
                evaluations.append(candidate)
                if candidate.elapsed_seconds is not None:
                    runtime_samples.append(candidate.elapsed_seconds)

                if candidate.sort_loss + improvement_tolerance < round_best_loss:
                    round_best = candidate
                    round_best_loss = candidate.sort_loss

        if round_best is None:
            return BaselineSearchResult(
                mode="coordinate",
                initial=initial,
                best=best,
                evaluations=evaluations,
                accepted_tags=accepted_tags,
                stopped_reason="no_improving_candidate",
            )

        round_best.accepted = True
        accepted_tags.append(round_best.tag)
        best = round_best
        if best.target_satisfied:
            return BaselineSearchResult(
                mode="coordinate",
                initial=initial,
                best=best,
                evaluations=evaluations,
                accepted_tags=accepted_tags,
                stopped_reason="target_satisfied",
            )

    return BaselineSearchResult(
        mode="coordinate",
        initial=initial,
        best=best,
        evaluations=evaluations,
        accepted_tags=accepted_tags,
        stopped_reason="max_rounds_reached",
    )


def random_search_tuning(
    manifest: str | Path | TuningManifest,
    workspace: str | Path,
    *,
    initial_values: dict[str, float] | None = None,
    runner: SonnetRunner | None = None,
    server: str | None = None,
    evaluate_fn: Callable[..., Any] = evaluate_tuning_step,
    reward_config: RewardConfig | None = None,
    samples: int = 20,
    seed: int | None = None,
    improvement_tolerance: float = 1e-9,
    stop_on_success: bool = True,
    verbose: bool = False,
    timeout: int | None = None,
) -> BaselineSearchResult:
    tuning_manifest = _load_manifest(manifest)
    workspace_path = Path(workspace).resolve()
    rng = Random(seed)
    current_values = tuning_manifest.resolve_values(initial_values)
    initial = _evaluate_candidate(
        tuning_manifest,
        current_values,
        workspace_path / "initial",
        tag="initial",
        runner=runner,
        server=server,
        evaluate_fn=evaluate_fn,
        reward_config=reward_config,
        verbose=verbose,
        timeout=timeout,
    )
    evaluations = [initial]
    accepted_tags: list[str] = []
    best = initial
    runtime_samples = [initial.elapsed_seconds] if initial.elapsed_seconds is not None else []

    if best.target_satisfied and stop_on_success:
        return BaselineSearchResult(
            mode="random",
            initial=initial,
            best=best,
            evaluations=evaluations,
            accepted_tags=accepted_tags,
            stopped_reason="initial_target_satisfied",
        )

    seen = {_evaluation_key(tuning_manifest, best.variable_values)}
    generated = 0
    attempts = 0
    max_attempts = max(samples * 10, 10)

    while generated < samples and attempts < max_attempts:
        attempts += 1
        raw_candidate: dict[str, float] = {}
        for variable in tuning_manifest.variables:
            if variable.min_value is not None and variable.max_value is not None:
                raw_candidate[variable.name] = rng.uniform(variable.min_value, variable.max_value)
            elif variable.default_value is not None:
                raw_candidate[variable.name] = float(variable.default_value)
            else:
                raise ValueError(f"Unable to random-sample variable '{variable.name}' without bounds or default value")

        candidate_values = tuning_manifest.resolve_values(raw_candidate)
        candidate_key = _evaluation_key(tuning_manifest, candidate_values)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        generated += 1

        candidate = _evaluate_candidate(
            tuning_manifest,
            candidate_values,
            workspace_path / f"sample_{generated:04d}",
            tag=f"sample_{generated:04d}",
            runner=runner,
            server=server,
            evaluate_fn=evaluate_fn,
            reward_config=reward_config,
            verbose=verbose,
            timeout=timeout,
            runtime_reference_seconds=build_runtime_reference(runtime_samples),
        )
        evaluations.append(candidate)
        if candidate.elapsed_seconds is not None:
            runtime_samples.append(candidate.elapsed_seconds)

        if candidate.sort_loss + improvement_tolerance < best.sort_loss:
            candidate.accepted = True
            accepted_tags.append(candidate.tag)
            best = candidate
            if best.target_satisfied and stop_on_success:
                return BaselineSearchResult(
                    mode="random",
                    initial=initial,
                    best=best,
                    evaluations=evaluations,
                    accepted_tags=accepted_tags,
                    stopped_reason="target_satisfied",
                )

    return BaselineSearchResult(
        mode="random",
        initial=initial,
        best=best,
        evaluations=evaluations,
        accepted_tags=accepted_tags,
        stopped_reason="sample_budget_exhausted" if generated >= samples else "sampling_exhausted",
    )
