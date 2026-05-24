from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable, Sequence

from .analysis import TouchstoneAnalysis
from .filter_targets import FilterTargetSpec
from .runtime_heuristics import build_runtime_reference
from .tuning_manifest import TuningManifest, TuningVariableSpec, load_tuning_manifest
from .tuning_session import evaluate_tuning_step


@dataclass(slots=True)
class RewardConfig:
    improvement_weight: float = 1.0
    absolute_error_weight: float = 0.2
    constraint_violation_weight: float = 1.0
    tracking_error_weight: float = 0.25
    action_penalty_weight: float = 0.05
    success_bonus: float = 10.0
    analysis_failure_penalty: float = 5.0


@dataclass(slots=True)
class ObjectiveSummary:
    violation_components: dict[str, float]
    tracking_components: dict[str, float]
    constraint_violation: float
    tracking_error: float
    total_loss: float
    target_satisfied: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "violation_components": self.violation_components,
            "tracking_components": self.tracking_components,
            "constraint_violation": self.constraint_violation,
            "tracking_error": self.tracking_error,
            "total_loss": self.total_loss,
            "target_satisfied": self.target_satisfied,
        }


def _load_manifest(manifest: str | Path | TuningManifest) -> TuningManifest:
    if isinstance(manifest, TuningManifest):
        return manifest
    return load_tuning_manifest(manifest)


def _target_from_analysis(analysis: TouchstoneAnalysis) -> FilterTargetSpec | None:
    return analysis.target


def _scale(value: float | None, fallback: float = 1.0) -> float:
    if value is None:
        return fallback
    magnitude = abs(value)
    return magnitude if magnitude > 1e-12 else fallback


def _normalize_variable(value: float, variable: TuningVariableSpec) -> float:
    if variable.min_value is None or variable.max_value is None or variable.max_value == variable.min_value:
        return value
    midpoint = (variable.max_value + variable.min_value) / 2.0
    half_span = (variable.max_value - variable.min_value) / 2.0
    return (value - midpoint) / half_span


def _normalize_metric(actual: float | None, scale: float) -> float:
    if actual is None:
        return 0.0
    return actual / _scale(scale)


def _normalize_error(error: float | None, scale: float) -> float:
    if error is None:
        return 0.0
    return error / _scale(scale)


def _positive_error(error: float | None) -> float:
    if error is None:
        return 0.0
    return max(float(error), 0.0)


def _symmetric_violation(error: float | None, tolerance: float | None, scale: float) -> float:
    if error is None:
        return 0.0
    tol = max(float(tolerance or 0.0), 0.0)
    excess = max(abs(float(error)) - tol, 0.0)
    divisor = tol if tol > 1e-12 else scale
    return excess / _scale(divisor)


def _insertion_loss_db(analysis: TouchstoneAnalysis) -> float:
    return -analysis.metrics.main_peak_s21_db


def _return_loss_db(analysis: TouchstoneAnalysis) -> float:
    return -analysis.metrics.best_s11_db


def _high_side_zero_depth_db(analysis: TouchstoneAnalysis) -> float | None:
    if analysis.metrics.high_side_zero_s21_db is None:
        return None
    return -analysis.metrics.high_side_zero_s21_db


def _normalized_error_components(result: Any) -> list[float]:
    analysis = result.analysis
    target = _target_from_analysis(analysis)
    errors = analysis.errors
    if target is None or errors is None:
        return []

    return [
        _normalize_error(
            errors.center_freq_error_hz,
            None if target.center_freq_hz is None else target.center_freq_hz.tolerance or target.center_freq_hz.target,
        ),
        _normalize_error(
            errors.bandwidth_error_hz,
            None if target.bandwidth_hz is None else target.bandwidth_hz.tolerance or target.bandwidth_hz.target,
        ),
        _normalize_error(errors.insertion_loss_error_db, target.max_insertion_loss_db),
        _normalize_error(errors.return_loss_error_db, target.min_return_loss_db),
        _normalize_error(
            errors.high_side_zero_freq_error_hz,
            None
            if target.high_side_zero_freq_hz is None
            else target.high_side_zero_freq_hz.tolerance or target.high_side_zero_freq_hz.target,
        ),
        _normalize_error(errors.high_side_zero_depth_error_db, target.min_high_side_zero_depth_db),
    ]


def _objective_component_maps(result: Any) -> tuple[dict[str, float], dict[str, float]]:
    analysis = result.analysis
    target = _target_from_analysis(analysis)
    errors = analysis.errors
    if target is None or errors is None:
        return {}, {}

    violation_components: dict[str, float] = {}
    tracking_components: dict[str, float] = {}

    if target.center_freq_hz is not None:
        scale = target.center_freq_hz.tolerance or target.center_freq_hz.target
        violation_components["center_freq"] = _symmetric_violation(
            errors.center_freq_error_hz,
            target.center_freq_hz.tolerance,
            scale,
        )
        tracking_components["center_freq"] = abs(_normalize_error(errors.center_freq_error_hz, scale))

    if target.bandwidth_hz is not None:
        scale = target.bandwidth_hz.tolerance or target.bandwidth_hz.target
        violation_components["bandwidth"] = _symmetric_violation(
            errors.bandwidth_error_hz,
            target.bandwidth_hz.tolerance,
            scale,
        )
        tracking_components["bandwidth"] = abs(_normalize_error(errors.bandwidth_error_hz, scale))

    if target.max_insertion_loss_db is not None:
        scale = target.max_insertion_loss_db
        violation_components["insertion_loss"] = _positive_error(errors.insertion_loss_error_db) / _scale(scale)
        tracking_components["insertion_loss"] = violation_components["insertion_loss"]

    if target.min_return_loss_db is not None:
        scale = target.min_return_loss_db
        violation_components["return_loss"] = _positive_error(errors.return_loss_error_db) / _scale(scale)
        tracking_components["return_loss"] = violation_components["return_loss"]

    if target.high_side_zero_freq_hz is not None:
        scale = target.high_side_zero_freq_hz.tolerance or target.high_side_zero_freq_hz.target
        violation_components["high_side_zero_freq"] = _symmetric_violation(
            errors.high_side_zero_freq_error_hz,
            target.high_side_zero_freq_hz.tolerance,
            scale,
        )
        tracking_components["high_side_zero_freq"] = abs(_normalize_error(errors.high_side_zero_freq_error_hz, scale))

    if target.min_high_side_zero_depth_db is not None:
        scale = target.min_high_side_zero_depth_db
        violation_components["high_side_zero_depth"] = _positive_error(errors.high_side_zero_depth_error_db) / _scale(scale)
        tracking_components["high_side_zero_depth"] = violation_components["high_side_zero_depth"]

    return violation_components, tracking_components


def summarize_objective(result: Any, reward_config: RewardConfig | None = None) -> ObjectiveSummary:
    config = reward_config or RewardConfig()
    violation_components, tracking_components = _objective_component_maps(result)
    constraint_violation = sum(violation_components.values())
    tracking_error = sum(tracking_components.values())
    total_loss = (
        config.constraint_violation_weight * constraint_violation
        + config.tracking_error_weight * tracking_error
    )
    return ObjectiveSummary(
        violation_components=violation_components,
        tracking_components=tracking_components,
        constraint_violation=constraint_violation,
        tracking_error=tracking_error,
        total_loss=total_loss,
        target_satisfied=bool(result.analysis.target_satisfied),
    )


def build_observation(result: Any) -> tuple[float, ...]:
    manifest = result.manifest
    analysis = result.analysis
    target = _target_from_analysis(analysis)

    variable_vector = [
        _normalize_variable(float(result.variable_values[variable.name]), variable) for variable in manifest.variables
    ]

    metric_vector = [
        _normalize_metric(
            analysis.metrics.main_peak_freq_hz,
            1.0 if target is None or target.center_freq_hz is None else target.center_freq_hz.target,
        ),
        _normalize_metric(
            analysis.metrics.bandwidth_3db_hz,
            1.0 if target is None or target.bandwidth_hz is None else target.bandwidth_hz.target,
        ),
        _normalize_metric(
            _insertion_loss_db(analysis),
            1.0 if target is None or target.max_insertion_loss_db is None else target.max_insertion_loss_db,
        ),
        _normalize_metric(
            _return_loss_db(analysis),
            1.0 if target is None or target.min_return_loss_db is None else target.min_return_loss_db,
        ),
        _normalize_metric(
            analysis.metrics.high_side_zero_freq_hz,
            1.0 if target is None or target.high_side_zero_freq_hz is None else target.high_side_zero_freq_hz.target,
        ),
        _normalize_metric(
            _high_side_zero_depth_db(analysis),
            1.0 if target is None or target.min_high_side_zero_depth_db is None else target.min_high_side_zero_depth_db,
        ),
    ]

    error_vector = _normalized_error_components(result)
    return tuple(variable_vector + metric_vector + error_vector)


def _error_loss(result: Any) -> float:
    return summarize_objective(result).total_loss


def compute_reward(previous_result: Any, current_result: Any, reward_config: RewardConfig | None = None) -> float:
    config = reward_config or RewardConfig()
    previous_loss = summarize_objective(previous_result, config).total_loss
    current_loss = summarize_objective(current_result, config).total_loss
    improvement = previous_loss - current_loss

    step_magnitude = 0.0
    for variable in current_result.manifest.variables:
        previous_value = float(previous_result.variable_values[variable.name])
        current_value = float(current_result.variable_values[variable.name])
        delta = abs(current_value - previous_value)
        scale = variable.max_delta
        if scale is None:
            if variable.min_value is not None and variable.max_value is not None:
                scale = (variable.max_value - variable.min_value) / 10.0
            else:
                scale = 1.0
        step_magnitude += delta / _scale(scale)

    reward = (
        config.improvement_weight * improvement
        - config.absolute_error_weight * current_loss
        - config.action_penalty_weight * step_magnitude
    )
    if current_result.analysis.errors is None:
        reward -= config.analysis_failure_penalty
    if current_result.analysis.target_satisfied:
        reward += config.success_bonus
    return reward


class FilterTuningEnv:
    def __init__(
        self,
        manifest: str | Path | TuningManifest,
        workspace: str | Path,
        *,
        evaluate_fn: Callable[..., Any] = evaluate_tuning_step,
        runner: Any | None = None,
        server: str | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int = 20,
        seed: int | None = None,
        verbose: bool = False,
        timeout: int | None = None,
    ) -> None:
        self.manifest = _load_manifest(manifest)
        self.workspace = Path(workspace).resolve()
        self.evaluate_fn = evaluate_fn
        self.runner = runner
        self.server = server
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.verbose = verbose
        self.timeout = timeout
        self.rng = Random(seed)
        self.current_result: Any | None = None
        self.episode_index = 0
        self.step_index = 0
        self.runtime_samples: list[float] = []

        missing_step = [variable.name for variable in self.manifest.variables if variable.max_delta is None]
        if missing_step:
            raise ValueError(f"FilterTuningEnv requires max_delta for all variables: {', '.join(missing_step)}")

    @property
    def action_names(self) -> list[str]:
        return [variable.name for variable in self.manifest.variables]

    def _sample_initial_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for variable in self.manifest.variables:
            if variable.min_value is not None and variable.max_value is not None:
                values[variable.name] = self.rng.uniform(variable.min_value, variable.max_value)
            elif variable.default_value is not None:
                values[variable.name] = float(variable.default_value)
            else:
                raise ValueError(f"Unable to sample initial value for variable '{variable.name}'")
        return values

    def _episode_workspace(self) -> Path:
        return self.workspace / f"episode_{self.episode_index:04d}" / f"step_{self.step_index:04d}"

    def _evaluate(self, variable_values: dict[str, float]) -> Any:
        workspace = self._episode_workspace()
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            return self.evaluate_fn(
                self.manifest,
                variable_values,
                workspace,
                runner=self.runner,
                server=self.server,
                verbose=self.verbose,
                timeout=self.timeout,
                runtime_reference_seconds=build_runtime_reference(self.runtime_samples),
            )
        except TypeError:
            return self.evaluate_fn(self.manifest, variable_values, workspace)

    def _info(self, result: Any) -> dict[str, Any]:
        analysis = result.analysis
        objective = summarize_objective(result, self.reward_config)
        return {
            "episode_index": self.episode_index,
            "step_index": self.step_index,
            "variable_values": dict(result.variable_values),
            "metrics": analysis.metrics.to_dict(),
            "errors": None if analysis.errors is None else analysis.errors.to_dict(),
            "target_satisfied": analysis.target_satisfied,
            "objective": objective.to_dict(),
            "observation": list(build_observation(result)),
            "elapsed_seconds": result.run.execution.elapsed_seconds,
            "runtime_report": None if getattr(result, "runtime_report", None) is None else result.runtime_report.to_dict(),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[float, ...], dict[str, Any]]:
        if seed is not None:
            self.rng = Random(seed)

        self.episode_index += 1
        self.step_index = 0
        self.runtime_samples = []
        options = options or {}
        if "variable_values" in options:
            initial_values = self.manifest.resolve_values(options["variable_values"])
        elif options.get("randomize"):
            initial_values = self.manifest.resolve_values(self._sample_initial_values())
        else:
            initial_values = self.manifest.resolve_values()

        self.current_result = self._evaluate(initial_values)
        self.runtime_samples.append(float(self.current_result.run.execution.elapsed_seconds))
        return build_observation(self.current_result), self._info(self.current_result)

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]:
        if self.current_result is None:
            raise RuntimeError("FilterTuningEnv.reset() must be called before step()")

        if len(action) != len(self.manifest.variables):
            raise ValueError(f"Expected action dimension {len(self.manifest.variables)}, got {len(action)}")

        current_values = dict(self.current_result.variable_values)
        updated_values: dict[str, float] = {}
        for variable, raw_component in zip(self.manifest.variables, action):
            component = max(-1.0, min(1.0, float(raw_component)))
            candidate = float(current_values[variable.name]) + component * float(variable.max_delta)
            if variable.min_value is not None:
                candidate = max(candidate, float(variable.min_value))
            if variable.max_value is not None:
                candidate = min(candidate, float(variable.max_value))
            updated_values[variable.name] = candidate

        previous_result = self.current_result
        self.step_index += 1
        self.current_result = self._evaluate(updated_values)
        self.runtime_samples.append(float(self.current_result.run.execution.elapsed_seconds))

        reward = compute_reward(previous_result, self.current_result, self.reward_config)
        terminated = bool(self.current_result.analysis.target_satisfied)
        truncated = self.step_index >= self.max_steps and not terminated
        info = self._info(self.current_result)
        info["reward"] = reward
        return build_observation(self.current_result), reward, terminated, truncated, info

    def close(self) -> None:
        self.current_result = None
