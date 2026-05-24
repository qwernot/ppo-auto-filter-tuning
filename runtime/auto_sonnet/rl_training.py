from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

import numpy as np

from .rl_env import FilterTuningEnv, RewardConfig
from .tuning_manifest import TuningManifest, load_tuning_manifest

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None
    spaces = None


def _resolve_optional_path(raw: str | None, *, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base_dir / expanded).resolve()


def _load_manifest(manifest: str | Path | TuningManifest) -> TuningManifest:
    if isinstance(manifest, TuningManifest):
        return manifest
    return load_tuning_manifest(manifest)


@dataclass(slots=True)
class ResetJitterSpec:
    minus: float
    plus: float


@dataclass(slots=True)
class ResetPolicySpec:
    mode: str = "manifest_default"
    anchor_values: dict[str, float] = field(default_factory=dict)
    jitter: dict[str, ResetJitterSpec] = field(default_factory=dict)


@dataclass(slots=True)
class PpoAlgorithmConfig:
    total_timesteps: int = 128
    policy: str = "MlpPolicy"
    n_steps: int = 32
    batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
    device: str = "auto"


@dataclass(slots=True)
class PpoEvaluationConfig:
    episodes: int = 4
    deterministic: bool = True
    seed: int = 123
    reset_policy: ResetPolicySpec | None = None


@dataclass(slots=True)
class PpoRunConfig:
    name: str
    manifest_path: Path
    output_root: Path
    max_steps: int = 8
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    reset_policy: ResetPolicySpec = field(default_factory=ResetPolicySpec)
    training: PpoAlgorithmConfig = field(default_factory=PpoAlgorithmConfig)
    evaluation: PpoEvaluationConfig = field(default_factory=PpoEvaluationConfig)
    source_path: Path | None = None


def ppo_run_config_to_dict(config: PpoRunConfig) -> dict[str, Any]:
    def _reset_policy_to_dict(policy: ResetPolicySpec | None) -> dict[str, Any] | None:
        if policy is None:
            return None
        return {
            "mode": policy.mode,
            "anchor_values": dict(policy.anchor_values),
            "jitter": {
                name: {"minus": spec.minus, "plus": spec.plus}
                for name, spec in policy.jitter.items()
            },
        }

    return {
        "name": config.name,
        "manifest_path": str(config.manifest_path),
        "output_root": str(config.output_root),
        "max_steps": config.max_steps,
        "reward_config": {
            "improvement_weight": config.reward_config.improvement_weight,
            "absolute_error_weight": config.reward_config.absolute_error_weight,
            "constraint_violation_weight": config.reward_config.constraint_violation_weight,
            "tracking_error_weight": config.reward_config.tracking_error_weight,
            "action_penalty_weight": config.reward_config.action_penalty_weight,
            "success_bonus": config.reward_config.success_bonus,
            "analysis_failure_penalty": config.reward_config.analysis_failure_penalty,
        },
        "reset_policy": _reset_policy_to_dict(config.reset_policy),
        "training": {
            "total_timesteps": config.training.total_timesteps,
            "policy": config.training.policy,
            "n_steps": config.training.n_steps,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "gamma": config.training.gamma,
            "gae_lambda": config.training.gae_lambda,
            "ent_coef": config.training.ent_coef,
            "clip_range": config.training.clip_range,
            "vf_coef": config.training.vf_coef,
            "max_grad_norm": config.training.max_grad_norm,
            "seed": config.training.seed,
            "device": config.training.device,
        },
        "evaluation": {
            "episodes": config.evaluation.episodes,
            "deterministic": config.evaluation.deterministic,
            "seed": config.evaluation.seed,
            "reset_policy": _reset_policy_to_dict(config.evaluation.reset_policy),
        },
    }


def _parse_reset_policy(raw: dict[str, Any] | None) -> ResetPolicySpec:
    raw = raw or {}
    jitter_raw = raw.get("jitter") or {}
    jitter: dict[str, ResetJitterSpec] = {}
    for name, entry in jitter_raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Reset jitter for '{name}' must be an object")
        jitter[str(name)] = ResetJitterSpec(
            minus=float(entry.get("minus", 0.0)),
            plus=float(entry.get("plus", 0.0)),
        )
    return ResetPolicySpec(
        mode=str(raw.get("mode", "manifest_default")),
        anchor_values={str(name): float(value) for name, value in (raw.get("anchor_values") or {}).items()},
        jitter=jitter,
    )


def load_ppo_run_config(path: str | Path) -> PpoRunConfig:
    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("PPO run config must contain a top-level JSON object")
    base_dir = config_path.parent

    reward_raw = data.get("reward_config") or {}
    training_raw = data.get("training") or {}
    evaluation_raw = data.get("evaluation") or {}

    manifest_path = _resolve_optional_path(data.get("manifest_path"), base_dir=base_dir)
    output_root = _resolve_optional_path(data.get("output_root"), base_dir=base_dir)
    if manifest_path is None:
        raise ValueError("PPO run config requires manifest_path")
    if output_root is None:
        raise ValueError("PPO run config requires output_root")

    return PpoRunConfig(
        name=str(data.get("name", config_path.stem)),
        manifest_path=manifest_path,
        output_root=output_root,
        max_steps=int(data.get("max_steps", 8)),
        reward_config=RewardConfig(
            improvement_weight=float(reward_raw.get("improvement_weight", 1.0)),
            absolute_error_weight=float(reward_raw.get("absolute_error_weight", 0.2)),
            constraint_violation_weight=float(reward_raw.get("constraint_violation_weight", 1.0)),
            tracking_error_weight=float(reward_raw.get("tracking_error_weight", 0.25)),
            action_penalty_weight=float(reward_raw.get("action_penalty_weight", 0.05)),
            success_bonus=float(reward_raw.get("success_bonus", 10.0)),
            analysis_failure_penalty=float(reward_raw.get("analysis_failure_penalty", 5.0)),
        ),
        reset_policy=_parse_reset_policy(data.get("reset_policy")),
        training=PpoAlgorithmConfig(
            total_timesteps=int(training_raw.get("total_timesteps", 128)),
            policy=str(training_raw.get("policy", "MlpPolicy")),
            n_steps=int(training_raw.get("n_steps", 32)),
            batch_size=int(training_raw.get("batch_size", 32)),
            learning_rate=float(training_raw.get("learning_rate", 3e-4)),
            gamma=float(training_raw.get("gamma", 0.99)),
            gae_lambda=float(training_raw.get("gae_lambda", 0.95)),
            ent_coef=float(training_raw.get("ent_coef", 0.01)),
            clip_range=float(training_raw.get("clip_range", 0.2)),
            vf_coef=float(training_raw.get("vf_coef", 0.5)),
            max_grad_norm=float(training_raw.get("max_grad_norm", 0.5)),
            seed=int(training_raw.get("seed", 42)),
            device=str(training_raw.get("device", "auto")),
        ),
        evaluation=PpoEvaluationConfig(
            episodes=int(evaluation_raw.get("episodes", 4)),
            deterministic=bool(evaluation_raw.get("deterministic", True)),
            seed=int(evaluation_raw.get("seed", 123)),
            reset_policy=_parse_reset_policy(evaluation_raw.get("reset_policy"))
            if evaluation_raw.get("reset_policy") is not None
            else None,
        ),
        source_path=config_path,
    )


def sample_reset_values(
    manifest: str | Path | TuningManifest,
    policy: ResetPolicySpec | None,
    rng: Random,
) -> dict[str, float]:
    tuning_manifest = _load_manifest(manifest)
    policy = policy or ResetPolicySpec()
    mode = policy.mode.strip().lower()

    if mode == "manifest_default":
        return tuning_manifest.resolve_values()

    if mode == "random_uniform":
        sampled: dict[str, float] = {}
        for variable in tuning_manifest.variables:
            if variable.min_value is None or variable.max_value is None:
                if variable.default_value is None:
                    raise ValueError(f"Variable '{variable.name}' cannot be random-sampled without bounds or default value")
                sampled[variable.name] = float(variable.default_value)
            else:
                sampled[variable.name] = rng.uniform(float(variable.min_value), float(variable.max_value))
        return tuning_manifest.resolve_values(sampled)

    if mode == "anchor_jitter":
        base_values = tuning_manifest.resolve_values(policy.anchor_values or None)
        raw_values: dict[str, float] = {}
        for variable in tuning_manifest.variables:
            base_value = float(base_values[variable.name])
            jitter = policy.jitter.get(variable.name)
            if jitter is None:
                raw_values[variable.name] = base_value
                continue
            lower = base_value - float(jitter.minus)
            upper = base_value + float(jitter.plus)
            raw_values[variable.name] = rng.uniform(lower, upper)
        return tuning_manifest.resolve_values(raw_values)

    raise ValueError(f"Unsupported reset policy mode: {policy.mode}")


def build_reset_options(
    manifest: str | Path | TuningManifest,
    policy: ResetPolicySpec | None,
    rng: Random,
) -> dict[str, object]:
    return {"variable_values": sample_reset_values(manifest, policy, rng)}


class GymFilterTuningEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(
        self,
        manifest: str | Path | TuningManifest,
        workspace: str | Path,
        *,
        runner: Any | None = None,
        server: str | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int = 8,
        reset_policy: ResetPolicySpec | None = None,
        seed: int | None = None,
        verbose: bool = False,
        timeout: int | None = None,
    ) -> None:
        if gym is None or spaces is None:  # pragma: no cover - optional dependency
            raise ImportError("gymnasium is required to construct GymFilterTuningEnv")

        self.manifest = _load_manifest(manifest)
        self.reset_policy = reset_policy or ResetPolicySpec()
        self.rng = Random(seed)
        self.inner = FilterTuningEnv(
            self.manifest,
            workspace,
            runner=runner,
            server=server,
            reward_config=reward_config,
            max_steps=max_steps,
            seed=seed,
            verbose=verbose,
            timeout=timeout,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.manifest.variables),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.manifest.variables) + 12,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = Random(seed)
        effective_options = dict(options or {})
        if "variable_values" not in effective_options and not effective_options.get("randomize"):
            effective_options.update(build_reset_options(self.manifest, self.reset_policy, self.rng))
        observation, info = self.inner.reset(seed=seed, options=effective_options)
        return np.asarray(observation, dtype=np.float32), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.inner.step(action)
        return np.asarray(observation, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.inner.close()
