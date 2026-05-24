from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(slots=True)
class RolloutStepRecord:
    step_index: int
    action: tuple[float, ...]
    observation: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action": list(self.action),
            "observation": list(self.observation),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }


@dataclass(slots=True)
class RolloutEpisodeRecord:
    action_names: tuple[str, ...]
    initial_observation: tuple[float, ...]
    initial_info: dict[str, Any]
    steps: list[RolloutStepRecord]

    @property
    def total_reward(self) -> float:
        return sum(step.reward for step in self.steps)

    @property
    def terminated(self) -> bool:
        return bool(self.steps and self.steps[-1].terminated)

    @property
    def truncated(self) -> bool:
        return bool(self.steps and self.steps[-1].truncated)

    @property
    def final_info(self) -> dict[str, Any]:
        if self.steps:
            return self.steps[-1].info
        return self.initial_info

    @property
    def success(self) -> bool:
        return bool(self.final_info.get("target_satisfied"))

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": self.success,
            "final_info": self.final_info,
        }

    def trajectory_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for step in self.steps:
            row: dict[str, Any] = {
                "step_index": step.step_index,
                "reward": step.reward,
                "terminated": step.terminated,
                "truncated": step.truncated,
            }
            for name, value in zip(self.action_names, step.action):
                row[f"action_{name}"] = value

            info = step.info
            for name, value in info.get("variable_values", {}).items():
                row[f"var_{name}"] = value
            for name, value in info.get("metrics", {}).items():
                row[f"metric_{name}"] = value

            errors = info.get("errors") or {}
            for name, value in errors.items():
                row[f"error_{name}"] = value

            row["target_satisfied"] = info.get("target_satisfied")
            rows.append(row)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_names": list(self.action_names),
            "initial_observation": list(self.initial_observation),
            "initial_info": self.initial_info,
            "steps": [step.to_dict() for step in self.steps],
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": self.success,
            "final_info": self.final_info,
        }


def rollout_filter_env(
    env: Any,
    actions: Sequence[Sequence[float]],
    *,
    reset_options: dict[str, Any] | None = None,
    seed: int | None = None,
) -> RolloutEpisodeRecord:
    initial_observation, initial_info = env.reset(seed=seed, options=reset_options)
    steps: list[RolloutStepRecord] = []

    for step_index, action in enumerate(actions, start=1):
        observation, reward, terminated, truncated, info = env.step(action)
        steps.append(
            RolloutStepRecord(
                step_index=step_index,
                action=tuple(float(value) for value in action),
                observation=tuple(float(value) for value in observation),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
            )
        )
        if terminated or truncated:
            break

    return RolloutEpisodeRecord(
        action_names=tuple(getattr(env, "action_names", [])),
        initial_observation=tuple(float(value) for value in initial_observation),
        initial_info=initial_info,
        steps=steps,
    )
