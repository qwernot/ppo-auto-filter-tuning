from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from auto_sonnet import SonnetRunner, load_tuning_manifest, ppo_run_config_to_dict  # noqa: E402
from auto_sonnet.rl_training import GymFilterTuningEnv, load_ppo_run_config  # noqa: E402


def _import_rl_stack() -> tuple[Any, Any, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "PPO training dependencies are not installed.\n"
            "Install them first, for example:\n"
            "  pip install stable-baselines3 gymnasium torch\n"
            f"Original import error: {exc}"
        ) from exc
    return PPO, BaseCallback, Monitor


def _prepare_output_root(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _assert_manifest_ready(manifest_path: Path) -> None:
    manifest = load_tuning_manifest(manifest_path)
    if manifest.template_path.exists():
        return
    raise SystemExit(f"Template not found: {manifest.template_path}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_train_env(config: Any, runner: SonnetRunner, output_root: Path, *, server: str | None):
    return GymFilterTuningEnv(
        config.manifest_path,
        output_root / "train_env",
        runner=runner,
        server=server,
        reward_config=config.reward_config,
        max_steps=config.max_steps,
        reset_policy=config.reset_policy,
        seed=config.training.seed,
    )


def _build_eval_env(config: Any, runner: SonnetRunner, output_root: Path, *, server: str | None):
    return GymFilterTuningEnv(
        config.manifest_path,
        output_root / "eval_env",
        runner=runner,
        server=server,
        reward_config=config.reward_config,
        max_steps=config.max_steps,
        reset_policy=config.evaluation.reset_policy or config.reset_policy,
        seed=config.evaluation.seed,
    )


def _evaluate_policy(model: Any, env: Any, *, episodes: int, deterministic: bool, seed: int) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    episode_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []

    success_count = 0
    total_steps = 0
    total_reward = 0.0

    for episode_index in range(episodes):
        observation, info = env.reset(seed=seed + episode_index)
        terminated = False
        truncated = False
        episode_reward = 0.0
        step_count = 0

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            episode_reward += float(reward)

            row: dict[str, object] = {
                "episode_index": episode_index + 1,
                "step_index": step_count,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "target_satisfied": bool(info.get("target_satisfied")),
            }
            for index, value in enumerate(action, start=1):
                row[f"action_{index}"] = float(value)
            for name, value in info.get("variable_values", {}).items():
                row[f"var_{name}"] = value
            for name, value in info.get("metrics", {}).items():
                row[f"metric_{name}"] = value
            trajectory_rows.append(row)

        success = bool(info.get("target_satisfied"))
        if success:
            success_count += 1
        total_steps += step_count
        total_reward += episode_reward
        episode_rows.append(
            {
                "episode_index": episode_index + 1,
                "success": success,
                "steps": step_count,
                "episode_reward": episode_reward,
                "final_center_freq_hz": info.get("metrics", {}).get("main_peak_freq_hz"),
                "final_bandwidth_hz": info.get("metrics", {}).get("bandwidth_3db_hz"),
                "final_insertion_loss_db": None if info.get("metrics", {}).get("main_peak_s21_db") is None else -float(info["metrics"]["main_peak_s21_db"]),
                "final_return_loss_db": None if info.get("metrics", {}).get("best_s11_db") is None else -float(info["metrics"]["best_s11_db"]),
                "final_zero_freq_hz": info.get("metrics", {}).get("high_side_zero_freq_hz"),
            }
        )

    summary = {
        "episodes": episodes,
        "success_rate": success_count / episodes if episodes else 0.0,
        "mean_steps": total_steps / episodes if episodes else 0.0,
        "mean_episode_reward": total_reward / episodes if episodes else 0.0,
    }
    return summary, episode_rows, trajectory_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on the packaged fixheight low-cost 5-variable environment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "specs" / "filter_example_one_ppo_train_config_fixheight_v2.json",
        help="Path to the PPO run config JSON.",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional explicit output directory.")
    parser.add_argument(
        "--sonnet-dir",
        type=Path,
        default=Path(r"D:\Program Files\Sonnet Software\19.52.2025\bin"),
        help="Sonnet installation root or bin directory.",
    )
    parser.add_argument("--server", help="Optional Sonnet Remote EM server name or alias.")
    parser.add_argument("--timesteps", type=int, help="Override training.total_timesteps from the config.")
    parser.add_argument("--seed", type=int, help="Override training.seed from the config.")
    parser.add_argument("--device", help="Override training.device from the config.")
    args = parser.parse_args()

    PPO, BaseCallback, Monitor = _import_rl_stack()
    config = load_ppo_run_config(args.config)
    _assert_manifest_ready(config.manifest_path)

    if args.timesteps is not None:
        config.training.total_timesteps = int(args.timesteps)
    if args.seed is not None:
        config.training.seed = int(args.seed)
    if args.device is not None:
        config.training.device = str(args.device)

    output_root = _prepare_output_root(args.output_dir.resolve() if args.output_dir is not None else config.output_root.resolve())
    runner = SonnetRunner.from_discovery(sonnet_dir=args.sonnet_dir)

    class EpisodeStatsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.rows: list[dict[str, object]] = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                episode = info.get("episode")
                if episode is None:
                    continue
                self.rows.append(
                    {
                        "timesteps": self.num_timesteps,
                        "episode_reward": float(episode["r"]),
                        "episode_length": int(episode["l"]),
                    }
                )
            return True

    train_env = Monitor(_build_train_env(config, runner, output_root, server=args.server))
    eval_env = _build_eval_env(config, runner, output_root, server=args.server)
    callback = EpisodeStatsCallback()

    model = PPO(
        config.training.policy,
        train_env,
        verbose=1,
        seed=config.training.seed,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        ent_coef=config.training.ent_coef,
        clip_range=config.training.clip_range,
        vf_coef=config.training.vf_coef,
        max_grad_norm=config.training.max_grad_norm,
        device=config.training.device,
    )

    try:
        model.learn(total_timesteps=config.training.total_timesteps, callback=callback, progress_bar=False)
        model.save(output_root / "ppo_filter_example_one_fixheight_v2")

        eval_summary, eval_episode_rows, eval_trajectory_rows = _evaluate_policy(
            model,
            eval_env,
            episodes=config.evaluation.episodes,
            deterministic=config.evaluation.deterministic,
            seed=config.evaluation.seed,
        )

        _write_json(output_root / "run_config_used.json", ppo_run_config_to_dict(config))
        _write_json(
            output_root / "training_summary.json",
            {
                "name": config.name,
                "manifest_path": str(config.manifest_path),
                "output_root": str(output_root),
                "total_timesteps": config.training.total_timesteps,
                "episode_records": len(callback.rows),
            },
        )
        _write_csv(output_root / "training_curve.csv", callback.rows)
        _write_json(output_root / "evaluation_summary.json", eval_summary)
        _write_csv(output_root / "evaluation_episode_summary.csv", eval_episode_rows)
        _write_csv(output_root / "evaluation_trajectory.csv", eval_trajectory_rows)
    finally:
        train_env.close()
        eval_env.close()

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "model_path": str((output_root / "ppo_filter_example_one_fixheight_v2.zip").resolve()),
                "training_curve": str((output_root / "training_curve.csv").resolve()),
                "evaluation_summary": str((output_root / "evaluation_summary.json").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
