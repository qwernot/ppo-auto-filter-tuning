# PPO Cross-Coupled Filter Tuning

This repository contains the source code and configuration files for a PPO-based automatic tuning workflow for a cross-coupled microwave filter. It is intended as a clean code appendix: source files, runnable scripts, model templates, and JSON specifications are kept in the repository; generated simulation outputs are ignored.

## Repository Layout

- `runtime/auto_sonnet/`: reusable Python runtime for Sonnet project editing, simulation execution, Touchstone parsing, metric extraction, and reinforcement-learning environment logic.
- `scripts/`: command-line entry points for template checking, one-step evaluation, and PPO training.
- `specs/`: target definitions, tuning manifests, and PPO run configuration.
- `models/`: packaged Sonnet templates and reference run files required by the included manifests.
- `requirements.txt`: Python dependencies for training and evaluation.

## Environment

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The evaluation and training scripts require a local Sonnet installation. If Sonnet is not installed in the default location used by the scripts, pass `--sonnet-dir` explicitly.

## Basic Usage

Check that the packaged template contains the variables required by the tuning manifest:

```powershell
python scripts/check_filter_example_one_template.py
```

Evaluate one tuning point:

```powershell
python scripts/evaluate_filter_example_one_step.py --adj-gap 1.10
```

Run a short PPO training job:

```powershell
python scripts/train_filter_ppo.py --timesteps 32
```

Run with the default training configuration:

```powershell
python scripts/train_filter_ppo.py
```

Generated artifacts are written under `outputs/` by default and are excluded from version control.
