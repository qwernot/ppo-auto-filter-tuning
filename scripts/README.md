# Scripts

Command-line entry points for the packaged filter tuning workflow.

## Commands

Check template variables:

```powershell
python scripts/check_filter_example_one_template.py
```

Evaluate one tuning point:

```powershell
python scripts/evaluate_filter_example_one_step.py --adj-gap 1.10
```

Run a short training job:

```powershell
python scripts/train_filter_ppo.py --timesteps 32
```

Run with the default configuration:

```powershell
python scripts/train_filter_ppo.py
```

Use `--sonnet-dir` when Sonnet is installed outside the default path.
