# Bulk Offload Campaign

## Worker-machine run command

From the repository root:

```powershell
python projects/ppo_cross_coupled_filter/scripts/run_filter_example_one_bulk_campaign.py --campaign-dir "projects/ppo_cross_coupled_filter/offload_campaigns/filter_example_one_local_bulk_320_r1"
```

If auto-discovery does not find Sonnet on the worker machine, run:

```powershell
python projects/ppo_cross_coupled_filter/scripts/run_filter_example_one_bulk_campaign.py --campaign-dir "projects/ppo_cross_coupled_filter/offload_campaigns/filter_example_one_local_bulk_320_r1" --sonnet-dir "C:\Program Files\Sonnet Software\19.52.2025\bin"
```

## Current task size

- samples: `320`
- local best anchor before campaign:

```json
{
  "outer_resonator_width": 12.0,
  "middle_resonator_width": 23.0,
  "adjacent_coupling_gap": 0.5,
  "cross_coupling_gap": 1.8,
  "feed_offset": 4.6
}
```

- runtime reference: `63.46` seconds / sample
- estimated unattended worker time: `5.64` hours for `320` samples


The runner is resumable. If it is interrupted, run the same command again.
