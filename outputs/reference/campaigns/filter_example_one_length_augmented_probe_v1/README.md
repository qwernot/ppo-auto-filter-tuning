# Length-Augmented Probe Campaign

## Purpose

This campaign is the first focused 2D probe after adding the two length variables:

- `middle_resonator_height`
- `outer_resonator_top_feed_clearance`

All other variables stay fixed at the current anchor point.

## Worker-machine run command

From the repository root:

```powershell
python projects/ppo_cross_coupled_filter/scripts/run_filter_example_one_bulk_campaign.py --campaign-dir "projects/ppo_cross_coupled_filter/offload_campaigns/filter_example_one_length_augmented_probe_v1"
```

If Sonnet is not discovered automatically:

```powershell
python projects/ppo_cross_coupled_filter/scripts/run_filter_example_one_bulk_campaign.py --campaign-dir "projects/ppo_cross_coupled_filter/offload_campaigns/filter_example_one_length_augmented_probe_v1" --sonnet-dir "C:\Program Files\Sonnet Software\19.52.2025\bin"
```

## Current task size

- samples: `30`
- varying axes:
  - `middle_resonator_height`: `[11.6, 11.8, 12.0, 12.2, 12.4]`
  - `outer_resonator_top_feed_clearance`: `[16.8, 17.0, 17.2, 17.4, 17.6, 17.8]`

## Fixed anchor

```json
{
  "outer_resonator_width": 12.0,
  "middle_resonator_width": 23.0,
  "middle_resonator_height": 12.0,
  "outer_resonator_top_feed_clearance": 17.4,
  "adjacent_coupling_gap": 1.2,
  "cross_coupling_gap": 1.8,
  "feed_offset": 4.7,
  "feed_width": 1.1,
  "middle_open_gap": 1.1,
  "outer_open_gap": 1.0
}
```
