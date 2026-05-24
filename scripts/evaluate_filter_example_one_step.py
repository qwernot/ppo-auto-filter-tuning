from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from auto_sonnet import SonnetRunner, evaluate_tuning_step, load_tuning_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one Sonnet tuning step for the packaged Filter Example One project.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "specs" / "filter_example_one_tuning_manifest_lowcost_fixheight.json",
        help="Path to the tuning manifest JSON.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "manual_step_eval",
        help="Workspace directory for generated evaluation artifacts.",
    )
    parser.add_argument("--outer-width", type=float, help="Outer resonator pair width in mm.")
    parser.add_argument("--middle-width", type=float, help="Middle resonator width in mm.")
    parser.add_argument("--adj-gap", type=float, help="Adjacent coupling gap in mm.")
    parser.add_argument("--cross-gap", type=float, help="Cross-coupling gap in mm.")
    parser.add_argument("--feed-offset", type=float, help="Feed offset in mm.")
    parser.add_argument(
        "--sonnet-dir",
        type=Path,
        default=Path(r"D:\Program Files\Sonnet Software\19.52.2025\bin"),
        help="Sonnet installation root or bin directory.",
    )
    parser.add_argument("--server", help="Optional Sonnet Remote EM server name or alias.")
    parser.add_argument("--timeout", type=int, help="Optional timeout passed to runmacro in seconds.")
    parser.add_argument("--verbose", action="store_true", help="Pass -v to runmacro.")
    args = parser.parse_args()

    manifest = load_tuning_manifest(args.manifest)
    if not manifest.template_path.exists():
        raise SystemExit(f"Template not found: {manifest.template_path}")

    overrides = {
        key: value
        for key, value in {
            "outer_resonator_width": args.outer_width,
            "middle_resonator_width": args.middle_width,
            "adjacent_coupling_gap": args.adj_gap,
            "cross_coupling_gap": args.cross_gap,
            "feed_offset": args.feed_offset,
        }.items()
        if value is not None
    }

    runner = SonnetRunner.from_discovery(sonnet_dir=args.sonnet_dir)
    result = evaluate_tuning_step(
        manifest,
        overrides,
        args.workspace,
        runner=runner,
        server=args.server,
        verbose=args.verbose,
        timeout=args.timeout,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
