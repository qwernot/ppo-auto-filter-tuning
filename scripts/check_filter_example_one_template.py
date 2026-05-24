from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from auto_sonnet import load_tuning_manifest, read_project_variables  # noqa: E402


def check_template_variables(manifest_path: Path) -> dict[str, object]:
    manifest = load_tuning_manifest(manifest_path)
    if not manifest.template_path.exists():
        return {
            "template_path": str(manifest.template_path),
            "template_exists": False,
            "present_variables": [],
            "required_variables": sorted({name for variable in manifest.variables for name in variable.sonnet_names}),
            "missing_variables": sorted({name for variable in manifest.variables for name in variable.sonnet_names}),
            "ready": False,
        }

    present_variable_map = read_project_variables(manifest.template_path)
    required_variables = sorted({name for variable in manifest.variables for name in variable.sonnet_names})
    missing_variables = [name for name in required_variables if name not in present_variable_map]
    return {
        "template_path": str(manifest.template_path),
        "template_exists": True,
        "present_variables": sorted(present_variable_map.keys()),
        "required_variables": required_variables,
        "missing_variables": missing_variables,
        "ready": not missing_variables,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the packaged Example One template contains all required variables.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "specs" / "filter_example_one_tuning_manifest_lowcost_fixheight.json",
        help="Path to the tuning manifest JSON.",
    )
    args = parser.parse_args()

    result = check_template_variables(args.manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
