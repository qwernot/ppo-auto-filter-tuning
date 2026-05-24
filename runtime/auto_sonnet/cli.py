from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import analyze_touchstone
from .discovery import discover_installation
from .filter_metrics import FilterMetricConfig
from .runner import SonnetRunner, run_macro
from .specs import load_automation_jobs, load_automation_spec
from .tuning_session import evaluate_tuning_step
from .workflow import SonnetAutomation


def _default_workspace(spec_path: Path) -> Path:
    return Path("build") / spec_path.stem


def _job_workspace(base_workspace: Path, job_key: str, total_jobs: int) -> Path:
    if total_jobs <= 1:
        return base_workspace
    return base_workspace / job_key


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Python automation framework for Sonnet 19 macros.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-install", help="Detect Sonnet and print runmacro information.")
    verify.add_argument("--sonnet-dir", type=Path, help="Sonnet installation root or bin directory.")
    verify.add_argument("--runmacro-path", type=Path, help="Explicit runmacro executable path.")

    generate = subparsers.add_parser("generate", help="Generate a macro file from a JSON/TOML spec.")
    generate.add_argument("--spec", type=Path, required=True, help="Path to the automation spec.")
    generate.add_argument("--workspace", type=Path, help="Output workspace directory.")
    generate.add_argument("--macro-name", default="sonnet_macro.smc", help="Generated macro filename.")
    generate.add_argument("--print-macro", action="store_true", help="Print the generated macro content.")

    analyze = subparsers.add_parser("analyze-touchstone", help="Analyze a Touchstone .s2p file and extract filter metrics.")
    analyze.add_argument("--s2p", type=Path, required=True, help="Path to the Touchstone .s2p file.")
    analyze.add_argument("--target", type=Path, help="Optional target JSON file for error evaluation.")
    analyze.add_argument("--output-json", type=Path, help="Optional path for writing analysis JSON.")
    analyze.add_argument("--peak-search-start-ghz", type=float, help="Optional passband peak search start frequency in GHz.")
    analyze.add_argument("--peak-search-stop-ghz", type=float, help="Optional passband peak search stop frequency in GHz.")
    analyze.add_argument("--high-side-search-start-ghz", type=float, help="Optional high-side zero search start frequency in GHz.")
    analyze.add_argument("--high-side-search-stop-ghz", type=float, help="Optional high-side zero search stop frequency in GHz.")

    evaluate = subparsers.add_parser(
        "evaluate-step",
        help="Evaluate a single tuning step from a manifest by editing a Sonnet template and running analysis.",
    )
    evaluate.add_argument("--manifest", type=Path, required=True, help="Path to the tuning manifest JSON file.")
    evaluate.add_argument("--workspace", type=Path, required=True, help="Workspace directory for generated evaluation artifacts.")
    evaluate.add_argument(
        "--set",
        dest="variable_assignments",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override one tuning variable. Repeat to set multiple values.",
    )
    evaluate.add_argument("--result-filename", default="step_result.json", help="JSON filename written inside the workspace.")
    evaluate.add_argument("--sonnet-dir", type=Path, help="Sonnet installation root or bin directory.")
    evaluate.add_argument("--runmacro-path", "--runmacro", dest="runmacro_path", type=Path, help="Explicit runmacro executable path.")
    evaluate.add_argument("--server", help="Optional Sonnet Remote EM server name or alias.")
    evaluate.add_argument("--timeout", type=int, help="Timeout passed to subprocess.run in seconds.")
    evaluate.add_argument("--verbose", action="store_true", help="Pass -v to runmacro.")

    run = subparsers.add_parser("run", help="Run a generated or existing Sonnet macro.")
    run_source = run.add_mutually_exclusive_group(required=True)
    run_source.add_argument("--spec", type=Path, help="Path to the automation spec.")
    run_source.add_argument("--macro", type=Path, help="Run an existing macro file directly.")
    run.add_argument("--workspace", type=Path, help="Output workspace directory for generated artifacts.")
    run.add_argument("--macro-name", default="sonnet_macro.smc", help="Generated macro filename.")
    run.add_argument("--sonnet-dir", type=Path, help="Sonnet installation root or bin directory.")
    run.add_argument("--runmacro-path", "--runmacro", dest="runmacro_path", type=Path, help="Explicit runmacro executable path.")
    run.add_argument("--timeout", type=int, help="Timeout passed to subprocess.run in seconds.")
    run.add_argument("--verbose", action="store_true", help="Pass -v to runmacro.")
    run.add_argument("--print-macro", action="store_true", help="Print the generated macro content.")
    run.add_argument("--analyze-touchstone", action="store_true", help="Analyze the first generated .s2p file after a successful spec-based run.")
    run.add_argument("--target", type=Path, help="Optional target JSON file used during Touchstone analysis.")
    run.add_argument("--output-json", type=Path, help="Optional path for writing Touchstone analysis JSON.")
    run.add_argument("--peak-search-start-ghz", type=float, help="Optional passband peak search start frequency in GHz.")
    run.add_argument("--peak-search-stop-ghz", type=float, help="Optional passband peak search stop frequency in GHz.")
    run.add_argument("--high-side-search-start-ghz", type=float, help="Optional high-side zero search start frequency in GHz.")
    run.add_argument("--high-side-search-stop-ghz", type=float, help="Optional high-side zero search stop frequency in GHz.")
    return parser


def _analysis_config_from_args(args: argparse.Namespace) -> FilterMetricConfig:
    return FilterMetricConfig(
        peak_search_start_hz=None if args.peak_search_start_ghz is None else args.peak_search_start_ghz * 1e9,
        peak_search_stop_hz=None if args.peak_search_stop_ghz is None else args.peak_search_stop_ghz * 1e9,
        high_side_search_start_hz=None
        if args.high_side_search_start_ghz is None
        else args.high_side_search_start_ghz * 1e9,
        high_side_search_stop_hz=None
        if args.high_side_search_stop_ghz is None
        else args.high_side_search_stop_ghz * 1e9,
    )


def _parse_variable_assignments(assignments: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_assignment in assignments:
        if "=" not in raw_assignment:
            raise ValueError(f"Invalid variable assignment '{raw_assignment}'. Expected NAME=VALUE.")
        name, raw_value = raw_assignment.split("=", 1)
        key = name.strip()
        if not key:
            raise ValueError(f"Invalid variable assignment '{raw_assignment}'. Variable name is empty.")
        try:
            values[key] = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in assignment '{raw_assignment}'.") from exc
    return values


def _cmd_verify_install(args: argparse.Namespace) -> int:
    installation = discover_installation(sonnet_dir=args.sonnet_dir, runmacro_path=args.runmacro_path)
    runner = SonnetRunner(installation=installation)
    print(f"Detected Sonnet directory: {installation.sonnet_dir}")
    print(f"Detected runmacro path: {installation.runmacro_path}")
    print(f"Discovery source: {installation.source}")
    if installation.version:
        print(f"Registry version: {installation.version}")
    print(f"runmacro version: {runner.version()}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    spec_path = args.spec.resolve()
    workspace = (args.workspace or _default_workspace(spec_path)).resolve()
    jobs = load_automation_jobs(spec_path)
    automation = SonnetAutomation()
    for index, job in enumerate(jobs, start=1):
        job_workspace = _job_workspace(workspace, job.key, len(jobs))
        artifacts = automation.generate(job.spec, job_workspace, macro_name=args.macro_name)
        if len(jobs) > 1:
            print(f"\n[{index}/{len(jobs)}] Job: {job.key}")
        print(f"Workspace: {artifacts.workspace}")
        print(f"Macro file: {artifacts.macro_path}")
        print(f"Project target: {artifacts.project_path}")
        if args.print_macro:
            print("\n--- macro ---")
            print(artifacts.macro_path.read_text(encoding="utf-8"))
    return 0


def _cmd_analyze_touchstone(args: argparse.Namespace) -> int:
    analysis = analyze_touchstone(
        args.s2p,
        target_path=args.target,
        config=_analysis_config_from_args(args),
        output_path=args.output_json,
    )
    print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_evaluate_step(args: argparse.Namespace) -> int:
    variable_values = _parse_variable_assignments(args.variable_assignments)
    runner = SonnetRunner.from_discovery(sonnet_dir=args.sonnet_dir, runmacro_path=args.runmacro_path)
    result = evaluate_tuning_step(
        args.manifest,
        variable_values,
        args.workspace,
        runner=runner,
        server=args.server,
        result_filename=args.result_filename,
        verbose=args.verbose,
        timeout=args.timeout,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return result.run.execution.returncode


def _cmd_run(args: argparse.Namespace) -> int:
    if args.macro is not None:
        completed = run_macro(
            args.macro,
            sonnet_dir=args.sonnet_dir,
            runmacro_path=args.runmacro_path,
            cwd=args.macro.resolve().parent,
            verbose=args.verbose,
            timeout=args.timeout,
        )
        if completed.stdout.strip():
            print(completed.stdout.rstrip())
        if completed.stderr.strip():
            print(completed.stderr.rstrip(), file=sys.stderr)
        return completed.returncode

    spec_path = args.spec.resolve()
    workspace = (args.workspace or _default_workspace(spec_path)).resolve()
    jobs = load_automation_jobs(spec_path)
    runner = SonnetRunner.from_discovery(sonnet_dir=args.sonnet_dir, runmacro_path=args.runmacro_path)
    automation = SonnetAutomation(runner=runner)
    overall_returncode = 0
    should_analyze = args.analyze_touchstone or args.target is not None or args.output_json is not None
    analysis_config = _analysis_config_from_args(args)
    for index, job in enumerate(jobs, start=1):
        job_workspace = _job_workspace(workspace, job.key, len(jobs))
        if should_analyze:
            if args.output_json is None:
                analysis_filename = "metrics.json"
            elif len(jobs) == 1:
                analysis_filename = str(args.output_json)
            else:
                analysis_filename = str((args.output_json.parent / f"{job.key}-{args.output_json.name}").resolve())
            analyzed = automation.run_and_analyze(
                job.spec,
                job_workspace,
                macro_name=args.macro_name,
                target_path=args.target,
                config=analysis_config,
                analysis_filename=analysis_filename,
                verbose=args.verbose,
                timeout=args.timeout,
            )
            artifacts = analyzed.run
            analysis = analyzed.analysis
        else:
            artifacts = automation.run(
                job.spec,
                job_workspace,
                macro_name=args.macro_name,
                verbose=args.verbose,
                timeout=args.timeout,
            )
            analysis = None
        if len(jobs) > 1:
            print(f"\n[{index}/{len(jobs)}] Job: {job.key}")
        print(f"Workspace: {artifacts.workspace}")
        print(f"Macro file: {artifacts.macro_path}")
        print(f"Project target: {artifacts.project_path}")
        print(f"Return code: {artifacts.execution.returncode}")
        if args.print_macro:
            print("\n--- macro ---")
            print(artifacts.macro_path.read_text(encoding="utf-8"))
        if artifacts.execution.stdout.strip():
            print("\n--- stdout ---")
            print(artifacts.execution.stdout.rstrip())
        if artifacts.execution.stderr.strip():
            print("\n--- stderr ---", file=sys.stderr)
            print(artifacts.execution.stderr.rstrip(), file=sys.stderr)
        if analysis is not None:
            print("\n--- analysis ---")
            print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False))
        if artifacts.execution.returncode != 0 and overall_returncode == 0:
            overall_returncode = artifacts.execution.returncode
    return overall_returncode


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify-install":
        return _cmd_verify_install(args)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "analyze-touchstone":
        return _cmd_analyze_touchstone(args)
    if args.command == "evaluate-step":
        return _cmd_evaluate_step(args)
    if args.command == "run":
        return _cmd_run(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2
