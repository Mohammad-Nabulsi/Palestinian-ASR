"""Command-line entry point: ``python -m pipeline``."""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from .config import load_config
from .context import RunContext, StageResult
from .stages import STAGES, get_stage


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Run the unified Palestinian-ASR data curation pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run the stages defined in a config.")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--run-root", type=Path, default=None,
                            help="Where reports/logs go. Defaults to the config's run_root var.")
    run_parser.add_argument("--only", default=None,
                            help="Comma-separated stage ids to run (in config order).")
    run_parser.add_argument("--skip", default=None, help="Comma-separated stage ids to skip.")
    run_parser.add_argument("--dry-run", action="store_true",
                            help="Resolve config and discover inputs without writing data.")
    run_parser.add_argument("--continue-on-error", action="store_true",
                            help="Keep going after a stage fails instead of aborting.")

    list_parser = sub.add_parser("list", help="List the stages a config defines.")
    list_parser.add_argument("--config", required=True, type=Path)

    sub.add_parser("stages", help="List available stage types.")

    return parser.parse_args(argv)


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_root = args.run_root or Path(config.vars.get("run_root", Path.cwd() / ".pipeline_runs" / config.name))
    ctx = RunContext(config, Path(run_root), dry_run=args.dry_run)

    only = args.only.split(",") if args.only else None
    skip = args.skip.split(",") if args.skip else None
    selected = config.select(only, skip)

    ctx.log(f"pipeline={config.name} config={config.source_path}")
    ctx.log(f"run_root={ctx.run_root} dry_run={args.dry_run}")
    ctx.log(f"stages: {[s.id for s in selected]}")

    failed = 0
    for spec in selected:
        ctx.bind_stage(spec.id)
        started = time.monotonic()
        ctx.log(f"--- start {spec.id} ({spec.stage}) ---")
        try:
            result = get_stage(spec.stage)(ctx, spec)
        except Exception as exc:
            failed += 1
            result = StageResult(stage_id=spec.id, stage=spec.stage, status="failed")
            result.notes.append(f"{type(exc).__name__}: {exc}")
            result.duration_sec = time.monotonic() - started
            ctx.log(f"FAILED {spec.id}: {type(exc).__name__}: {exc}")
            ctx.write_text(
                ctx.stage_reports_dir(spec.id) / "traceback.txt", traceback.format_exc()
            )
            ctx.record(result)
            if not args.continue_on_error:
                ctx.bind_stage(None)
                ctx.write_manifest()
                return 1
            continue

        result.duration_sec = time.monotonic() - started
        ctx.record(result)
        ctx.log(f"--- done {spec.id} in {result.duration_sec:.1f}s ---")

    ctx.bind_stage(None)
    manifest = ctx.write_manifest()
    ctx.log(f"manifest: {manifest}")
    return 1 if failed else 0


def cmd_list(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(f"{config.name}  ({config.source_path})")
    for spec in config.stages:
        mark = " " if spec.enabled else "-"
        print(f" {mark} {spec.id:<20} {spec.stage}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "stages":
        for name in sorted(STAGES):
            print(name)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
