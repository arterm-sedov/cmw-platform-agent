#!/usr/bin/env python3
"""
Master Script: Run All Steps Sequentially

Runs Steps 1-5 in sequence with resume support.
Can be interrupted and resumed.

Usage:
    python tool_analyze_all.py --app Volga --extract-dir /path/to/extract --output-dir /path/to/output
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent


def run_step(script_name: str, args: list, cwd: Path = None) -> int:
    """Run a step script and return exit code."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)] + args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd[:3])} ... {' '.join(cmd[3:])}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=cwd or APP_DIR)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all analysis steps sequentially")
    parser.add_argument("--app", required=True)
    parser.add_argument("--extract-dir", default=None)
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    extract_dir = Path(args.extract_dir or f"/tmp/cmw-transfer")
    output_dir = Path(args.output_dir or f"/tmp/cmw-transfer/{args.app}_tr")
    app_dir = extract_dir / args.app
    os.makedirs(output_dir, exist_ok=True)

    state_file = output_dir / f"{args.app}_master_state.json"
    state = {"current_step": 0, "completed_steps": [], "last_run": ""}

    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    print(f"=== Master Script: Full Analysis for {args.app} ===")
    print(f"Extract dir: {extract_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Last state: step {state.get('current_step')}, completed: {state.get('completed_steps', [])}")

    start_time = time.time()
    current_step = state.get("current_step", 0)

    steps = [
        ("tool_extract_aliases.py", [
            "--app", args.app,
            "--extract-dir", str(extract_dir),
            "--output-dir", str(output_dir),
        ]),
        ("tool_collect_platform.py", [
            "--app", args.app,
            "--output-dir", str(output_dir),
        ]),
        ("tool_verify_aliases.py", None),  # Special handling - run per folder
        ("tool_find_dangerous.py", [
            "--app", args.app,
            "--extract-dir", str(extract_dir),
            "--output-dir", str(output_dir),
        ]),
        ("tool_finalize.py", [
            "--app", args.app,
            "--output-dir", str(output_dir),
        ]),
    ]

    step_names = ["extract", "collect_platform", "verify", "find_dangerous", "finalize"]

    if current_step >= 1 and current_step < 2:
        print("\nStep 1 already completed, skipping...")

    if current_step >= 2 and current_step < 3:
        print("\nSteps 1-2 already completed, skipping to Step 3...")

    if current_step >= 3 and current_step < 4:
        print("\nSteps 1-3 already completed, skipping to Step 4...")

    if current_step >= 4 and current_step < 5:
        print("\nSteps 1-4 already completed, skipping to Step 5...")

    for i, (step_idx, step_name) in enumerate(zip(range(1, 6), step_names)):
        if i < current_step:
            continue

        print(f"\n{'#'*60}")
        print(f"# Step {step_idx}: {step_name}")
        print(f"{'#'*60}")

        if step_idx == 3:
            folders_file = output_dir / f"{args.app}_extraction_state.json"
            if folders_file.exists():
                with open(folders_file) as f:
                    folders_data = json.load(f)
                folders = folders_data.get("completed_folders", [])
            else:
                folders = []

            for folder in folders:
                folder_verified = output_dir / f"{args.app}_{folder}_verified.json"
                if folder_verified.exists():
                    try:
                        with open(folder_verified) as f:
                            data = json.load(f)
                        if data.get("verified"):
                            print(f"  {folder}: already verified")
                            continue
                    except (json.JSONDecodeError, OSError):
                        pass

                print(f"Verifying folder: {folder}")
                exit_code = run_step("tool_verify_aliases.py", [
                    "--app", args.app,
                    "--folder", folder,
                    "--output-dir", str(output_dir),
                ])
                if exit_code != 0:
                    print(f"Error verifying folder {folder}, retrying...")
                    time.sleep(2)
                    exit_code = run_step("tool_verify_aliases.py", [
                        "--app", args.app,
                        "--folder", folder,
                        "--output-dir", str(output_dir),
                    ])
                    if exit_code != 0:
                        print(f"Failed to verify folder {folder}, continuing...")

            state["current_step"] = 4
            state["completed_steps"] = step_names[:i]
            state["last_run"] = datetime.now().isoformat()
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        else:
            script_name, script_args = steps[i]
            if script_args is None:
                continue

            exit_code = run_step(script_name, script_args)

            if exit_code != 0:
                print(f"Error in step {step_idx}, retrying...")
                time.sleep(2)
                exit_code = run_step(script_name, script_args)

                if exit_code != 0:
                    print(f"Step {step_idx} failed after retry")
                    print(f"Run manually: python {script_name} {' '.join(script_args)}")
                    return 1

            state["current_step"] = step_idx
            state["completed_steps"] = step_names[:i+1]
            state["last_run"] = datetime.now().isoformat()
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

    elapsed = time.time() - start_time

    print(f"\n{'#'*60}")
    print(f"# All Steps Complete")
    print(f"{'#'*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
