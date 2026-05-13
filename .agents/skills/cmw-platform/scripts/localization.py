#!/usr/bin/env python3
"""
Main localization workflow orchestrator.

Usage:
    python localization.py --app Volga --json-folder C:\\tmp\\cmw-transfer\\Volga_ctf --output-dir C:\\tmp\\cmw-transfer --create-tr
    python localization.py --app Volga --json-folder C:\\tmp\\cmw-transfer\\Volga_ctf --output-dir C:\\tmp\\cmw-transfer --update-path
    python localization.py --app Volga --json-folder C:\\tmp\\cmw-transfer\\Volga_ctf --output-dir C:\\tmp\\cmw-transfer --fix-expressions
    python localization.py --app Volga --json-folder C:\\tmp\\cmw-transfer\\Volga_ctf --output-dir C:\\tmp\\cmw-transfer --apply-display-names --path-mode original
    python localization.py --app Volga --json-folder C:\\tmp\\cmw-transfer\\Volga_ctf --output-dir C:\\tmp\\cmw-transfer --apply-expressions
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def run_script(script_name: str, args: list[str], description: str = "") -> int:
    """Run a helper script via subprocess."""
    script_path = SCRIPTS_DIR / script_name

    cmd = [sys.executable, str(script_path)] + args

    if description:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd[:4])} ...")
        print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Localization workflow for system names")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--json-folder", required=True, help="Path to folder with CTF JSON files")
    parser.add_argument("--output-dir", default=None, help="Path to save output files")
    parser.add_argument("--create-tr", action="store_true", help="Create _tr copy from original")
    parser.add_argument("--translate-one", metavar="ALIAS", help="Translate single alias")
    parser.add_argument("--resume", action="store_true", help="Resume from last translated alias")
    parser.add_argument("--fix-expressions", action="store_true", help="Fix _calc aliases in expressions")
    parser.add_argument("--apply-display-names", action="store_true", help="Apply displayNameRenamed to CTF JSON files")
    parser.add_argument("--apply-expressions", action="store_true", help="Apply expressionRenamed to CTF JSON files")
    parser.add_argument("--update-path", action="store_true", help="Regenerate jsonPathRenamed from aliasRenamed")
    parser.add_argument("--path-mode", default="renamed", choices=["original", "renamed"], help="Which path to use")
    parser.add_argument("--dangerous-suffix", default="_calc", help="Suffix for dangerous aliases")
    parser.add_argument("--safe-suffix", default="_sv", help="Suffix for safe aliases")

    args = parser.parse_args()

    output_dir = args.output_dir or args.json_folder

    if args.create_tr:
        exit_code = run_script(
            "create_tr.py",
            ["--app", args.app, "--output-dir", output_dir, "--dangerous-suffix", args.dangerous_suffix],
            "Create _tr copy"
        )
        if exit_code != 0:
            return exit_code

    if args.translate_one:
        exit_code = run_script(
            "translate_one.py",
            ["--app", args.app, "--output-dir", output_dir, "--alias", args.translate_one,
             "--dangerous-suffix", args.dangerous_suffix, "--safe-suffix", args.safe_suffix],
            "Translate single alias"
        )
        if exit_code != 0:
            return exit_code

    if args.resume:
        exit_code = run_script(
            "translate_one.py",
            ["--app", args.app, "--output-dir", output_dir, "--resume",
             "--dangerous-suffix", args.dangerous_suffix, "--safe-suffix", args.safe_suffix],
            "Resume translation"
        )
        if exit_code != 0:
            return exit_code

    if args.update_path:
        exit_code = run_script(
            "update_path.py",
            ["--app", args.app, "--output-dir", output_dir],
            "Update jsonPathRenamed"
        )
        if exit_code != 0:
            return exit_code

    if args.fix_expressions:
        exit_code = run_script(
            "fix_expressions.py",
            ["--app", args.app, "--output-dir", output_dir, "--dangerous-suffix", args.dangerous_suffix],
            "Fix _calc expressions"
        )
        if exit_code != 0:
            return exit_code

    if args.apply_display_names:
        exit_code = run_script(
            "apply_display_names.py",
            ["--app", args.app, "--json-folder", args.json_folder, "--output-dir", output_dir, "--path-mode", args.path_mode],
            "Apply display names"
        )
        if exit_code != 0:
            return exit_code

    if args.apply_expressions:
        exit_code = run_script(
            "apply_expressions.py",
            ["--app", args.app, "--json-folder", args.json_folder, "--output-dir", output_dir],
            "Apply expressions"
        )
        if exit_code != 0:
            return exit_code

    print("\nLocalization workflow complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())