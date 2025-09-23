#!/usr/bin/env python3
"""
Linting script for cmw-platform-agent
=====================================

Simple script to run Ruff linting and formatting with proper configuration.
"""

from pathlib import Path
import subprocess
import sys
from collections.abc import Iterable


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stderr)
        return False


def _git_list_files(args: list[str]) -> list[str]:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
        return [p.strip() for p in result.stdout.splitlines() if p.strip()]
    except Exception:
        return []


def _filter_python_files(paths: Iterable[str]) -> list[str]:
    return [p for p in paths if p.endswith(".py") and Path(p).exists()]


def _usage() -> None:
    print(
        """
Usage: python lint.py [--staged | --changed | --all] [FILES...]

Lints only the provided files or the changed ones by default.

Options:
  --staged   Lint staged Python files (git diff --cached)
  --changed  Lint files changed vs HEAD (default if no FILES given)
  --all      Lint the entire repository (not recommended for CI hooks)

Examples:
  python lint.py file_a.py dir/module_b.py
  python lint.py --staged
  python lint.py --changed
  python lint.py --all
        """.strip()
    )


def resolve_target_files(argv: list[str]) -> list[str]:
    mode = None
    files: list[str] = []

    for arg in argv:
        if arg in {"--staged", "--changed", "--all"}:
            mode = arg
        else:
            files.append(arg)

    # If explicit files provided, use them
    if files:
        return _filter_python_files(files)

    # No explicit files; decide by mode
    if mode == "--all":
        # Fall back to repo root; ruff will discover pyproject settings
        return ["."]

    if mode == "--staged":
        candidates = _git_list_files(
            ["diff", "--name-only", "--cached", "--diff-filter=ACMR"]
        )
        return _filter_python_files(candidates)

    # Default: changed vs HEAD (tracked modifications, additions, renames)
    candidates = _git_list_files(["diff", "--name-only", "HEAD", "--diff-filter=ACMR"])
    return _filter_python_files(candidates)


def main():
    """Main linting workflow."""
    print("🚀 Starting cmw-platform-agent linting workflow...")

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print(
            "❌ Error: pyproject.toml not found. Run this script from the project root."
        )
        sys.exit(1)

    targets = resolve_target_files(sys.argv[1:])

    if not targets:
        print("i No Python files to lint. Nothing to do.")
        sys.exit(0)

    # If repo-wide lint requested, keep original behavior
    is_repo_wide = targets == ["."]
    target_label = ", ".join(targets) if not is_repo_wide else "repository"

    # Run Ruff checks
    check_cmd = ["ruff", "check", "--fix", *targets]
    check_success = run_command(
        check_cmd, f"Ruff checks and auto-fixes for {target_label}"
    )

    # Run Ruff formatting
    format_cmd = ["ruff", "format", *targets]
    format_success = run_command(
        format_cmd, f"Ruff formatting for {target_label}"
    )

    # Summary
    if check_success and format_success:
        print("\n🎉 All linting tasks completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some linting tasks failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
