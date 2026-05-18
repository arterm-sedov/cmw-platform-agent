# tool_localize.py - Localization workflow for system names (aliases) and display names
# Delegates to localization.py script via subprocess
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field
except ImportError:
    from tools.models import BaseModel, Field
    from tools.tool_utils import tool

SCRIPTS_DIR = Path(__file__).parent.parent.parent / ".agents/skills/cmw-platform/scripts"


class LocalizeSchema(BaseModel):
    """Schema for localization tool arguments."""
    application_system_name: str = Field(description="Application system name")
    json_folder: str = Field(description="Path to folder with CTF JSON files")
    output_dir: str | None = Field(default=None, description="Path to save output files")
    create_tr: bool = Field(default=False, description="Create _tr copy from original")
    translate_one: str | None = Field(default=None, description="Translate single alias")
    resume: bool = Field(default=False, description="Resume from last translated alias")
    apply_renames: bool = Field(default=False, description="Apply renames to platform")
    fix_expressions: bool = Field(default=False, description="Fix _calc aliases in expressions")
    apply_display_names: bool = Field(default=False, description="Apply displayNameRenamed to CTF JSON files")
    apply_expressions: bool = Field(default=False, description="Apply expressionRenamed to CTF JSON files")
    update_paths: bool = Field(default=False, description="Regenerate jsonPathRenamed from aliasRenamed")
    path_mode: str = Field(default="renamed", description="Which path to use: 'original' or 'renamed'")
    dry_run: bool = Field(
        default=True,
        description="If True, skip the scripted extract→finalize pipeline (steps 1–6). Set False to run it.",
    )
    dangerous_suffix: str = Field(default="_calc", description="Suffix for dangerous system names")
    safe_suffix: str = Field(default="_sv", description="Suffix for safe system names")
    rename_mode: str = Field(default="safe", description="Rename mode: 'all', 'safe' (no expressions), 'danger' (has expressions)")


@tool("localize_aliases", return_direct=False, args_schema=LocalizeSchema)
def localize_aliases(
    application_system_name: str = "",
    json_folder: str = "",
    output_dir: str | None = None,
    create_tr: bool = False,
    translate_one: str | None = None,
    resume: bool = False,
    apply_renames: bool = False,
    fix_expressions: bool = False,
    apply_display_names: bool = False,
    apply_expressions: bool = False,
    update_paths: bool = False,
    path_mode: str = "renamed",
    dry_run: bool = True,
    dangerous_suffix: str = "_calc",
    safe_suffix: str = "_sv",
    rename_mode: str = "safe",
) -> dict[str, any]:
    """
    Localization workflow for system names (aliases) and display names.

    Delegates to localization.py script which orchestrates helper scripts:
    - create_tr.py: Create _tr copy from original
    - translate_one.py: Translate single alias interactively
    - update_path.py: Regenerate jsonPathRenamed
    - fix_expressions.py: Fix _calc aliases in expressions
    - apply_display_names.py: Apply displayNameRenamed to CTF
    - apply_expressions.py: Apply expressionRenamed to CTF
    - apply_renames.py: Apply renames to platform (via direct import)
    """
    results = {
        "success": True,
        "phase": "started",
        "actions": [],
        "errors": [],
    }

    output_dir = output_dir or json_folder

    localization_script = SCRIPTS_DIR / "localization.py"

    cmd = [
        sys.executable,
        str(localization_script),
        "--app", application_system_name,
        "--json-folder", json_folder,
        "--output-dir", output_dir,
    ]

    if create_tr:
        cmd.append("--create-tr")
    if translate_one:
        cmd.extend(["--translate-one", translate_one])
    if resume:
        cmd.append("--resume")
    if fix_expressions:
        cmd.append("--fix-expressions")
    if apply_display_names:
        cmd.append("--apply-display-names")
    if apply_expressions:
        cmd.append("--apply-expressions")
    if update_paths:
        cmd.append("--update-path")

    cmd.extend(["--path-mode", path_mode])
    cmd.extend(["--dangerous-suffix", dangerous_suffix])
    cmd.extend(["--safe-suffix", safe_suffix])

    if apply_renames:
        results["phase"] = "apply_renames"

        old_argv = sys.argv
        sys.argv = ["apply_renames.py", "--app", application_system_name, "--output-dir", output_dir, "--mode", rename_mode]

        try:
            script_path = SCRIPTS_DIR / "apply_renames.py"
            import importlib.util
            spec = importlib.util.spec_from_file_location("apply_renames", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            exit_code = module.main(application_system_name, output_dir, mode=rename_mode)
            if exit_code != 0:
                results["success"] = False
                results["errors"].append(f"apply_renames.py failed (exit {exit_code})")
            else:
                results["actions"].append("Applied renames to platform")
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
        finally:
            sys.argv = old_argv

        results["phase"] = "complete"
        return results

    results["phase"] = "localization"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            results["success"] = False
            results["errors"].append(f"localization.py failed: {result.stderr}")
        else:
            results["actions"].append("Localization workflow completed")
            results["phase"] = "complete"
    except Exception as e:
        results["success"] = False
        results["errors"].append(str(e))
        results["phase"] = "error"

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Localization workflow for system names (schema.json format)")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--json-folder", required=True, help="Path to folder with JSON files")
    parser.add_argument("--output-dir", default=None, help="Path to save output files")
    parser.add_argument("--create-tr", action="store_true", help="Create _tr copy from original")
    parser.add_argument("--translate-one", metavar="ALIAS", help="Translate single alias")
    parser.add_argument("--resume", action="store_true", help="Resume from last translated alias")
    parser.add_argument("--apply-renames", action="store_true", help="Apply renames to platform")
    parser.add_argument("--fix-expressions", action="store_true", help="Fix _calc aliases in expressions")
    parser.add_argument("--apply-display-names", action="store_true", help="Apply displayNameRenamed to CTF JSON files")
    parser.add_argument("--apply-expressions", action="store_true", help="Apply expressionRenamed to CTF JSON files")
    parser.add_argument("--update-paths", action="store_true", help="Regenerate jsonPathRenamed from aliasRenamed")
    parser.add_argument("--path-mode", default="renamed", choices=["original", "renamed"], help="Which path to use for applying fixes")
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Run extract/finalize pipeline (disabled by default)",
    )
    parser.add_argument("--dangerous-suffix", default="_calc", help="Suffix for dangerous aliases")
    parser.add_argument("--safe-suffix", default="_sv", help="Suffix for safe aliases")

    args = parser.parse_args()

    result = localize_aliases.invoke({
        "application_system_name": args.app,
        "json_folder": args.json_folder,
        "output_dir": args.output_dir,
        "create_tr": args.create_tr,
        "translate_one": args.translate_one,
        "resume": args.resume,
        "apply_renames": args.apply_renames,
        "fix_expressions": args.fix_expressions,
        "apply_display_names": args.apply_display_names,
        "apply_expressions": args.apply_expressions,
        "update_paths": args.update_paths,
        "path_mode": args.path_mode,
        "dry_run": args.dry_run,
        "dangerous_suffix": args.dangerous_suffix,
        "safe_suffix": args.safe_suffix,
    })
    print(result)