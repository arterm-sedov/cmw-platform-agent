# tool_localize.py - Localization workflow for system names (aliases) and display names
# Orchestrates helper scripts via direct imports
# Output format: {domain}_{app}_aliases.json, {domain}_{app}_aliases_tr.json
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import os
from typing import Any

try:
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field
except ImportError:
    from tools.models import BaseModel, Field
    from tools.tool_utils import tool

# ============================================================================
# CONSTANTS
# ============================================================================
SCRIPTS_DIR = Path(__file__).parent.parent.parent / ".agents/skills/cmw-platform/scripts"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def run_via_import(script_name: str, args: list[str], description: str = "") -> None:
    """Run a helper script via import + mocked sys.argv. Raises RuntimeError on failure."""
    if description:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {script_name} {' '.join(args[:3])} ...")
        print(f"{'='*60}")

    # Mock sys.argv
    old_argv = sys.argv
    sys.argv = [script_name] + args

    try:
        # Import and call main()
        script_path = SCRIPTS_DIR / script_name
        import importlib.util
        spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        exit_code = module.main()
        if exit_code != 0:
            raise RuntimeError(f"{script_name} failed (exit {exit_code})")
    except Exception as e:
        raise RuntimeError(f"{script_name} failed: {e}")
    finally:
        sys.argv = old_argv


def get_domain_from_config() -> str:
    """Extract domain from config URL."""
    try:
        from tools.config import get_config
        base_url = get_config().get("base_url", "")
        from urllib.parse import urlparse
        return urlparse(base_url).netloc.split('.')[0] or "cmw"
    except Exception:
        return "cmw"


def load_resume_state(output_dir: str, app: str) -> dict | None:
    """Load resume state for translation."""
    state_file = Path(output_dir) / f"{app}_localize_resume.json"
    if state_file.exists():
        try:
            with open(state_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_resume_state(output_dir: str, app: str, last_alias: str, index: int):
    """Save resume state."""
    from datetime import datetime
    state_file = Path(output_dir) / f"{app}_localize_resume.json"
    state = {
        "last_alias": last_alias,
        "last_index": index,
        "updated": datetime.now().isoformat(),
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def fix_expressions_in_memory(objects: list, dangerous_suffix: str = "_calc") -> int:
    """Fix _calc aliases in expression fields in objects list (in memory only)."""
    import re
    fixed_count = 0

    for obj in objects:
        if not obj.get("expressions"):
            continue

        alias_original = obj.get("aliasOriginal", "")
        alias_renamed = obj.get("aliasRenamed", "")
        if not alias_renamed or not alias_renamed.endswith(dangerous_suffix):
            continue

        for expr in obj.get("expressions", []):
            orig_expr = expr.get("expressionOriginal", "")
            if not orig_expr:
                continue

            new_expr = orig_expr
            safe_alias = re.escape(alias_original)

            patterns = [
                (r'"' + safe_alias + r'"', f'"{alias_renamed}"'),
                (r"\$\{" + safe_alias + r"\}", f"${{{alias_renamed}}}"),
                (r"->\{" + safe_alias + r"\}", f"->{{{alias_renamed}}}"),
                (r"\{" + safe_alias + r"\}->", f"{{{alias_renamed}}}->"),
            ]

            for pattern, replacement in patterns:
                new_expr = re.sub(pattern, replacement, new_expr)

            if new_expr != orig_expr:
                expr["expressionRenamed"] = new_expr
                fixed_count += 1

    return fixed_count


def convert_to_schema_format(entries: list, domain: str, app: str, output_dir: str) -> str:
    """Save entries to schema.json format file.

    Since tool_finalize.py now outputs in schema format directly,
    this function just saves the entries to the expected filename.
    """
    output_file = Path(output_dir) / f"{domain}_{app}_aliases.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    return str(output_file)


# ============================================================================
# SCHEMA AND MAIN FUNCTION
# ============================================================================
class LocalizeSchema(BaseModel):
    application_system_name: str = Field(description="Application system name")
    json_folder: str = Field(description="Path to folder containing JSON files to analyze")
    output_dir: str | None = Field(default=None, description="Path to save output files")
    create_tr: bool = Field(default=False, description="Create _tr copy from original")
    translate_one: str | None = Field(default=None, description="Translate single alias")
    resume: bool = Field(default=False, description="Resume from last translated alias")
    apply_renames: bool = Field(default=False, description="Apply renames to platform")
    fix_expressions: bool = Field(default=False, description="Fix _calc aliases in expressions")
    dry_run: bool = Field(default=True, description="If True, only analyze without making changes")
    dangerous_suffix: str = Field(default="_calc", description="Suffix for dangerous system names")
    safe_suffix: str = Field(default="_sv", description="Suffix for safe system names")


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
    dry_run: bool = True,
    dangerous_suffix: str = "_calc",
    safe_suffix: str = "_sv",
) -> dict[str, Any]:
    """
    Localization workflow for system names (aliases) and display names.

    Follows schema.json specification:
    - Output files: {domain}_{app}_aliases.json and {domain}_{app}_aliases_tr.json
    - Per-object structure with aliasOriginal, aliasRenamed, jsonPathOriginal, jsonPathRenamed, expressions

    Workflow (orchestrates helper scripts via import):
    1. Extract aliases (tool_extract_aliases.py)
    2. Collect platform data (tool_collect_platform.py)
    3. Verify aliases (tool_verify_aliases.py per folder)
    4. Find dangerous aliases (tool_find_dangerous.py)
    5. Finalize (tool_finalize.py)
    6. Convert to schema format
    7. --create-tr: Copy to _tr, fix _calc aliases in expressions
    8. --translate-one: Translate single alias, save state
    9. --resume: Continue from last translated alias
    10. --apply-renames: Rename on platform (apply_renames.py)
    11. --fix-expressions: Fix _calc aliases in expressions
    """
    from datetime import datetime

    results = {
        "success": True,
        "phase": "started",
        "actions": [],
        "errors": [],
    }

    if not output_dir:
        output_dir = json_folder

    os.makedirs(output_dir, exist_ok=True)

    extract_dir = str(Path(json_folder).parent)

    # ========================================================================
    # STEPS 1-5: ORCHESTRATE HELPER SCRIPTS VIA IMPORT
    # ========================================================================
    if not dry_run:
        try:
            # Step 1: Extract aliases
            run_via_import("tool_extract_aliases.py", [
                "--app", application_system_name,
                "--extract-dir", extract_dir,
                "--output-dir", output_dir,
            ], "Step 1: Extract Aliases")

            # Step 2: Collect platform data
            run_via_import("tool_collect_platform.py", [
                "--app", application_system_name,
                "--output-dir", output_dir,
                "--workers", "8",
            ], "Step 2: Collect Platform Data")

            # Step 3: Verify aliases (per folder)
            state_file = Path(output_dir) / f"{application_system_name}_extraction_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    folders = json.load(f).get("completed_folders", [])
            else:
                folders = [d.name for d in Path(json_folder).iterdir() if d.is_dir()]

            for folder in folders:
                vf = Path(output_dir) / f"{application_system_name}_{folder}_verified.json"
                if not vf.exists():
                    run_via_import("tool_verify_aliases.py", [
                        "--app", application_system_name,
                        "--folder", folder,
                        "--output-dir", output_dir,
                    ], f"Step 3: Verify {folder}")

            # Step 4: Find dangerous aliases
            run_via_import("tool_find_dangerous.py", [
                "--app", application_system_name,
                "--extract-dir", extract_dir,
                "--output-dir", output_dir,
                "--workers", "4",
            ], "Step 4: Find Dangerous Aliases")

            # Step 5: Finalize (merge + locking)
            run_via_import("tool_finalize.py", [
                "--app", application_system_name,
                "--output-dir", output_dir,
            ], "Step 5: Finalize")

            # Step 6: Convert to schema format
            verified_file = Path(output_dir) / f"{application_system_name}_verified_complete.json"
            if verified_file.exists():
                with open(verified_file) as f:
                    entries = json.load(f)

                domain = get_domain_from_config()
                output_path = convert_to_schema_format(entries, domain, application_system_name, output_dir)

                results["output_file"] = output_path
                results["total_entries"] = len(entries)

                locked = sum(1 for e in entries if e.get("aliasLocked"))
                results["locked_entries"] = locked
                results["actions"].append(f"Saved schema format: {output_path}")

            # Load dangerous aliases count
            dangerous_file = Path(output_dir) / f"{application_system_name}_dangerous_aliases.json"
            if dangerous_file.exists():
                with open(dangerous_file) as f:
                    dangerous_data = json.load(f)
                results["dangerous_count"] = len(dangerous_data.get("dangerous_aliases", []))

        except RuntimeError as e:
            results["success"] = False
            results["errors"].append(str(e))
            results["phase"] = "error"
            return results

    # ========================================================================
    # STEP 7: CREATE TR (inline - lightweight JSON manipulation)
    # ========================================================================
    if create_tr:
        results["phase"] = "create_tr"
        domain = get_domain_from_config()
        original_file = Path(output_dir) / f"{domain}_{application_system_name}_aliases.json"

        if not original_file.exists():
            original_file = Path(output_dir) / f"{application_system_name}_verified_complete.json"

        if not original_file.exists():
            results["success"] = False
            results["errors"].append("Original aliases file not found. Run without --create-tr first.")
            return results

        with open(original_file) as f:
            original_data = json.load(f)

        tr_data = []
        for obj in original_data:
            obj_copy = obj.copy()
            is_dangerous = bool(obj_copy.get("expressions"))
            suffix = dangerous_suffix if is_dangerous else safe_suffix
            obj_copy["aliasRenamed"] = obj_copy.get("aliasOriginal", "") + suffix
            obj_copy["jsonPathRenamed"] = obj_copy.get("jsonPathOriginal", [])[:]

            for expr in obj_copy.get("expressions", []):
                expr["jsonPathRenamed"] = expr.get("jsonPathOriginal", "")
                orig_expr = expr.get("expressionOriginal", "")
                alias_orig = obj.get("aliasOriginal", "")
                alias_new = obj_copy["aliasRenamed"]
                expr["expressionRenamed"] = orig_expr.replace(alias_orig, alias_new) if orig_expr else ""

            tr_data.append(obj_copy)

        tr_file = Path(output_dir) / f"{domain}_{application_system_name}_aliases_tr.json"
        with open(tr_file, "w", encoding="utf-8") as f:
            json.dump(tr_data, f, indent=2, ensure_ascii=False)

        results["actions"].append(f"Created translation copy: {tr_file}")

        if fix_expressions or create_tr:
            fixed = fix_expressions_in_memory(tr_data, dangerous_suffix)
            results["actions"].append(f"Fixed _calc aliases in {fixed} expressions (in memory)")

            with open(tr_file, "w", encoding="utf-8") as f:
                json.dump(tr_data, f, indent=2, ensure_ascii=False)

            results["actions"].append(f"Updated translation file: {tr_file}")

        results["phase"] = "complete"
        return results

    # ========================================================================
    # STEP 8-9: TRANSLATE (inline - interactive)
    # ========================================================================
    if translate_one or resume:
        results["phase"] = "translate"

        domain = get_domain_from_config()
        tr_file = Path(output_dir) / f"{domain}_{application_system_name}_aliases_tr.json"

        if not tr_file.exists():
            results["success"] = False
            results["errors"].append("Translation file not found. Run with --create-tr first.")
            return results

        with open(tr_file) as f:
            tr_data = json.load(f)

        resume_state = load_resume_state(output_dir, application_system_name)
        start_index = 0

        if resume and resume_state:
            start_index = resume_state.get("last_index", 0)
            results["actions"].append(f"Resuming from index {start_index}")

        target_index = start_index
        if translate_one:
            for i, obj in enumerate(tr_data):
                if obj.get("aliasOriginal") == translate_one:
                    target_index = i
                    break

        if target_index >= len(tr_data):
            results["success"] = False
            results["errors"].append(f"Alias {translate_one} not found in translation file")
            return results

        obj = tr_data[target_index]
        alias_orig = obj.get("aliasOriginal", "")
        alias_new = obj.get("aliasRenamed", "")

        print(f"Alias {target_index + 1}/{len(tr_data)}: {alias_orig}")
        print(f"  Type: {obj.get('type', '')}")
        print(f"  ID: {obj.get('ids', [])}")
        print(f"  Display Name: {obj.get('displayNameOriginal', '')}")
        print(f"  Current aliasRenamed: {alias_new}")

        if not alias_new:
            suffix = dangerous_suffix if obj.get("expressions") else safe_suffix
            new_alias = alias_orig + suffix
            print(f"  Suggested: {new_alias}")
            print("  Enter new aliasRenamed (or press Enter to accept suggested): ")

            new_input = input("  > ")
            if new_input.strip():
                obj["aliasRenamed"] = new_input.strip()
            else:
                obj["aliasRenamed"] = new_alias

            obj["jsonPathRenamed"] = obj.get("jsonPathOriginal", [])[:]

            for expr in obj.get("expressions", []):
                expr["jsonPathRenamed"] = expr.get("jsonPathOriginal", "")
                orig_expr = expr.get("expressionOriginal", "")
                expr["expressionRenamed"] = orig_expr.replace(alias_orig, obj["aliasRenamed"]) if orig_expr else ""

            save_resume_state(output_dir, application_system_name, alias_orig, target_index)

            results["actions"].append(f"Translated {alias_orig} -> {obj['aliasRenamed']}")

        results["phase"] = "complete"
        return results

    # ========================================================================
    # STEP 10: APPLY RENAMES (import)
    # ========================================================================
    if apply_renames:
        results["phase"] = "apply_renames"

        try:
            run_via_import("apply_renames.py", [], "Apply Renames")
            results["actions"].append("Applied renames to platform")
        except RuntimeError as e:
            results["success"] = False
            results["errors"].append(str(e))

        results["phase"] = "complete"
        return results

    # ========================================================================
    # STEP 11: FIX EXPRESSIONS (inline)
    # ========================================================================
    if fix_expressions:
        results["phase"] = "fix_expressions"

        domain = get_domain_from_config()
        tr_file = Path(output_dir) / f"{domain}_{application_system_name}_aliases_tr.json"

        if not tr_file.exists():
            results["success"] = False
            results["errors"].append("Translation file not found.")
            return results

        with open(tr_file) as f:
            tr_data = json.load(f)

        fixed = fix_expressions_in_memory(tr_data, dangerous_suffix)
        results["actions"].append(f"Fixed _calc aliases in {fixed} expressions (in memory)")

        with open(tr_file, "w", encoding="utf-8") as f:
            json.dump(tr_data, f, indent=2, ensure_ascii=False)

        results["actions"].append(f"Updated translation file: {tr_file}")

        results["phase"] = "complete"
        return results

    results["phase"] = "complete"
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
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Actually perform changes (not just preview)")
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
        "dry_run": args.dry_run,
        "dangerous_suffix": args.dangerous_suffix,
        "safe_suffix": args.safe_suffix,
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))
