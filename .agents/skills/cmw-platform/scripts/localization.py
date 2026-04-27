#!/usr/bin/env python3
"""
Volga1 Localization Utility Script

Automates localization workflow per schema.json:
0. Export CTF with date prefix (yyyyddMM-HHmmss)
1. Collect aliases from JSON (all CTF folder levels)
2. Verify IDs via API
3. Analyze expressions for dangerous aliases (_calc suffix)
4. Create _tr copy for translation
5. Interactive per-alias translation with state saving
6. Apply renames
7. Request platform restart
8. Re-export CTF
9. Fix expressions with _calc aliases
10. Import modified CTF

Usage:
    python localization.py --app <app_name> [--step N] [--resume]
"""

import argparse
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(APP_DIR))

from tools.applications_tools.tool_list_applications import list_applications
from tools.applications_tools.tool_get_ontology_objects import get_ontology_objects
from tools.applications_tools.tool_update_object_property import update_object_property
from tools.transfer_tools.tool_export_application import export_application
from tools.transfer_tools.tool_import_application import import_application


def get_timestamp() -> str:
    """Generate timestamp prefix: yyyyddMM-HHmmss."""
    return datetime.now().strftime("%Y%d%m-%H%M%S")


TYPE_FOLDER_MAP = {
    "RecordTemplate": "RecordTemplates",
    "ProcessTemplate": "ProcessTemplates",
    "Workspace": "Workspaces",
    "Page": "Pages",
    "RoleTemplate": "Roles",
    "Role": "Roles",
    "AccountTemplate": "Accounts",
    "OrgStructureTemplate": "OrgStructure",
    "MessageTemplate": "MessageTemplates",
    "Stream": "Streams",
    "Route": "Routes",
    "Trigger": "Triggers",
    "Attribute": "Attributes",
    "Dataset": "Datasets",
    "Form": "Forms",
    "Toolbar": "Toolbars",
    "UserCommand": "UserCommands",
    "Card": "Cards",
    "WidgetConfig": "WidgetConfigs",
}

FOLDER_TYPE_MAP = {v: k for k, v in TYPE_FOLDER_MAP.items()}

SUB_TYPES = {"Attributes", "Datasets", "Forms", "Toolbars", "UserCommands"}

EXPRESSION_KEYS = {"Expression", "Calculation", "DefaultExpression"}

SYSTEM_ALIASES = {
    "create", "edit", "delete", "archive", "deleteRole",
    "defaultCard", "defaultList", "defaultFormToolbar", "defaultListToolbar",
    "defaultModelToolbar", "defaultTaskToolbar", "defaultProcessToolbar",
    "defaultDiagramToolbar",
    "startEdit", "endEdit",
    "RoleTemplate", "OrgStructureTemplate", "RecordTemplate", "ProcessTemplate",
    "systemPage_architect",
    "editTask", "migrate", "editDiagram", "cancelProcess", "reassignTask",
    "completeTask", "archiveProcess", "createProcess", "retryTokens", "createToken",
    "noneStartEvent1", "executionFlow1", "pool1", "noneEndEvent1",
    "defaultForm",
    "roleForm", "Test_OrganizationalStructure",
}

SYSTEM_PREFIXES = (
    "cmw.", "oa.", "pa.", "msgt.", "aa.", "ra.", "os.",
    "form.", "tb.", "lst.", "event.", "card.", "trigger.", "workspace.",
    "role.", "fw.",
)


def find_app(app_name: str) -> dict | None:
    """Find application by name."""
    result = list_applications.invoke({})
    if result["success"]:
        for app in result["data"]:
            if app_name.lower() in app.get("Name", "").lower():
                return app
    return None


def export_ctf(app_name: str) -> tuple[Path | None, str]:
    """Export application to CTF format. Returns (path, timestamp)."""
    ts = get_timestamp()
    print(f"[{ts}] Exporting {app_name} to CTF...")
    result = export_application.invoke({
        "application_system_name": app_name,
        "save_to_file": True,
    })
    if result["success"]:
        return Path(result["ctf_file_path"]), ts
    print(f"Error: {result.get('error')}")
    return None, ts


def extract_ctf(ctf_path: Path, extract_dir: Path) -> Path:
    """Extract CTF to directory."""
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(ctf_path, "r") as z:
        z.extractall(extract_dir)
    return extract_dir


def collect_objects(base: Path, app_name: str) -> list[dict]:
    """Collect all objects from CTF JSON at all folder levels."""
    app_dir = base / app_name
    if not app_dir.exists():
        print(f"Error: {app_dir} not found")
        return []

    objects = []

    def scan_folder(folder_path: Path, relative_prefix: str = ""):
        """Recursively scan folder for JSON files with GlobalAlias."""
        if not folder_path.exists():
            return

        for item in folder_path.iterdir():
            if item.is_dir():
                sub_prefix = f"{relative_prefix}/{item.name}" if relative_prefix else item.name
                if item.name in SUB_TYPES:
                    scan_folder(item, sub_prefix)
                else:
                    scan_folder(item, sub_prefix)
            elif item.suffix == ".json":
                scan_json_file(item, relative_prefix)

    def scan_json_file(json_path: Path, folder_prefix: str):
        """Extract alias, display name, and expressions from a JSON file."""
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            return

        global_alias = data.get("GlobalAlias", {})
        alias = global_alias.get("Alias")
        if not alias:
            return

        if alias in SYSTEM_ALIASES:
            return
        if alias.startswith(SYSTEM_PREFIXES):
            return

        obj_type = _infer_type(json_path, folder_prefix, global_alias)
        display_name = data.get("Name", "")

        expressions = []
        for key in EXPRESSION_KEYS:
            if key in data:
                expr_val = data[key]
                if isinstance(expr_val, str) and expr_val.strip():
                    expressions.append({
                        "jsonPathOriginal": f"{folder_prefix}/{json_path.name}",
                        "jsonPathRenamed": "",
                        "expressionOriginal": expr_val,
                        "expressionRenamed": "",
                    })

        path_in_ctf = f"{folder_prefix}/{json_path.name}" if folder_prefix else json_path.name

        objects.append({
            "type": obj_type,
            "id": "",
            "aliasOriginal": alias,
            "aliasRenamed": "",
            "displayNameOriginal": display_name,
            "displayNameRenamed": "",
            "jsonPathOriginal": path_in_ctf,
            "jsonPathRenamed": "",
            "expressions": expressions,
        })

    def _infer_type(json_path: Path, folder_prefix: str, global_alias: dict) -> str:
        """Infer object type from folder structure and GlobalAlias."""
        parts = Path(folder_prefix).parts if folder_prefix else []
        if parts:
            first_folder = parts[0]
            if first_folder in FOLDER_TYPE_MAP:
                return FOLDER_TYPE_MAP[first_folder]
            if first_folder in SUB_TYPES:
                return first_folder[:-1] if first_folder.endswith("s") else first_folder

        if global_alias.get("$type") == "cmw.container":
            return "RecordTemplate"
        if global_alias.get("$type") == "cmw.process":
            return "ProcessTemplate"

        return "Unknown"

    scan_folder(app_dir)
    return objects


def verify_aliases(objects: list[dict], app_name: str) -> list[dict]:
    """Verify aliases via API and populate IDs. Only keep aliases found in CTF."""
    all_types = list(set(o["type"] for o in objects if o["type"] != "Unknown"))

    verified = {}
    for obj_type in all_types:
        print(f"  Verifying {obj_type}...")
        result = get_ontology_objects.invoke({
            "application_system_name": app_name,
            "types": [obj_type],
            "parameter": "alias",
            "min_count": 1,
            "max_count": 5000,
        })

        if result["success"] and result.get("data"):
            for obj in result["data"]:
                verified[obj["systemName"]] = obj["id"]

        time.sleep(0.3)

    verified_objects = []
    skipped = 0
    for obj in objects:
        alias = obj["aliasOriginal"]
        if alias in verified:
            obj["id"] = verified[alias]
            verified_objects.append(obj)
        else:
            skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} aliases not in platform")

    return verified_objects


def analyze_dangerous(objects: list[dict], extract_dir: Path, app_name: str) -> dict:
    """Analyze aliases used in expressions, return {alias: suffix}."""
    base = extract_dir / app_name
    all_aliases = {o["aliasOriginal"] for o in objects}
    dangerous = set()

    for folder in ["RecordTemplates", "ProcessTemplates"]:
        folder_path = base / folder
        if not folder_path.exists():
            continue

        for json_file in folder_path.rglob("*.json"):
            try:
                content = json_file.read_text()
                data = json.loads(content)
            except (json.JSONDecodeError, OSError):
                continue

            for kw in EXPRESSION_KEYS:
                if kw not in data:
                    continue
                expr = str(data[kw])
                for alias in all_aliases:
                    if alias in expr and re.search(rf'\b{re.escape(alias)}\b', expr):
                        dangerous.add(alias)

    suffix_map = {}
    for obj in objects:
        alias = obj["aliasOriginal"]
        if alias in dangerous:
            suffix_map[alias] = "_calc"
        else:
            suffix_map[alias] = "_calc"

    return suffix_map


def generate_tr_file(objects: list[dict], extract_dir: Path, app_name: str, ts: str) -> Path:
    """Create _tr copy of objects with date prefix."""
    tr_dir = extract_dir / f"{app_name}_tr"
    os.makedirs(tr_dir, exist_ok=True)
    tr_file = tr_dir / f"{ts}_{app_name}_tr.json"
    with open(tr_file, "w", encoding="utf-8") as f:
        json.dump(objects, f, indent=2, ensure_ascii=False)
    return tr_file


def generate_report(objects: list[dict], ts: str, app_name: str, step_name: str) -> Path:
    """Generate human-readable markdown report."""
    lines = [
        f"# Localization Report: {app_name} - {step_name}",
        f"**Generated:** {ts}",
        f"**Total Objects:** {len(objects)}",
        "",
        "## Objects",
        "",
        "| # | Type | ID | aliasOriginal | aliasRenamed | displayNameOriginal |",
        "|---|------|----|---------------|--------------|---------------------|",
    ]

    for i, obj in enumerate(objects, 1):
        alias_orig = obj.get("aliasOriginal", "")
        alias_renamed = obj.get("aliasRenamed", "")
        display_orig = obj.get("displayNameOriginal", "")
        obj_id = obj.get("id", "")
        obj_type = obj.get("type", "")

        lines.append(f"| {i} | {obj_type} | {obj_id} | `{alias_orig}` | {alias_renamed} | {display_orig} |")

    report_text = "\n".join(lines)
    report_path = Path("/tmp/cmw-transfer") / f"{ts}_{app_name}_{step_name.replace(' ', '_')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print(f"  Report: {report_path}")
    return report_path


def load_tr_file(tr_file: Path) -> tuple[list[dict], str]:
    """Load _tr file and return (objects, last_renamed_alias)."""
    with open(tr_file, encoding="utf-8") as f:
        objects = json.load(f)

    last_alias = ""
    for obj in objects:
        if obj.get("aliasRenamed"):
            last_alias = obj["aliasOriginal"]

    return objects, last_alias


def save_tr_file(tr_file: Path, objects: list[dict]):
    """Save _tr file with current state."""
    with open(tr_file, "w", encoding="utf-8") as f:
        json.dump(objects, f, indent=2, ensure_ascii=False)


def interactive_translate(tr_file: Path, objects: list[dict], resume_from: str = ""):
    """Interactive per-alias translation with state saving."""
    resume = resume_from is not None and resume_from != ""
    started = False if resume else True

    print(f"\n=== Interactive Translation ===")
    print(f"Total aliases: {len(objects)}")
    print("Type new aliasRenamed (Enter to skip), displayNameRenamed (Enter to skip)")
    print("Commands: 'q' quit, 's' save, 'r <alias>' resume from alias\n")

    idx = 0
    while idx < len(objects):
        obj = objects[idx]

        if resume and not started:
            if obj["aliasOriginal"] == resume_from:
                started = True
            else:
                idx += 1
                continue
        elif not started:
            started = True

        print(f"[{idx + 1}/{len(objects)}] type={obj['type']}")
        print(f"  aliasOriginal: {obj['aliasOriginal']}")
        print(f"  displayNameOriginal: {obj['displayNameOriginal']}")
        print(f"  jsonPathOriginal: {obj['jsonPathOriginal']}")
        if obj["expressions"]:
            print(f"  expressions count: {len(obj['expressions'])}")

        alias_input = input(f"  aliasRenamed [{obj['aliasOriginal']}_calc]: ").strip()
        name_input = input(f"  displayNameRenamed [{obj['displayNameOriginal']}]: ").strip()

        obj["aliasRenamed"] = alias_input if alias_input else f"{obj['aliasOriginal']}_calc"
        obj["displayNameRenamed"] = name_input if name_input else obj["displayNameOriginal"]

        obj["jsonPathRenamed"] = obj["jsonPathOriginal"]

        if obj["expressions"]:
            print("  Processing expressions...")
            for expr in obj["expressions"]:
                expr["expressionRenamed"] = expr["expressionOriginal"].replace(
                    obj["aliasOriginal"], obj["aliasRenamed"]
                )
                expr["jsonPathRenamed"] = expr["jsonPathOriginal"]

        save_tr_file(tr_file, objects)
        print(f"  [Saved] aliasRenamed={obj['aliasRenamed']}\n")

        idx += 1

    print(f"\n=== Translation Complete ===")


def apply_renames(objects: list[dict], app_name: str) -> tuple[int, int]:
    """Apply aliasRenamed via API."""
    type_map = {
        "RecordTemplate": "RecordTemplate",
        "ProcessTemplate": "ProcessTemplate",
        "Workspace": "Workspace",
        "Page": "Page",
        "RoleTemplate": "RoleTemplate",
        "Role": "Role",
        "AccountTemplate": "AccountTemplate",
        "OrgStructureTemplate": "OrgStructureTemplate",
        "Attribute": "Attribute",
        "Dataset": "Dataset",
        "Form": "Form",
        "Toolbar": "Toolbar",
        "UserCommand": "UserCommand",
        "WidgetConfig": "WidgetConfig",
    }

    success = 0
    failed = 0

    for i, obj in enumerate(objects):
        if not obj.get("id") or not obj.get("aliasRenamed"):
            continue

        obj_type = obj["type"]
        plat_type = type_map.get(obj_type)
        if not plat_type:
            continue

        result = update_object_property.invoke({
            "object_id": obj["id"],
            "object_type": plat_type,
            "new_value": obj["aliasRenamed"],
        })

        if result.get("success"):
            success += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(objects)} | Success: {success}, Failed: {failed}")

        time.sleep(0.05)

    return success, failed


def fix_expressions_in_ctf(objects: list[dict], extract_dir: Path, app_name: str):
    """Replace _calc aliases in expression fields within CTF JSON."""
    base = extract_dir / app_name

    replacements = []
    for obj in objects:
        orig = obj["aliasOriginal"]
        renamed = obj["aliasRenamed"]
        if renamed and renamed != orig and renamed.endswith("_calc"):
            replacements.append((orig, renamed))

    if not replacements:
        print("  No _calc replacements needed")
        return

    modified = 0
    for orig_alias, new_alias in replacements:
        for json_file in base.rglob("*.json"):
            try:
                content = json_file.read_text()
            except OSError:
                continue

            new_content = content.replace(orig_alias, new_alias)
            if new_content != content:
                json_file.write_text(new_content)
                modified += 1

    print(f"  Fixed expressions in {modified} files")


def main():
    parser = argparse.ArgumentParser(
        description="Volga1 Localization Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python localization.py --app Volga1
  python localization.py --app Volga1 --step 5 --resume MyAlias
  python localization.py --app Volga1 --apply-only
        """,
    )
    parser.add_argument("--app", required=True, help="Application name (e.g., Volga1)")
    parser.add_argument(
        "--step",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        default=0,
        help="Resume from step (0=export, 5=translate, 6=apply, 10=import)",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Resume translation from this alias (use with --step 5)",
    )
    parser.add_argument(
        "--start-id",
        default="",
        help="Start from this alias ID (for step 3, 5, 6)",
    )
    parser.add_argument(
        "--apply-only",
        action="store_true",
        help="Skip export, apply renames from existing _tr file",
    )
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="Import modified CTF (skip other steps)",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/cmw-transfer",
        help="Output directory for CTF files (default: /tmp/cmw-transfer)",
    )

    args = parser.parse_args()
    app_name = args.app
    base_dir = Path(args.output_dir)
    os.makedirs(base_dir, exist_ok=True)

    ts = get_timestamp()
    extract_dir = base_dir / f"{app_name}-extract"
    tr_file = base_dir / f"{ts}_{app_name}_tr.json"

    print(f"=== Localization: {app_name} [{ts}] ===")

    if args.import_only:
        ctf_file = list(base_dir.glob(f"{ts}_{app_name}_modified.ctf"))
        if not ctf_file:
            print("No modified CTF found")
            return
        print(f"Importing {ctf_file[0]}...")
        result = import_application.invoke({
            "application_system_name": app_name,
            "ctf_file_path": str(ctf_file[0]),
            "update_existing": True,
        })
        if result["success"]:
            print("Import successful")
        else:
            print(f"Import failed: {result.get('error')}")
        return

    if not args.apply_only:
        if args.step <= 0:
            ctf_path, ts = export_ctf(app_name)
            if not ctf_path:
                return
            print(f"  CTF: {ctf_path}")

            extract_ctf(ctf_path, extract_dir)
            print(f"  Extracted to: {extract_dir}")

        if args.step <= 1:
            print(f"\n[{ts}] Step 1: Collecting objects...")
            objects = collect_objects(extract_dir, app_name)
            print(f"  Collected: {len(objects)} objects")

        if args.step <= 2:
            print(f"\n[{ts}] Step 2: Verifying aliases...")
            objects = verify_aliases(objects, app_name)
            print(f"  Verified: {sum(1 for o in objects if o.get('id'))} with IDs")
            generate_report(objects, ts, app_name, "original JSON")

        if args.step <= 3:
            print(f"\n[{ts}] Step 3: Analyzing expressions...")
            suffix_map = analyze_dangerous(objects, extract_dir, app_name)
            for obj in objects:
                if not obj.get("aliasRenamed"):
                    suffix = suffix_map.get(obj["aliasOriginal"], "_calc")
                    obj["aliasRenamed"] = f"{obj['aliasOriginal']}{suffix}"

        if args.step <= 4:
            print(f"\n[{ts}] Step 4: Creating _tr file...")
            tr_file = generate_tr_file(objects, extract_dir, app_name, ts)
            print(f"  Saved: {tr_file}")
            print("\n  Proceed with step 5 to translate aliases interactively")

        print(f"\n=== State saved. To continue: ===")
        print(f"  python localization.py --app {app_name} --step 5")
        print(f"  python localization.py --app {app_name} --step 5 --resume <alias>")
        return

    if args.step <= 5 and tr_file.exists():
        print(f"\n[{ts}] Step 5: Interactive translation...")
        objects, last_alias = load_tr_file(tr_file)
        resume_from = args.resume if args.resume else (last_alias if args.step == 5 else "")
        interactive_translate(tr_file, objects, resume_from)

        print(f"\n[{ts}] Step 5a: Generating jsonPathRenamed and report...")
        for obj in objects:
            if obj.get("aliasRenamed"):
                obj["jsonPathRenamed"] = obj["jsonPathOriginal"]
                for expr in obj.get("expressions", []):
                    if expr.get("expressionOriginal"):
                        expr["expressionRenamed"] = expr["expressionOriginal"].replace(
                            obj["aliasOriginal"], obj["aliasRenamed"]
                        )
                        expr["jsonPathRenamed"] = expr["jsonPathOriginal"]
        save_tr_file(tr_file, objects)
        generate_report(objects, ts, app_name, "JSON translated")

    if args.step <= 6 and tr_file.exists():
        print(f"\n[{ts}] Step 6: Applying renames...")
        objects, _ = load_tr_file(tr_file)
        objects = [o for o in objects if o.get("id") and o.get("aliasRenamed")]
        success, failed = apply_renames(objects, app_name)
        print(f"  Applied: {success} success, {failed} failed")

    if args.step <= 7:
        print(f"\n[{ts}] Step 7: Restart required")
        print("  Aliases have been renamed. Please restart the platform now.")
        print(f"  After restart, run: python localization.py --app {app_rename} --step 8")
        return

    if args.step <= 8:
        print(f"\n[{ts}] Step 8: Re-exporting CTF...")
        ctf_path, ts = export_ctf(app_name)
        if ctf_path:
            extract_ctf(ctf_path, extract_dir)
            print(f"  Re-exported and extracted to: {extract_dir}")

    if args.step <= 9:
        print(f"\n[{ts}] Step 9: Fixing expressions...")
        objects, _ = load_tr_file(tr_file)
        fix_expressions_in_ctf(objects, extract_dir, app_name)
        print(f"  Expressions fixed in CTF")

    if args.step <= 10:
        print(f"\n[{ts}] Step 10: Importing modified CTF...")

        import zipfile
        modified_ctf = base_dir / f"{ts}_{app_name}_modified.ctf"
        with zipfile.ZipFile(modified_ctf, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in extract_dir.rglob("*"):
                if file_path.is_file() and file_path.name != modified_ctf.name:
                    arcname = file_path.relative_to(extract_dir)
                    zf.write(file_path, arcname)

        result = import_application.invoke({
            "application_system_name": app_name,
            "ctf_file_path": str(modified_ctf),
            "update_existing": True,
        })
        if result["success"]:
            print("  Import successful")
        else:
            print(f"  Import failed: {result.get('error')}")


if __name__ == "__main__":
    main()