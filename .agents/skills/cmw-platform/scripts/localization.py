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

EXPRESSION_KEYS = {"Expression", "Code", "ValueExpression", "ValidationScript", "Calculation", "DefaultExpression"}

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
            # Check immediate parent folder first (for nested objects)
            immediate_parent = parts[-1] if parts else ""
            if immediate_parent in SUB_TYPES:
                return immediate_parent[:-1] if immediate_parent.endswith("s") else immediate_parent
            # Fall back to first folder
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
    """Analyze aliases used in expressions across ALL JSON files at ALL nesting levels.

    Uses optimized combined regex patterns for speed.
    """
    base = extract_dir / app_name
    all_aliases = list({o["aliasOriginal"] for o in objects})
    dangerous = set()

    print("  Building combined patterns...")
    pattern_start = time.time()

    p1 = r"\$\{(" + "|".join(re.escape(a) for a in all_aliases) + r")\}"
    p2 = r"\->\{(" + "|".join(re.escape(a) for a in all_aliases) + r")\}"
    p3 = r"\{(" + "|".join(re.escape(a) for a in all_aliases) + r")\}->"
    p4 = r"\"" + "|".join(re.escape(a) for a in all_aliases) + r"\""

    regex1 = re.compile(p1)
    regex2 = re.compile(p2)
    regex3 = re.compile(p3)
    regex4 = re.compile(p4)

    print(f"  Patterns compiled in {time.time() - pattern_start:.1f}s")

    files_with_expressions = []
    for json_file in base.rglob("*.json"):
        try:
            content = json_file.read_text(encoding="utf-8")
            if any(kw in content for kw in EXPRESSION_KEYS):
                files_with_expressions.append(json_file)
        except (OSError, UnicodeDecodeError):
            continue

    print(f"  Found {len(files_with_expressions)} files with expression content")

    def scan_expressions(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in EXPRESSION_KEYS and isinstance(value, str):
                    for regex in [regex1, regex2, regex3, regex4]:
                        for match in regex.findall(value):
                            dangerous.add(match)
                scan_expressions(value)
        elif isinstance(obj, list):
            for item in obj:
                scan_expressions(item)

    for json_file in files_with_expressions:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        scan_expressions(data)

    suffix_map = {}
    for obj in objects:
        alias = obj["aliasOriginal"]
        if alias in dangerous:
            suffix_map[alias] = "_calc"
        else:
            suffix_map[alias] = "_sv"

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
    """Generate human-readable markdown report with expressions."""
    dangerous = [o for o in objects if o.get("aliasRenamed", "").endswith("_calc")]
    safe = [o for o in objects if o.get("aliasRenamed", "").endswith("_sv")]

    lines = [
        f"# Localization Report: {app_name} - {step_name}",
        f"**Generated:** {ts}",
        f"**Total Objects:** {len(objects)}",
        f"**Dangerous (_calc):** {len(dangerous)}",
        f"**Safe (_sv):** {len(safe)}",
        "",
        "## Summary by Type",
        "",
        "| Type | Total | Dangerous | Safe |",
        "|------|-------|------------|------|",
    ]

    from collections import defaultdict
    type_stats = defaultdict(lambda: {"total": 0, "dangerous": 0, "safe": 0})
    for obj in objects:
        t = obj["type"]
        type_stats[t]["total"] += 1
        if obj.get("aliasRenamed", "").endswith("_calc"):
            type_stats[t]["dangerous"] += 1
        else:
            type_stats[t]["safe"] += 1

    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        lines.append(f"| {t} | {s['total']} | {s['dangerous']} | {s['safe']} |")

    lines.extend(["", "---", "", "## Dangerous Aliases (with expressions)"])

    for i, obj in enumerate(dangerous, 1):
        lines.append(f"### {i}. {obj['aliasOriginal']} → {obj['aliasRenamed']}")
        lines.append(f"- **Type:** {obj['type']}")
        lines.append(f"- **Object ID:** {obj['id']}")
        lines.append(f"- **Display Name:** {obj.get('displayNameOriginal', 'N/A')}")
        lines.append(f"- **JSON Path:** `{obj.get('jsonPathOriginal', 'N/A')}`")
        expressions = obj.get("expressions", [])
        if expressions:
            lines.append(f"- **Expressions ({len(expressions)}):**")
            for expr in expressions[:5]:
                lines.append(f"  - `{expr.get('jsonPathOriginal', 'unknown')}`")
                expr_text = expr.get('expressionOriginal', '')[:150]
                lines.append(f"    ```")
                lines.append(f"    {expr_text}...")
                lines.append(f"    ```")
        lines.append("")

    lines.extend(["", "---", "", "## Safe Aliases (sample)"])
    lines.append("")
    lines.append("| # | System Name | Display Name | Type | Renamed |")
    lines.append("|---|-------------|-------------|------|---------|")
    for i, obj in enumerate(safe[:100], 1):
        display = obj.get('displayNameOriginal', 'N/A')[:40]
        lines.append(f"| {i} | `{obj['aliasOriginal']}` | {display} | {obj['type']} | {obj['aliasRenamed']} |")

    if len(safe) > 100:
        lines.append(f"")
        lines.append(f"... and {len(safe) - 100} more safe aliases.")

    lines.extend(["", "---", "", "## Complete Object List"])
    lines.append("")
    lines.append("| Type | System Name | Display Name | ID | Renamed | Category |")
    lines.append("|------|-------------|-------------|----|---------|-----------|")
    for obj in objects:
        display = obj.get('displayNameOriginal', 'N/A')[:40]
        category = "Dangerous" if obj.get('aliasRenamed', '').endswith("_calc") else "Safe"
        lines.append(f"| {obj['type']} | {obj['aliasOriginal']} | {display} | {obj['id']} | {obj['aliasRenamed']} | {category} |")

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
    """Save _tr file with current state using atomic write."""
    import tempfile
    # Write to temp file first
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.tmp') as f:
        json.dump(objects, f, indent=2, ensure_ascii=False)
        temp_path = f.name
    # Atomic rename
    import os
    os.replace(temp_path, tr_file)


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


def apply_renames(objects: list[dict], app_name: str, tr_file: Path = None) -> tuple[int, int]:
    """Apply aliasRenamed via API with progress tracking and state saving."""
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
        "MessageTemplate": "MessageTemplate",
        "Trigger": "Trigger",
        "Cart": "Cart",
        "OrgStructure": "OrgStructure",
    }

    success = 0
    failed = 0
    total = len(objects)
    batch_num = 0
    batch_size = 100

    for i, obj in enumerate(objects):
        if not obj.get("id") or not obj.get("aliasRenamed"):
            continue

        # Skip already successfully renamed
        if obj.get("renameStatus") == "success":
            success += 1
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

        # Update status in JSON
        obj["renameStatus"] = "success" if result.get("success") else "failed"
        obj["renameTimestamp"] = datetime.now().isoformat()

        if result.get("success"):
            success += 1
        else:
            failed += 1

        # Save state after each batch
        if tr_file and (i + 1) % batch_size == 0:
            save_tr_file(tr_file, objects)
            batch_num += 1
            print(f"  Batch {batch_num}: Progress: {i + 1}/{total} | Success: {success}, Failed: {failed}")
            print(f"  State saved. Run script again to continue if timeout occurs.")

        time.sleep(0.05)

    # Final save
    if tr_file:
        save_tr_file(tr_file, objects)
        print(f"  Final: {total}/{total} | Success: {success}, Failed: {failed}")

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
        description="CMW Platform Localization Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python localization.py --app MyApp
  python localization.py --app MyApp --step 5 --resume MyAlias
  python localization.py --app MyApp --apply-only
        """,
    )
    parser.add_argument("--app", required=True, help="Application system name")
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

    # Look for existing _tr file if resuming
    tr_dir = extract_dir / f"{app_name}_tr"
    existing_tr_files = list(tr_dir.glob("*_tr.json")) if tr_dir.exists() else []
    existing_tr_files.extend(list(base_dir.glob(f"*_{app_name}_tr.json")))
    existing_tr_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

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
                    suffix = suffix_map.get(obj["aliasOriginal"], "_sv")
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

    # Use existing _tr file if available, otherwise use new one
    active_tr_file = existing_tr_files[0] if existing_tr_files else tr_file

    if args.step <= 5 and active_tr_file.exists():
        print(f"\n[{ts}] Step 5: Interactive translation...")
        print(f"  Using: {active_tr_file}")
        objects, last_alias = load_tr_file(active_tr_file)
        resume_from = args.resume if args.resume else (last_alias if args.step == 5 else "")
        interactive_translate(active_tr_file, objects, resume_from)

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
        save_tr_file(active_tr_file, objects)
        generate_report(objects, ts, app_name, "JSON translated")

    if args.step <= 6 and active_tr_file.exists():
        print(f"\n[{ts}] Step 6: Applying renames...")
        objects, _ = load_tr_file(active_tr_file)

        # Filter: only objects with id and aliasRenamed
        pending = [o for o in objects if o.get("id") and o.get("aliasRenamed")]

        # Skip already successfully renamed (resume support)
        pending = [o for o in pending if o.get("renameStatus") != "success"]

        print(f"  Total: {len([o for o in objects if o.get('id') and o.get('aliasRenamed')])}")
        print(f"  Pending: {len(pending)}")

        success, failed = apply_renames(objects, app_name, active_tr_file)  # Pass FULL list
        print(f"  Applied: {success} success, {failed} failed")
        print(f"  State saved to: {active_tr_file}")

    if args.step <= 7:
        print(f"\n[{ts}] Step 7: Restart required")
        print("  Aliases have been renamed. Please restart the platform now.")
        print(f"  After restart, run: python localization.py --app {app_name} --step 8")
        return

    if args.step <= 8:
        print(f"\n[{ts}] Step 8: Re-exporting CTF...")
        ctf_path, ts = export_ctf(app_name)
        if ctf_path:
            extract_ctf(ctf_path, extract_dir)
            print(f"  Re-exported and extracted to: {extract_dir}")

    if args.step <= 9:
        print(f"\n[{ts}] Step 9: Fixing expressions...")
        objects, _ = load_tr_file(active_tr_file)
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