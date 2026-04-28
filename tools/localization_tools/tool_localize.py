# tool_localize.py - Localization workflow for system names (aliases) and display names
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests

try:
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field
except ImportError:
    from tools.tool_utils import tool
    from tools.models import BaseModel, Field


TYPE_FOLDER_MAPPING: dict[str, str] = {
    "RecordTemplate": "RecordTemplates",
    "ProcessTemplate": "ProcessTemplates",
    "RoleTemplate": "Roles",
    "AccountTemplate": "Accounts",
    "OrgStructureTemplate": "OrgStructure",
    "MessageTemplate": "MessageTemplates",
    "Workspace": "Workspaces",
    "Page": "Pages",
    "Attribute": "Attributes",
    "Dataset": "Datasets",
    "Toolbar": "Toolbars",
    "Form": "Forms",
    "UserCommand": "UserCommands",
    "Card": "Cards",
    "Cart": "Carts",
    "Trigger": "Triggers",
    "Role": "Roles",
    "WidgetConfig": "WidgetConfigs",
}

FOLDER_TYPE_MAPPING: dict[str, str] = {v: k for k, v in TYPE_FOLDER_MAPPING.items()}

TYPE_PREDICATE_MAPPING: dict[str, str] = {
    "RecordTemplate": "cmw.container.alias",
    "ProcessTemplate": "cmw.container.alias",
    "RoleTemplate": "cmw.container.alias",
    "AccountTemplate": "cmw.container.alias",
    "OrgStructureTemplate": "cmw.container.alias",
    "MessageTemplate": "cmw.message.type.alias",
    "Workspace": "cmw.alias",
    "Page": "cmw.desktopPage.alias",
    "Attribute": "cmw.object.alias",
    "Dataset": "cmw.alias",
    "Toolbar": "cmw.alias",
    "Form": "cmw.alias",
    "UserCommand": "cmw.alias",
    "Card": "cmw.alias",
    "Cart": "cmw.cart.alias",
    "Trigger": "cmw.trigger.alias",
    "Role": "cmw.role.alias",
    "WidgetConfig": "cmw.form.alias",
}


class LocalizeSchema(BaseModel):
    ctf_file_path: str | None = Field(
        default=None,
        description="Path to CTF file. If not provided, will export from application."
    )
    application_system_name: str | None = Field(
        default=None,
        description="Application system name for export (required if ctf_file_path not provided)"
    )
    json_folder: str = Field(
        description="Path to folder containing JSON files to analyze"
    )
    dangerous_suffix: str = Field(
        default="_calc",
        description="Suffix for dangerous system names (mentioned outside alias)"
    )
    safe_suffix: str = Field(
        default="_sv",
        description="Suffix for safe system names (only in alias)"
    )
    dry_run: bool = Field(
        default=True,
        description="If True, only analyze without making changes"
    )
    collect_aliases: bool = Field(
        default=True,
        description="If True, collect alias data (system names)"
    )
    collect_display_names: bool = Field(
        default=True,
        description="If True, collect display name data (Name property)"
    )


def collect_aliases_from_json_folder(
    folder_path: str,
    collect_aliases: bool = True,
    collect_display_names: bool = True
) -> list[dict[str, Any]]:
    """
    Collect aliases and/or display names from JSON folder.

    Args:
        folder_path: Path to CTF JSON folder
        collect_aliases: If True, collect alias data (system names)
        collect_display_names: If True, collect display name data (Name property)

    Returns:
        List of objects with collected data
    """
    results = []
    path = Path(folder_path)

    for json_file in path.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        obj_type = None
        for ftype, fname in TYPE_FOLDER_MAPPING.items():
            if fname in json_file.parts:
                obj_type = ftype
                break

        relative_path = str(json_file.relative_to(path))

        # Collect Alias fields (skip system prefixes)
        if collect_aliases:
            for match in re.finditer(r'"Alias"\s*:\s*"([^"]+)"', content):
                alias = match.group(1)
                if alias.startswith(("cmw.", "oa.", "pa.", "msgt.", "aa.", "ra.", "os.")):
                    continue
                results.append({
                    "alias": alias,
                    "type": obj_type,
                    "json_path": relative_path + "#$.GlobalAlias.Alias",
                    "source": "alias",
                })

        # Collect Name fields (display names) - skip if looks like system alias
        if collect_display_names:
            for match in re.finditer(r'"Name"\s*:\s*"([^"]+)"', content):
                name = match.group(1)
                if name.startswith(("oa.", "pa.", "msgt.", "aa.", "ra.", "os.", "form.", "tb.", "lst.", "event.", "card.", "trigger.", "workspace.")):
                    continue
                if len(name) < 2:
                    continue
                results.append({
                    "alias": name,
                    "type": obj_type,
                    "json_path": relative_path + "#$.Name",
                    "source": "name",
                })

    return results


EXPRESSION_KEYS = {"Expression", "Code", "ValueExpression", "ValidationScript"}

def check_aliases_in_json_folder(folder_path: str, aliases: set[str]) -> dict[str, bool]:
    mentions = {alias: False for alias in aliases}
    path = Path(folder_path)

    for json_file in path.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        for alias in aliases:
            if mentions[alias]:
                continue

            safe_alias = re.escape(alias)

            for key in EXPRESSION_KEYS:
                expression_pattern = rf'"{key}"\s*:\s*"[^"]*{re.escape(alias)}[^"]*"'
                if re.search(expression_pattern, content):
                    mentions[alias] = True
                    break


def get_config() -> dict[str, Any]:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return {
            "base_url": os.environ.get("CMW_BASE_URL", "").strip(),
            "login": os.environ.get("CMW_LOGIN", "").strip(),
            "password": os.environ.get("CMW_PASSWORD", "").strip(),
            "timeout": int(os.environ.get("CMW_TIMEOUT", "30").strip()),
        }
    except Exception:
        return {
            "base_url": "",
            "login": "",
            "password": "",
            "timeout": 30,
        }


def get_headers() -> dict[str, str]:
    import base64
    cfg = get_config()
    credentials = base64.b64encode(f"{cfg['login']}:{cfg['password']}".encode("ascii")).decode("ascii")
    return {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


@tool("localize_aliases", return_direct=False, args_schema=LocalizeSchema)
def localize_aliases(
    ctf_file_path: str | None = None,
    application_system_name: str | None = None,
    json_folder: str = "",
    dangerous_suffix: str = "_calc",
    safe_suffix: str = "_sv",
    dry_run: bool = True,
    collect_aliases: bool = True,
    collect_display_names: bool = True,
) -> dict[str, Any]:
    """
    Localization workflow for system names (aliases) and display names.

    Collects aliases and/or display names from CTF JSON files, verifies them,
    and prepares localization data according to schema.json structure.

    Collection is optional:
    - collect_aliases=True: Collect and track alias data (system names)
    - collect_display_names=True: Collect and track displayName data (Name property)
    - Both can be enabled/disabled independently

    Workflow integration:
    - Aliases are applied via API (OntologyService/AddStatement)
    - DisplayNames are applied via CTF import (not API)

    Workflow:
    1. Export CTF if not provided
    2. Collect aliases and/or display names from JSON files with their types
    3. Verify aliases using get_ontology_objects tool (if collect_aliases=True)
    4. Analyze aliases for dangerous mentions outside 'alias' field
    5. Suggest suffixes for renaming
    6. Rename aliases using update_object_property tool (if not dry_run)
    7. Update JSON files with new aliases

    Args:
        ctf_file_path: Path to CTF file (optional, will export if not provided)
        application_system_name: Application system name (required if ctf not provided)
        json_folder: Path to folder with JSON files
        dangerous_suffix: Suffix for dangerous aliases (default: _calc)
        safe_suffix: Suffix for safe aliases (default: _sv)
        dry_run: If True, only analyze without changes (default: True)
        collect_aliases: If True, collect alias data (default: True)
        collect_display_names: If True, collect displayName data (default: True)

    Returns:
        dict with analysis results and rename report
    """
    results = {
        "success": True,
        "ctf_exported": False,
        "aliases_collected": 0,
        "display_names_collected": 0,
        "aliases_verified": 0,
        "aliases_missing": [],
        "dangerous_aliases": [],
        "safe_aliases": [],
        "renamed_aliases": [],
        "json_updated": 0,
        "errors": [],
        "collect_aliases": collect_aliases,
        "collect_display_names": collect_display_names,
    }

    if not ctf_file_path and application_system_name:
        try:
            from tools.transfer_tools.tool_export_application import export_application
            export_result = export_application.invoke({
                "application_system_name": application_system_name,
                "save_to_file": True,
            })
            if export_result.get("success"):
                ctf_file_path = export_result.get("ctf_file_path")
                results["ctf_exported"] = True
                results["ctf_file_path"] = ctf_file_path
        except Exception as e:
            results["errors"].append(f"CTF export failed: {e}")

    results["aliases"] = collect_aliases_from_json_folder(
        json_folder,
        collect_aliases=collect_aliases,
        collect_display_names=collect_display_names
    )

    # Count collected items by source
    alias_items = [item for item in results["aliases"] if item.get("source") == "alias"]
    name_items = [item for item in results["aliases"] if item.get("source") == "name"]

    results["aliases_collected"] = len(alias_items)
    results["display_names_collected"] = len(name_items)

    aliases_by_type: dict[str, set[str]] = {}
    for item in results["aliases"]:
        # Only process alias items for verification if collect_aliases is True
        if item.get("source") == "alias" and collect_aliases:
            obj_type = item["type"] or "Unknown"
            if obj_type not in aliases_by_type:
                aliases_by_type[obj_type] = set()
            aliases_by_type[obj_type].add(item["alias"])

    verified_aliases: dict[str, str] = {}

    # Only verify aliases if collect_aliases is True
    if collect_aliases:
        cfg = get_config()
        base_url = cfg["base_url"].rstrip("/")
        headers = get_headers()

        for obj_type, type_aliases in aliases_by_type.items():
            if obj_type not in TYPE_PREDICATE_MAPPING:
                continue

            predicate = TYPE_PREDICATE_MAPPING[obj_type]
            endpoint = f"{base_url}/api/public/system/Base/OntologyService/GetWithMultipleValues"

            request_body = {
                "predicate": predicate,
                "min": 1,
                "max": 10000,
            }

            try:
                resp = requests.post(endpoint, headers=headers, json=request_body, timeout=cfg["timeout"])
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                key = item.get("key", "")
                                value = item.get("value", [])
                                if value and isinstance(value, list):
                                    system_name = value[0]
                                else:
                                    match = re.match(r"^([\w.]+)\s*:\s*System\.String\[\]", key)
                                    system_name = match.group(1) if match else key

                                if system_name in type_aliases:
                                    clean_id = re.match(r"^([\w.]+)\s*:\s*System\.String\[\]", key)
                                    obj_id = clean_id.group(1) if clean_id else key
                                    verified_aliases[system_name] = obj_id
            except Exception as e:
                results["errors"].append(f"Verification failed for {obj_type}: {e}")

    results["verified_aliases"] = verified_aliases
    results["aliases_verified"] = len(verified_aliases)

    missing = []
    for obj_type, type_aliases in aliases_by_type.items():
        for alias in type_aliases:
            if alias not in verified_aliases:
                missing.append({"alias": alias, "type": obj_type})

    results["aliases_missing"] = missing

    all_aliases = set()
    for type_aliases in aliases_by_type.values():
        all_aliases.update(type_aliases)

    mentions = check_aliases_in_json_folder(json_folder, all_aliases)

    dangerous = []
    safe = []

    for item in results["aliases"]:
        alias = item["alias"]
        if alias in mentions and mentions[alias]:
            dangerous.append({
                "alias": alias,
                "type": item["type"],
                "new_alias": alias + dangerous_suffix,
                "mentions_outside": True,
            })
        else:
            safe.append({
                "alias": alias,
                "type": item["type"],
                "new_alias": alias + safe_suffix,
                "mentions_outside": False,
            })

    results["dangerous_aliases"] = dangerous
    results["safe_aliases"] = safe
    results["dangerous_count"] = len(dangerous)
    results["safe_count"] = len(safe)

    if not dry_run:
        endpoint = f"{base_url}/api/public/system/Base/OntologyService/AddStatement"

        renamed = []
        for item in dangerous + safe:
            alias = item["alias"]
            new_alias = item["new_alias"]
            obj_type = item["type"]

            if obj_type in TYPE_PREDICATE_MAPPING and alias in verified_aliases:
                obj_id = verified_aliases[alias]
                predicate = TYPE_PREDICATE_MAPPING[obj_type]

                request_body = {
                    "subject": obj_id,
                    "predicate": predicate,
                    "value": new_alias,
                }

                try:
                    resp = requests.post(endpoint, headers=headers, json=request_body, timeout=cfg["timeout"])
                    renamed.append({
                        "original": alias,
                        "new": new_alias,
                        "id": obj_id,
                        "success": resp.status_code == 200,
                    })
                except Exception as e:
                    renamed.append({
                        "original": alias,
                        "new": new_alias,
                        "id": obj_id,
                        "success": False,
                        "error": str(e),
                    })

        results["renamed_aliases"] = renamed

        new_ctf_path = None
        if renamed:
            try:
                from tools.transfer_tools.tool_export_application import export_application
                export_result = export_application.invoke({
                    "application_system_name": application_system_name,
                    "save_to_file": True,
                })
                if export_result.get("success"):
                    new_ctf_path = export_result.get("ctf_file_path")
                    results["new_ctf_path"] = new_ctf_path
            except Exception as e:
                results["errors"].append(f"Export new CTF failed: {e}")

        json_updated = 0
        for alias, new_alias in [(d["alias"], d["new_alias"]) for d in dangerous]:
            safe_alias = re.escape(alias)
            patterns = [
                r'"\s*' + safe_alias + r'\s*"',
                r'\$\{' + safe_alias + r'\}',
                r'->\{' + safe_alias + r'\}',
            ]

            for json_file in Path(json_folder).rglob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    original_content = content
                    for pattern in patterns:
                        content = re.sub(pattern, '"' + new_alias + '"', content)

                    if content != original_content:
                        with open(json_file, "w", encoding="utf-8") as f:
                            f.write(content)
                        json_updated += 1
                except Exception:
                    continue

        results["json_updated"] = json_updated

        results["import_ctf_note"] = (
            f"Renamed {len(renamed)} aliases, updated {json_updated} JSON files. "
            "Import the new CTF to apply changes to the platform."
        )
        if new_ctf_path:
            results["import_command"] = (
                f'import_application.invoke({{"application_system_name": "{application_system_name}", "ctf_file_path": "{new_ctf_path}", "update_existing": true}})'
            )
        else:
            results["import_command"] = "Export new CTF failed, cannot provide import command."

    return results


if __name__ == "__main__":
    result = localize_aliases.invoke({
        "application_system_name": "supportTest",
        "json_folder": "/tmp/cmw-workspaces",
        "dangerous_suffix": "_calc",
        "safe_suffix": "_sv",
        "dry_run": True,
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))