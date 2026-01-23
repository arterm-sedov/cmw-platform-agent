#!/usr/bin/env python3
"""
Dead Code Removal Script
=======================

This script helps remove the dead code that has been extracted to this directory.
It provides functions to safely remove dead code from the main codebase.

Usage:
    python remove_dead_code.py --dry-run    # Show what would be removed
    python remove_dead_code.py --remove     # Actually remove the dead code
"""

import os
import re
import argparse
from pathlib import Path

class DeadCodeRemover:
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.agent_ng_path = self.project_root / "agent_ng"

    def remove_reset_functions(self, dry_run: bool = True):
        """Remove reset/cleanup functions from the codebase"""
        files_to_update = [
            "langchain_agent.py",
            "llm_manager.py", 
            "stats_manager.py",
            "response_processor.py",
            "token_counter.py",
            "trace_manager.py",
            "debug_streamer.py"
        ]

        functions_to_remove = [
            "reset_agent_ng",
            "reset_langchain_agent", 
            "reset_llm_manager",
            "reset_stats_manager",
            "reset_response_processor",
            "reset_token_tracker",
            "cleanup_debug_system"
        ]

        for filename in files_to_update:
            filepath = self.agent_ng_path / filename
            if filepath.exists():
                self._remove_functions_from_file(filepath, functions_to_remove, dry_run)

    def remove_debug_functions(self, dry_run: bool = True):
        """Remove debug/utility functions from the codebase"""
        files_to_update = [
            "response_processor.py",
            "message_processor.py",
            "debug_streamer.py",
            "trace_manager.py"
        ]

        functions_to_remove = [
            "print_message_components",
            "_print_attribute",
            "get_recent_logs",
            "trace_prints_with_context",
            "trace_prints",
            "get_trace_history"
        ]

        classes_to_remove = [
            "Tee",
            "_SinkWriter"
        ]

        for filename in files_to_update:
            filepath = self.agent_ng_path / filename
            if filepath.exists():
                self._remove_functions_from_file(filepath, functions_to_remove, dry_run)
                self._remove_classes_from_file(filepath, classes_to_remove, dry_run)

    def remove_config_functions(self, dry_run: bool = True):
        """Remove configuration functions from the codebase"""
        files_to_update = [
            "agent_config.py",
            "langsmith_config.py", 
            "langfuse_config.py"
        ]

        functions_to_remove = [
            "print_config",
            "get_openai_wrapper",
            "get_langfuse_callback_handler"
        ]

        for filename in files_to_update:
            filepath = self.agent_ng_path / filename
            if filepath.exists():
                self._remove_functions_from_file(filepath, functions_to_remove, dry_run)

    def remove_app_functions(self, dry_run: bool = True):
        """Remove app functions from the codebase"""
        filepath = self.agent_ng_path / "app_ng_modular.py"
        if filepath.exists():
            functions_to_remove = [
                "detect_language_from_url",
                "create_safe_demo", 
                "reload_demo"
            ]
            self._remove_functions_from_file(filepath, functions_to_remove, dry_run)

    def remove_stats_functions(self, dry_run: bool = True):
        """Remove stats/export functions from the codebase"""
        filepath = self.agent_ng_path / "stats_manager.py"
        if filepath.exists():
            functions_to_remove = [
                "export_stats",
                "get_stats_summary",
                "get_performance_metrics",
                "get_error_summary"
            ]
            self._remove_functions_from_file(filepath, functions_to_remove, dry_run)

    def remove_test_files(self, dry_run: bool = True):
        """Remove standalone test files"""
        files_to_remove = [
            "agent_ng/_tests/test_vector_store.py",
            "agent_ng/_tests/debug_tools.py",
            "agent_ng/error_coverage_analyzer.py"
        ]

        for file_path in files_to_remove:
            full_path = self.project_root / file_path
            if full_path.exists():
                if dry_run:
                    print(f"Would remove: {full_path}")
                else:
                    full_path.unlink()
                    print(f"Removed: {full_path}")

    def remove_unused_imports(self, dry_run: bool = True):
        """Remove unused imports from the codebase"""
        # Remove convert_messages_for_mistral from utils.py
        utils_file = self.agent_ng_path / "utils.py"
        if utils_file.exists():
            self._remove_unused_imports_from_file(utils_file, dry_run)

        # Remove asdict from app_ng_modular.py
        app_file = self.agent_ng_path / "app_ng_modular.py"
        if app_file.exists():
            self._remove_unused_imports_from_file(app_file, dry_run)

    def _remove_functions_from_file(self, filepath: Path, functions: list, dry_run: bool):
        """Remove specific functions from a file"""
        if not filepath.exists():
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        for func_name in functions:
            # Pattern to match function definition and its body
            pattern = rf'^def {re.escape(func_name)}\(.*?\):.*?(?=^def |^class |^# |^$|\Z)'
            content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)

            # Also remove any aliases
            pattern = rf'^{re.escape(func_name)} = .*?$'
            content = re.sub(pattern, '', content, flags=re.MULTILINE)

        if content != original_content:
            if dry_run:
                print(f"Would update: {filepath}")
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Updated: {filepath}")

    def _remove_classes_from_file(self, filepath: Path, classes: list, dry_run: bool):
        """Remove specific classes from a file"""
        if not filepath.exists():
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        for class_name in classes:
            # Pattern to match class definition and its body
            pattern = rf'^class {re.escape(class_name)}.*?:.*?(?=^def |^class |^# |^$|\Z)'
            content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)

        if content != original_content:
            if dry_run:
                print(f"Would update: {filepath}")
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Updated: {filepath}")

    def _remove_unused_imports_from_file(self, filepath: Path, dry_run: bool):
        """Remove unused imports from a file"""
        if not filepath.exists():
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Remove convert_messages_for_mistral import
        if 'convert_messages_for_mistral' in content:
            content = re.sub(r'from \.provider_adapters import convert_messages_for_mistral\n', '', content)
            content = re.sub(r'convert_messages_for_mistral = None\n', '', content)

        # Remove asdict import
        if 'asdict' in content:
            content = re.sub(r'from dataclasses import asdict\n', 'from dataclasses import dataclass\n', content)

        if content != original_content:
            if dry_run:
                print(f"Would update imports in: {filepath}")
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Updated imports in: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Remove dead code from the codebase')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without actually removing')
    parser.add_argument('--remove', action='store_true', help='Actually remove the dead code')
    parser.add_argument('--project-root', help='Path to project root directory')

    args = parser.parse_args()

    if not args.dry_run and not args.remove:
        print("Please specify either --dry-run or --remove")
        return

    remover = DeadCodeRemover(args.project_root)
    dry_run = args.dry_run

    print("🧹 Dead Code Removal Script")
    print("=" * 50)

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    else:
        print("⚠️  REMOVAL MODE - Files will be modified")

    print("\n1. Removing reset/cleanup functions...")
    remover.remove_reset_functions(dry_run)

    print("\n2. Removing debug/utility functions...")
    remover.remove_debug_functions(dry_run)

    print("\n3. Removing configuration functions...")
    remover.remove_config_functions(dry_run)

    print("\n4. Removing app functions...")
    remover.remove_app_functions(dry_run)

    print("\n5. Removing stats/export functions...")
    remover.remove_stats_functions(dry_run)

    print("\n6. Removing test files...")
    remover.remove_test_files(dry_run)

    print("\n7. Removing unused imports...")
    remover.remove_unused_imports(dry_run)

    print("\n✅ Dead code removal complete!")

if __name__ == "__main__":
    main()
