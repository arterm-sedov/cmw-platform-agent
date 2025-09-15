#!/usr/bin/env python3
"""
Script to fix all import issues in tool files
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix import issues in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Fix direct requests_ import
        if 'import requests_ as requests_' in content:
            # Determine the correct import based on file location
            if 'applications_tools' in str(file_path):
                content = re.sub(r'import requests_ as requests_', 'from .. import requests_ as requests_', content)
                changes_made.append("Fixed applications_tools import")
            elif 'templates_tools' in str(file_path):
                content = re.sub(r'import requests_ as requests_', 'from .. import requests_ as requests_', content)
                changes_made.append("Fixed templates_tools import")
            elif 'attributes_tools' in str(file_path):
                content = re.sub(r'import requests_ as requests_', 'from .. import requests_ as requests_', content)
                changes_made.append("Fixed attributes_tools import")
            elif file_path.name == 'tool_utils.py':
                content = re.sub(r'import requests_ as requests_', 'from . import requests_ as requests_', content)
                changes_made.append("Fixed tool_utils import")
        
        # Fix tools.models import
        if 'from tools.models import' in content:
            if 'applications_tools' in str(file_path) or 'templates_tools' in str(file_path) or 'attributes_tools' in str(file_path):
                content = re.sub(r'from tools\.models import', 'from ...models import', content)
                changes_made.append("Fixed models import")
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            for change in changes_made:
                print(f"   - {change}")
            return True
        else:
            print(f"⏭️  No changes needed: {file_path}")
            return False
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all tool files"""
    tools_dir = Path("tools")
    fixed_count = 0
    
    # Find all Python files in tools directory
    for py_file in tools_dir.rglob("*.py"):
        if py_file.name != "__init__.py" and py_file.name != "requests_.py" and py_file.name != "tools.py":
            if fix_imports_in_file(py_file):
                fixed_count += 1
    
    print(f"\n🎉 Fixed {fixed_count} files")

if __name__ == "__main__":
    main()
