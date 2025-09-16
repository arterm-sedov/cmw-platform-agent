#!/usr/bin/env python3
"""
Script to fix tools.requests_ import issues in all tool files
"""

import os
import re
from pathlib import Path

def fix_file(file_path):
    """Fix tools.requests_ references in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace tools.requests_ with requests_
        original_content = content
        content = re.sub(r'tools\.requests_', 'requests_', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
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
        if fix_file(py_file):
            fixed_count += 1
    
    print(f"\n🎉 Fixed {fixed_count} files")

if __name__ == "__main__":
    main()
