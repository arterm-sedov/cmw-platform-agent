#!/usr/bin/env python3
"""
Pack folder back to CTF (zip archive).

Usage:
    python pack_ctf.py --input C:\\tmp\\cmw-transfer\\Volga_ctf --output C:\\tmp\\cmw-transfer\\Volga_tr_renamed.ctf
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path


def main(input_dir: str, output_ctf: str) -> int:
    """Pack folder to CTF archive."""
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    try:
        file_count = 0
        with zipfile.ZipFile(output_ctf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, input_dir)
                    zf.write(file_path, arcname)
                    file_count += 1
    except Exception as e:
        print(f"Error: Failed to pack: {e}")
        return 1

    print(f"Packed {file_count} files into: {output_ctf}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack folder to CTF archive")
    parser.add_argument("--input", required=True, help="Path to input directory")
    parser.add_argument("--output", required=True, help="Path to output CTF file")
    args = parser.parse_args()
    sys.exit(main(args.input, args.output))