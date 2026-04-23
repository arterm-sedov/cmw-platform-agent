# transfer_utils.py - Utility functions for CTF file operations
import base64
import os
from pathlib import Path
import tempfile
from typing import Optional
import uuid


class TransferUtils:
    """Utilities for CTF file operations."""

    CTF_EXTENSION = ".ctf"

    @staticmethod
    def save_ctf(ctf_data: str, application_name: str) -> str:
        """
        Save Base64-encoded CTF data to a file.

        Args:
            ctf_data: Base64-encoded CTF data.
            application_name: Application system name for filename.

        Returns:
            Path to the saved CTF file.
        """
        ctf_bytes = base64.b64decode(ctf_data)

        output_dir = os.path.join(tempfile.gettempdir(), "cmw-transfer")
        os.makedirs(output_dir, exist_ok=True)

        safe_name = "".join(c for c in application_name if c.isalnum() or c in "-_")
        timestamp = uuid.uuid4().hex[:8]
        filename = f"{safe_name}_{timestamp}{TransferUtils.CTF_EXTENSION}"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "wb") as f:
            f.write(ctf_bytes)

        return file_path

    @staticmethod
    def read_ctf_from_file(file_path: str) -> str | None:
        """
        Read CTF data from a file and return as Base64 string.

        Args:
            file_path: Path to the CTF file.

        Returns:
            Base64-encoded CTF data, or None if file not found.
        """
        if not os.path.exists(file_path):
            return None

        with open(file_path, "rb") as f:
            ctf_bytes = f.read()

        return base64.b64encode(ctf_bytes).decode("utf-8")

    @staticmethod
    def decode_ctf_to_bytes(ctf_data: str) -> bytes:
        """
        Decode Base64 CTF data to bytes.

        Args:
            ctf_data: Base64-encoded CTF data.

        Returns:
            Raw CTF bytes.
        """
        return base64.b64decode(ctf_data)
