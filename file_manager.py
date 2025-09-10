"""
File Manager for Comindware Analyst Copilot
==================================

This module handles all HuggingFace file upload operations for the Comindware Analyst Copilot.
It provides a centralized way to manage file operations and can be easily
enabled/disabled via configuration.

Usage:
    from file_manager import file_manager
    
    # Upload file to HuggingFace repository
    success = file_manager.save_and_commit_file(
        file_path="results.csv",
        content="csv content",
        commit_message="Add results"
    )

Environment Variables:
    - HF_TOKEN: HuggingFace API token for file operations
    - FILE_UPLOAD_ENABLED: Set to "true" to enable file uploading (default: disabled)
"""

import os
import json
import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Global flag to enable/disable file upload functionality
FILE_UPLOAD_ENABLED = os.getenv("FILE_UPLOAD_ENABLED", "false").lower() == "true"

# Import huggingface_hub components for API-based file operations
try:
    from huggingface_hub import HfApi, CommitOperationAdd
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

class FileManager:
    """
    Manages file upload operations for the Comindware Analyst Copilot.
    
    This class provides a centralized interface for uploading files to HuggingFace
    repositories. It can be easily disabled by setting the FILE_UPLOAD_ENABLED
    environment variable to "false".
    """
    
    def __init__(self):
        """Initialize the file manager."""
        self.enabled = FILE_UPLOAD_ENABLED and HF_HUB_AVAILABLE
        if not self.enabled:
            if not FILE_UPLOAD_ENABLED:
                print("ℹ️ File uploading is disabled (set FILE_UPLOAD_ENABLED=true to enable)")
            elif not HF_HUB_AVAILABLE:
                print("ℹ️ File uploading is disabled (huggingface_hub not available)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the file manager."""
        return {
            "enabled": self.enabled,
            "file_upload_enabled": FILE_UPLOAD_ENABLED,
            "hf_hub_available": HF_HUB_AVAILABLE
        }
    
    def _get_hf_api_client(self, token: Optional[str] = None) -> Optional[HfApi]:
        """
        Get HuggingFace API client.
        
        Args:
            token (str, optional): HuggingFace token
            
        Returns:
            HfApi: API client or None if failed
        """
        if not self.enabled:
            return None
            
        try:
            # Use provided token or get from environment
            if not token:
                token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            
            if not token:
                print("Error: HuggingFace token not provided and HF_TOKEN/HUGGINGFACEHUB_API_TOKEN not set")
                return None
            
            return HfApi(token=token)
            
        except Exception as e:
            print(f"Error creating HfApi client: {e}")
            return None
    
    def save_and_commit_file(
        self,
        file_path: str,
        content: str,
        commit_message: str,
        repo_id: Optional[str] = None,
        token: Optional[str] = None
    ) -> bool:
        """
        Save content to a file and commit it to a HuggingFace repository.
        
        Args:
            file_path (str): Path where the file should be saved in the repository
            content (str): Content to write to the file
            commit_message (str): Git commit message
            repo_id (str, optional): Repository ID (e.g., "username/repo-name"). 
                                   If None, tries to get from environment
            token (str, optional): HuggingFace token
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            print("ℹ️ File uploading is disabled")
            return False
            
        try:
            # Get API client
            api = self._get_hf_api_client(token)
            if not api:
                return False
            
            # Get repository ID from environment if not provided
            if not repo_id:
                space_id = os.getenv("SPACE_ID")
                if space_id:
                    repo_id = f"spaces/{space_id}"
                else:
                    print("Error: No repository ID provided and SPACE_ID not set")
                    return False
            
            # Create commit operation
            operation = CommitOperationAdd(
                path_in_repo=file_path,
                path_or_fileobj=content.encode('utf-8')
            )
            
            # Commit to repository
            commit_info = api.create_commit(
                repo_id=repo_id,
                repo_type="space",  # Assuming this is for HuggingFace Spaces
                operations=[operation],
                commit_message=commit_message
            )
            
            print(f"✅ File uploaded successfully: {file_path}")
            print(f"   Repository: {repo_id}")
            print(f"   Commit: {commit_info.commit_url}")
            return True
            
        except Exception as e:
            print(f"❌ Error uploading file: {e}")
            return False
    
    def upload_file(
        self,
        local_file_path: str,
        remote_file_path: str,
        commit_message: str,
        repo_id: Optional[str] = None,
        token: Optional[str] = None
    ) -> bool:
        """
        Upload a local file to a HuggingFace repository.
        
        Args:
            local_file_path (str): Path to the local file to upload
            remote_file_path (str): Path where the file should be saved in the repository
            commit_message (str): Git commit message
            repo_id (str, optional): Repository ID
            token (str, optional): HuggingFace token
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            print("ℹ️ File uploading is disabled")
            return False
            
        try:
            # Read local file
            with open(local_file_path, 'rb') as f:
                content = f.read()
            
            # Upload using save_and_commit_file
            return self.save_and_commit_file(
                file_path=remote_file_path,
                content=content.decode('utf-8'),
                commit_message=commit_message,
                repo_id=repo_id,
                token=token
            )
            
        except Exception as e:
            print(f"❌ Error reading local file: {e}")
            return False

# Global file manager instance
file_manager = FileManager()

# Convenience functions for backward compatibility
def save_and_commit_file(
    file_path: str,
    content: str,
    commit_message: str,
    repo_id: Optional[str] = None,
    token: Optional[str] = None
) -> bool:
    """Save content to a file and commit it to a HuggingFace repository."""
    return file_manager.save_and_commit_file(file_path, content, commit_message, repo_id, token)

def upload_file(
    local_file_path: str,
    remote_file_path: str,
    commit_message: str,
    repo_id: Optional[str] = None,
    token: Optional[str] = None
) -> bool:
    """Upload a local file to a HuggingFace repository."""
    return file_manager.upload_file(local_file_path, remote_file_path, commit_message, repo_id, token)

def get_file_manager_status() -> Dict[str, Any]:
    """Get file manager status."""
    return file_manager.get_status()
