"""
Dataset Manager for Comindware Analyst Copilot
=====================================

This module handles all dataset uploading functionality for the Comindware Analyst Copilot.
It provides a centralized way to manage dataset operations and can be easily
enabled/disabled via configuration.

Usage:
    from dataset_manager import dataset_manager
    
    # Upload initialization summary
    success = dataset_manager.upload_init_summary(init_data)
    
    # Upload run data
    success = dataset_manager.upload_run_data(run_data)

Environment Variables:
    - HF_TOKEN: HuggingFace API token for dataset operations
    - DATASET_ENABLED: Set to "true" to enable dataset uploading (default: disabled)
"""

import os
import json
import datetime
from typing import Optional, Union, Dict, Any, List
from pathlib import Path

# Global flag to enable/disable dataset functionality
DATASET_ENABLED = os.getenv("DATASET_ENABLED", "false").lower() == "true"

# Dataset constants
DATASET_ID = "arterm-sedov/agent-course-final-assignment"
DATASET_CONFIG_PATH = "dataset_config.json"

# Import huggingface_hub components for API-based file operations
try:
    from huggingface_hub import HfApi, CommitOperationAdd
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")


class DatasetManager:
    """
    Manages dataset uploading operations for the Comindware Analyst Copilot.
    
    This class provides a centralized interface for uploading initialization
    summaries and run data to HuggingFace datasets. It can be easily disabled
    by setting the DATASET_ENABLED environment variable to "false".
    """
    
    def __init__(self):
        """Initialize the dataset manager."""
        self.enabled = DATASET_ENABLED and HF_HUB_AVAILABLE
        if not self.enabled:
            if not DATASET_ENABLED:
                print("ℹ️ Dataset uploading is disabled (set DATASET_ENABLED=true to enable)")
            elif not HF_HUB_AVAILABLE:
                print("ℹ️ Dataset uploading is disabled (huggingface_hub not available)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the dataset manager."""
        return {
            "enabled": self.enabled,
            "dataset_enabled": DATASET_ENABLED,
            "hf_hub_available": HF_HUB_AVAILABLE,
            "dataset_id": DATASET_ID
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
                token = os.getenv("HF_TOKEN")
            
            if not token:
                print("Error: HuggingFace token not provided and HF_TOKEN not set")
                return None
            
            return HfApi(token=token)
            
        except Exception as e:
            print(f"Error creating HfApi client: {e}")
            return None
    
    def _load_dataset_schema(self) -> Optional[Dict]:
        """
        Load dataset schema from local dataset_config.json file.
        Tries multiple possible locations for robustness.
        """
        if not self.enabled:
            return None
            
        possible_paths = [
            Path("dataset_config.json"),  # Current working directory (root)
            Path("./dataset_config.json"),
            Path("../dataset_config.json"),  # Parent directory (if run from misc_files)
            Path(__file__).parent / "dataset_config.json",
            Path(__file__).parent.parent / "dataset_config.json"
        ]
        for path in possible_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        print("Warning: Dataset config file not found: dataset_config.json")
        return None
    
    def _get_dataset_features(self, split: str) -> Optional[Dict]:
        """
        Get features schema for a specific dataset split.
        
        Args:
            split (str): Dataset split name (init or runs)
            
        Returns:
            Dict: Features schema for the split or None if not found
        """
        if not self.enabled:
            return None
            
        schema = self._load_dataset_schema()
        if schema and "features" in schema and split in schema["features"]:
            features = schema["features"][split]
            print(f"🔍 Loaded schema for {split}: {list(features.keys())}")
            return features
        print(f"❌ No schema found for {split}")
        return None
    
    def _validate_data_structure(self, data: Dict, split: str) -> bool:
        """
        Validate that data matches the expected schema for the split.
        
        Args:
            data (Dict): Data to validate
            split (str): Dataset split name
            
        Returns:
            bool: True if data structure is valid
        """
        if not self.enabled:
            return True
            
        features = self._get_dataset_features(split)
        if not features:
            print(f"Warning: No schema found for split '{split}', skipping validation")
            return True
            
        # Debug: Print what we're checking
        print(f"🔍 Validating {split} split:")
        print(f"   Expected fields: {list(features.keys())}")
        print(f"   Actual fields: {list(data.keys())}")
            
        # Check that all required fields are present
        required_fields = set(features.keys())
        data_fields = set(data.keys())
        
        missing_fields = required_fields - data_fields
        if missing_fields:
            print(f"Warning: Missing required fields for {split} split: {missing_fields}")
            return False
        
        # Enhanced validation: Check nullable fields and data types
        for field_name, field_spec in features.items():
            if field_name in data:
                value = data[field_name]
                
                # Check nullable fields
                is_nullable = field_spec.get("nullable", False)
                if value is None and not is_nullable:
                    print(f"Warning: Field '{field_name}' is not nullable but contains None")
                    return False
                
                # Check data types for non-null values
                if value is not None:
                    expected_type = field_spec.get("dtype", "string")
                    if expected_type == "float64" and not isinstance(value, (int, float)):
                        print(f"Warning: Field '{field_name}' should be numeric but got {type(value)}")
                        return False
                    elif expected_type == "int64" and not isinstance(value, int):
                        print(f"Warning: Field '{field_name}' should be integer but got {type(value)}")
                        return False
        
        return True
    
    def upload_to_dataset(
        self,
        dataset_id: str,
        data: Union[Dict, List[Dict]],
        split: str = "train",
        token: Optional[str] = None
    ) -> bool:
        """
        Upload structured data to HuggingFace dataset.
        
        Args:
            dataset_id (str): Dataset repository ID (e.g., "username/dataset-name")
            data (Union[Dict, List[Dict]]): Data to upload (single dict or list of dicts)
            split (str): Dataset split name (default: "train")
            token (str, optional): HuggingFace token
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            print("ℹ️ Dataset uploading is disabled")
            return False
            
        try:
            # Get API client
            api = self._get_hf_api_client(token)
            if not api:
                return False
                
            # Prepare data as list
            if isinstance(data, dict):
                data_list = [data]
            else:
                data_list = data
                
            # Validate data structure against local schema only
            # Note: HuggingFace may show warnings about remote schema mismatch, but uploads still work
            for i, item in enumerate(data_list):
                if not self._validate_data_structure(item, split):
                    print(f"Warning: Data item {i} does not match local schema for split '{split}'")
                    # Continue anyway, but log the warning
                
            # Convert to JSONL format with proper serialization
            jsonl_content = ""
            for item in data_list:
                # Ensure all complex objects are serialized as strings
                serialized_item = {}
                for key, value in item.items():
                    if isinstance(value, (dict, list)):
                        serialized_item[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        serialized_item[key] = value
                jsonl_content += json.dumps(serialized_item, ensure_ascii=False) + "\n"
                
            # Create file path for dataset
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{split}-{timestamp}.jsonl"
            
            # Upload to dataset
            operation = CommitOperationAdd(
                path_in_repo=file_path,
                path_or_fileobj=jsonl_content.encode('utf-8')
            )
            
            commit_message = f"Add {split} data at {timestamp}"
            
            # Commit to dataset repository
            commit_info = api.create_commit(
                repo_id=dataset_id,
                repo_type="dataset",
                operations=[operation],
                commit_message=commit_message
            )
            
            print(f"✅ Data uploaded to dataset: {dataset_id}")
            print(f"   File: {file_path}")
            print(f"   Records: {len(data_list)}")
            return True
            
        except Exception as e:
            print(f"❌ Error uploading to dataset: {e}")
            return False
    
    def upload_init_summary(
        self,
        init_data: Dict,
        token: Optional[str] = None
    ) -> bool:
        """
        Upload agent initialization summary to init split.
        
        Args:
            init_data (Dict): Initialization data including LLM config, model status, etc.
            token (str, optional): HuggingFace token
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.upload_to_dataset(DATASET_ID, init_data, "init", token)
    
    def upload_run_data(
        self,
        run_data: Dict,
        split: str = "runs_new",
        token: Optional[str] = None
    ) -> bool:
        """
        Upload evaluation run data to specified split.
        
        Args:
            run_data (Dict): Evaluation run data including results, stats, etc.
            split (str): Dataset split name (default: "runs_new" for current schema)
            token (str, optional): HuggingFace token
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.upload_to_dataset(DATASET_ID, run_data, split, token)
    
    def get_dataset_info(self) -> Optional[Dict]:
        """
        Get information about the dataset.
        
        Returns:
            Dict: Dataset information or None if failed
        """
        if not self.enabled:
            return None
            
        try:
            api = self._get_hf_api_client()
            if not api:
                return None
                
            # Get dataset info
            dataset_info = api.dataset_info(repo_id=DATASET_ID)
            return {
                "id": dataset_info.id,
                "author": dataset_info.author,
                "last_modified": dataset_info.last_modified,
                "tags": dataset_info.tags,
                "downloads": dataset_info.downloads,
                "likes": dataset_info.likes
            }
            
        except Exception as e:
            print(f"Error getting dataset info: {e}")
            return None


# Global dataset manager instance
dataset_manager = DatasetManager()

# Convenience functions for backward compatibility
def upload_init_summary(init_data: Dict, token: Optional[str] = None) -> bool:
    """Upload agent initialization summary to init split."""
    return dataset_manager.upload_init_summary(init_data, token)

def upload_run_data(run_data: Dict, split: str = "runs_new", token: Optional[str] = None) -> bool:
    """Upload evaluation run data to specified split."""
    return dataset_manager.upload_run_data(run_data, split, token)

def upload_to_dataset(dataset_id: str, data: Union[Dict, List[Dict]], split: str = "train", token: Optional[str] = None) -> bool:
    """Upload structured data to HuggingFace dataset."""
    return dataset_manager.upload_to_dataset(dataset_id, data, split, token)

def get_dataset_status() -> Dict[str, Any]:
    """Get dataset manager status."""
    return dataset_manager.get_status()
