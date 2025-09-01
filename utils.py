import os
import datetime
import json
from typing import Optional, Union, Dict, Any, List
from pathlib import Path

# Global constants
TRACES_DIR = "traces"  # Directory for uploading trace files (won't trigger Space restarts)

# Dataset constants
DATASET_ID = "arterm-sedov/agent-course-final-assignment"
DATASET_CONFIG_PATH = "dataset_config.json"  # Local copy of dataset config

# Import huggingface_hub components for API-based file operations
try:
    from huggingface_hub import HfApi, CommitOperationAdd
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

def load_dataset_schema() -> Optional[Dict]:
    """
    Load dataset schema from local dataset_config.json file.
    Tries multiple possible locations for robustness.
    """
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

def get_dataset_features(split: str) -> Optional[Dict]:
    """
    Get features schema for a specific dataset split.
    
    Args:
        split (str): Dataset split name (init or runs)
        
    Returns:
        Dict: Features schema for the split or None if not found
    """
    schema = load_dataset_schema()
    if schema and "features" in schema and split in schema["features"]:
        features = schema["features"][split]
        print(f"🔍 Loaded schema for {split}: {list(features.keys())}")
        return features
    print(f"❌ No schema found for {split}")
    return None

def validate_data_structure(data: Dict, split: str) -> bool:
    """
    Validate that data matches the expected schema for the split.
    
    Args:
        data (Dict): Data to validate
        split (str): Dataset split name
        
    Returns:
        bool: True if data structure is valid
    """
    features = get_dataset_features(split)
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
                expected_dtype = field_spec.get("dtype", "string")
                if expected_dtype == "float64" and not isinstance(value, (int, float)):
                    print(f"Warning: Field '{field_name}' should be float64 but got {type(value)}")
                    return False
                elif expected_dtype == "int64" and not isinstance(value, int):
                    print(f"Warning: Field '{field_name}' should be int64 but got {type(value)}")
                    return False
                elif expected_dtype == "string" and not isinstance(value, str):
                    print(f"Warning: Field '{field_name}' should be string but got {type(value)}")
                    return False
        
    return True

def get_hf_api_client(token: Optional[str] = None):
    """
    Create and configure an HfApi client for repository operations.
    
    Args:
        token (str, optional): HuggingFace token. If None, uses environment variable.
        
    Returns:
        HfApi: Configured API client or None if not available
    """
    if not HF_HUB_AVAILABLE:
        return None
        
    try:
        # Get token from parameter or environment
        hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            print("Warning: No HuggingFace token found. API operations will fail.")
            return None
            
        # Create API client
        api = HfApi(token=hf_token)
        return api
    except Exception as e:
        print(f"Error creating HfApi client: {e}")
        return None



def upload_to_dataset(
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
    if not HF_HUB_AVAILABLE:
        print("Error: huggingface_hub not available for dataset operations")
        return False
        
    try:
        # Get API client
        api = get_hf_api_client(token)
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
            if not validate_data_structure(item, split):
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
    return upload_to_dataset(DATASET_ID, init_data, "init", token)

def upload_run_data(
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
    return upload_to_dataset(DATASET_ID, run_data, split, token)

def get_dataset_info() -> Optional[Dict]:
    """
    Get dataset information from the local config file.
    
    Returns:
        Dict: Dataset info including splits and features, or None if not found
    """
    schema = load_dataset_schema()
    if schema and "dataset_info" in schema:
        return schema["dataset_info"]
    return None

def print_dataset_schema():
    """
    Print the dataset schema for debugging purposes.
    """
    schema = load_dataset_schema()
    if schema:
        print("📊 Dataset Schema:")
        print(f"   Dataset: {schema.get('dataset_info', {}).get('dataset_name', 'Unknown')}")
        print(f"   Splits: {list(schema.get('features', {}).keys())}")
        for split_name, features in schema.get('features', {}).items():
            print(f"   {split_name} split fields: {list(features.keys())}")
    else:
        print("❌ No dataset schema found")

def ensure_valid_answer(answer: Any) -> str:
    """
    Ensure the answer is a valid string, never None or empty.
    
    Args:
        answer (Any): The answer to validate
        
    Returns:
        str: A valid string answer, defaulting to "No answer provided" if invalid
    """
    if answer is None:
        return "No answer provided"
    elif not isinstance(answer, str):
        return str(answer)
    elif answer.strip() == "":
        return "No answer provided"
    else:
        return answer

def get_nullable_field_value(value: Any, field_name: str, default: Any = None) -> Any:
    """
    Get a value for a nullable field, handling None values appropriately.
    
    Args:
        value (Any): The value to process
        field_name (str): Name of the field for logging
        default (Any): Default value if None
        
    Returns:
        Any: The processed value or default
    """
    if value is None:
        print(f"📝 Field '{field_name}' is None, using default: {default}")
        return default
    return value

def validate_nullable_field(value: Any, field_name: str, expected_type: str) -> bool:
    """
    Validate a nullable field against expected type.
    
    Args:
        value (Any): The value to validate
        field_name (str): Name of the field
        expected_type (str): Expected data type (string, float64, int64)
        
    Returns:
        bool: True if valid
    """
    if value is None:
        return True  # Null is always valid for nullable fields
    
    if expected_type == "float64" and not isinstance(value, (int, float)):
        print(f"❌ Field '{field_name}' should be float64 but got {type(value)}")
        return False
    elif expected_type == "int64" and not isinstance(value, int):
        print(f"❌ Field '{field_name}' should be int64 but got {type(value)}")
        return False
    elif expected_type == "string" and not isinstance(value, str):
        print(f"❌ Field '{field_name}' should be string but got {type(value)}")
        return False
    
    return True 