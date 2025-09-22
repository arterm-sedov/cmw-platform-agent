# Mistral Tool Call ID Fix

## Problem Description

The agent was encountering a 400 error when using the Mistral provider:

```
❌ Error: Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{"object":"error","message":"Tool call id was call_98660886 but must be a-z, A-Z, 0-9, with a length of 9.","type":"invalid_function_call","param":null,"code":"3280"}', 'provider_name': 'Mistral'}}
```

## Root Cause

Mistral AI has strict requirements for tool call IDs:
- Must be exactly 9 characters long
- Must contain only alphanumeric characters (a-z, A-Z, 0-9)
- No special characters, hyphens, underscores, etc.

LangChain was generating tool call IDs like `call_98660886` (13 characters) which violated these requirements.

## Solution

### 1. Custom Tool Call ID Generator

Created a custom tool call ID generator in `agent_ng/provider_adapters.py`:

```python
def generate_mistral_tool_call_id() -> str:
    """
    Generate a tool call ID that meets Mistral's requirements.
    
    Mistral requires tool call IDs to be:
    - Exactly 9 characters long
    - Only alphanumeric characters (a-z, A-Z, 0-9)
    
    Returns:
        A 9-character alphanumeric string suitable for Mistral tool calls
    """
    # Generate 9 random alphanumeric characters
    characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    return ''.join(random.choice(characters) for _ in range(9))
```

### 2. Tool Call ID Fixer

Created a function to fix existing tool call IDs:

```python
def fix_tool_call_ids_for_mistral(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fix tool call IDs in a list of tool calls to meet Mistral's requirements.
    
    Args:
        tool_calls: List of tool call dictionaries
        
    Returns:
        List of tool call dictionaries with fixed IDs
    """
    fixed_tool_calls = []
    for tool_call in tool_calls:
        fixed_tool_call = tool_call.copy()
        # Generate a new ID that meets Mistral's requirements
        fixed_tool_call['id'] = generate_mistral_tool_call_id()
        fixed_tool_calls.append(fixed_tool_call)
    return fixed_tool_calls
```

### 3. Updated MistralWrapper

Enhanced the `MistralWrapper` class to automatically fix tool call IDs in responses:

```python
def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
    """Invoke the LLM with message conversion for Mistral compatibility."""
    try:
        # Convert messages to Mistral format
        converted_messages = MistralMessageConverter.convert_messages(messages)
        
        # Call the original invoke method with converted messages
        response = self.llm.invoke(converted_messages, **kwargs)
        
        # Fix tool call IDs in the response if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            response.tool_calls = fix_tool_call_ids_for_mistral(response.tool_calls)
            print(f"[Mistral Wrapper] Fixed {len(response.tool_calls)} tool call IDs for Mistral compatibility")
        
        return response
    except Exception as e:
        # Fallback to original messages if conversion fails
        print(f"[Mistral Wrapper] Message conversion failed, using original format: {e}")
        return self.llm.invoke(messages, **kwargs)
```

## Testing

### Unit Tests

Created comprehensive unit tests in `misc_files/test_mistral_tool_call_fix.py`:

- ✅ Tool call ID generation (10 IDs)
- ✅ Tool call ID fixing (3 problematic IDs)
- ✅ Mistral requirements compliance (100 IDs)

### Integration Tests

Created integration tests in `misc_files/test_mistral_agent_integration.py`:

- ✅ MistralWrapper tool call ID fixing
- ✅ Async MistralWrapper functionality
- ✅ Direct tool call ID generator testing

### Test Results

```
📊 Test Results: 3/3 tests passed
🎉 All tests passed! Mistral tool call ID fix is working correctly.

📊 Integration Test Results: 3/3 tests passed
🎉 All integration tests passed! Mistral tool call ID fix is working correctly.
```

## Files Modified

1. **`agent_ng/provider_adapters.py`**
   - Added `generate_mistral_tool_call_id()` function
   - Added `fix_tool_call_ids_for_mistral()` function
   - Updated `MistralWrapper.invoke()` method
   - Updated `MistralWrapper.ainvoke()` method
   - Added `MistralWrapper.stream()` method
   - Added `MistralWrapper.astream()` method

2. **`misc_files/test_mistral_tool_call_fix.py`** (new)
   - Unit tests for tool call ID generation and fixing

3. **`misc_files/test_mistral_agent_integration.py`** (new)
   - Integration tests for MistralWrapper functionality

4. **`docs/MISTRAL_TOOL_CALL_ID_FIX.md`** (new)
   - This documentation file

## Impact

- ✅ Resolves the 400 error with Mistral provider
- ✅ Maintains compatibility with all other providers
- ✅ No breaking changes to existing functionality
- ✅ Automatic tool call ID fixing for Mistral
- ✅ Comprehensive test coverage

## Usage

The fix is automatically applied when using Mistral models. No changes are required in the application code. The `MistralWrapper` will automatically:

1. Convert messages to Mistral format
2. Fix any tool call IDs in the response to meet Mistral's requirements
3. Return the corrected response

## Example

**Before (causing error):**
```python
tool_call_id = "call_98660886"  # 13 characters, contains underscore
```

**After (working):**
```python
tool_call_id = "dmhgaD9Ja"  # 9 characters, alphanumeric only
```

The fix ensures that all tool call IDs generated for Mistral meet the strict requirements, preventing the 400 error and allowing successful tool calls.
