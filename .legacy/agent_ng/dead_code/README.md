# Dead Code Archive

This directory contains all the dead code that was identified and extracted from the `agent_ng` module. Dead code refers to functions, classes, and files that are defined but never called or used anywhere in the codebase.

## 📁 File Structure

```
.legacy/agent_ng/dead_code/
├── README.md                           # This file
├── reset_functions.py                  # Reset/cleanup functions
├── debug_utility_functions.py         # Debug and utility functions
├── config_setup_functions.py          # Configuration and setup functions
├── app_functions.py                   # App-related functions
├── stats_export_functions.py          # Statistics and export functions
├── test_vector_store.py               # Test file for non-existent vector store
├── debug_tools.py                     # Debug script for tool detection
└── error_coverage_analyzer.py         # Analysis script for error coverage
```

## 🔍 Dead Code Categories

### 1. Reset/Cleanup Functions (`reset_functions.py`)
These functions were designed to reset global instances but were never called:
- `reset_agent_ng()` - Reset global agent instance
- `reset_llm_manager()` - Reset LLM manager instance
- `reset_stats_manager()` - Reset stats manager instance
- `reset_response_processor()` - Reset response processor instance
- `reset_token_tracker()` - Reset token tracker instances
- `reset_trace_manager()` - Reset trace manager instances
- `cleanup_debug_system()` - Cleanup debug system

### 2. Debug/Utility Functions (`debug_utility_functions.py`)
Debug and utility functions that were never used:
- `print_message_components()` - Print detailed message components
- `_print_attribute()` - Print single attribute with formatting
- `get_recent_logs()` - Get recent log entries (returns empty list)
- `trace_prints_with_context()` - Decorator for tracing prints with context
- `trace_prints()` - Decorator for tracing prints
- `Tee` class - Duplicate writes to multiple streams
- `_SinkWriter` class - Writer that sends data to sink function
- `get_trace_history()` - Get trace history

### 3. Configuration/Setup Functions (`config_setup_functions.py`)
Configuration functions that were rarely used:
- `print_config()` - Print current configuration (only used with `--config` CLI flag)
- `get_openai_wrapper()` - Get OpenAI wrapper for LangSmith (only used in tests)
- `get_langfuse_callback_handler()` - Get Langfuse callback handler (only used in streaming)

### 4. App Functions (`app_functions.py`)
App-related functions that were never called:
- `detect_language_from_url()` - Detect language from URL parameters
- `create_safe_demo()` - Create safe demo instance
- `reload_demo()` - Reload demo for Gradio hot reloading

### 5. Stats/Export Functions (`stats_export_functions.py`)
Statistics and export functions that were never used:
- `export_stats()` - Export statistics to JSON file
- `get_stats_summary()` - Get human-readable stats summary
- `get_performance_metrics()` - Get recent performance metrics
- `get_error_summary()` - Get error summary statistics

### 6. Standalone Test/Debug Files
Complete files that were standalone and unused:
- `test_vector_store.py` - Test file for non-existent vector store functionality
- `debug_tools.py` - Debug script for tool detection and binding
- `error_coverage_analyzer.py` - Analysis script for error code coverage

## 🗑️ Unused Imports

The following imports were also identified as unused:
- `convert_messages_for_mistral` in `agent_ng/utils.py` (imported but never used)
- `asdict` from dataclasses in `agent_ng/app_ng_modular.py` (imported but never used)

## 📊 Summary

**Total Dead Code Identified:**
- **Reset/Cleanup Functions:** 7 functions
- **Debug/Utility Functions:** 8 functions  
- **Configuration Functions:** 3 functions
- **App Functions:** 3 functions
- **Stats/Export Functions:** 4 functions
- **Test Files:** 3 complete files
- **Unused Imports:** 2 imports

## ⚠️ Important Notes

1. **Safe to Remove:** All code in this directory can be safely removed from the main codebase without affecting functionality.

2. **Future Use:** Some functions might be useful for future debugging or maintenance, so they're preserved here rather than deleted.

3. **Dependencies:** When removing these functions from the main codebase, also remove any unused imports and references.

4. **Testing:** The test files can be removed entirely as they test non-existent functionality.

## 🔄 Restoration

If any of this code is needed in the future, it can be restored by:
1. Copying the relevant functions back to their original files
2. Adding the necessary imports
3. Updating any function calls as needed

## 📅 Extraction Date

Extracted on: 2025-01-26
Extracted by: AI Assistant
Reason: Code cleanup and maintenance optimization
