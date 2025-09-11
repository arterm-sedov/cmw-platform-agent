---
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# Comindware Analyst Copilot

---

**Authors:** Arte(r)m Sedov & Marat Mutalimov

**Github:** <https://github.com/arterm-sedov/>

**This repo:** <https://github.com/arterm-sedov/cmw-platform-agent>

## 🚀 The Comindware Analyst Copilot

Behold the Comindware Analyst Copilot — a robust and extensible system designed for real-world reliability and performance in creating entities within the Comindware Platform.

## 🕵🏻‍♂️ What is this project?

This is an **experimental multi-LLM agent** that demonstrates AI agent and CMW Platform integration:

- **Input**: The user asks the Comindware Analyst Copilot to create entities in the CMW Platform instance.
- **Task**: The agent has a set of tools to translate natural language user requests into CMW Platform API calls for entity creation.
- **Output**: Entities (templates, attributes, workflows, etc.) are created in the CMW Platform based on user specifications.

## 🎯 Project Goals

To create an agent that will allow batch entity creation within the CMW Platform, enabling users to:

- Create templates with custom attributes
- Define workflows and business processes
- Set up data models and relationships
- Automate platform configuration through natural language

## ❓ Why This Project?

This experimental system is based on current AI agent technology and demonstrates:

- **Advanced Tool Usage**: Seamless integration of 20+ specialized tools including AI-powered tools and third-party AI engines
- **Multi-Provider Resilience**: Automatic testing and switching between different LLM providers
- **Comprehensive Tracing**: Complete visibility into the agent's decision-making process
- **Structured Initialization Summary:** After startup, a clear table shows which models/providers are available, with/without tools, and any errors—so you always know your agent's capabilities.

## 📊 What You'll Find Here

- **Documentation**: Detailed technical specifications and usage guides
- **CMW Platform Integration**: Tools and utilities for working with Comindware Platform APIs
- **Entity Creation Capabilities**: Specialized tools for creating templates, attributes, and workflows

## 🏗️ Technical Architecture

### LLM Configuration

The agent uses a sophisticated multi-LLM approach with the following providers in sequence:

1. **OpenRouter** (Primary)
   - Models: `deepseek/deepseek-chat-v3-0324:free`, `mistralai/mistral-small-3.2-24b-instruct:free`, `openrouter/cypher-alpha:free`
   - Token Limits: 100K-1M tokens
   - Tool Support: ✅ Full tool-calling capabilities

2. **Mistral AI** (Secondary)
   - Models: `mistral-small-latest`, `mistral-medium-latest`, `mistral-large-latest`
   - Token Limits: 32K tokens
   - Rate Limit: 500,000 tokens per minute (free tier)
   - Tool Support: ✅ Full tool-calling capabilities

3. **Google Gemini** (Fallback)
   - Model: `gemini-2.5-pro`
   - Token Limit: 2M tokens (virtually unlimited)
   - Tool Support: ✅ Full tool-calling capabilities

4. **Groq** (Second Fallback)
   - Models: `qwen-qwq-32b`, `llama-3.1-8b-instant`, `llama-3.3-70b-8192`
   - Token Limits: 16K tokens
   - Rate Limits: Generous free tier limits (see [Groq docs](https://console.groq.com/docs/rate-limits))
   - Tool Support: ✅ Full tool-calling capabilities

5. **HuggingFace** (Final Fallback)
   - Models: `Qwen/Qwen2.5-Coder-32B-Instruct`, `microsoft/DialoGPT-medium`, `gpt2`
   - Token Limits: 1K tokens
   - Tool Support: ❌ No tool-calling (text-only responses)

### Tool Suite

The agent includes 20+ specialized tools:

- **CMW Platform Tools**: Create templates, attributes, workflows, and other platform entities
- **Web Search**: Deep research capabilities for gathering information
- **Code Execution**: Python code execution for data processing
- **File Analysis**: Document processing and analysis
- **Mathematical Operations**: Complex calculations and data analysis
- **Image Processing**: OCR and image analysis capabilities

### Performance Expectations

- **Success Rate**: 50-65% entities created successfully
- **Response Time**: 30-100 seconds per entity creation request (depending on complexity and LLM)
- **Tool Usage**: 2-8 tool calls per request on average
- **Fallback Rate**: 20-40% of requests require human clarification

## 🏗️ Modular Architecture

The codebase follows a clean modular design with clear separation of concerns:

### Core Modules
- **`similarity_manager.py`**: Handles vector similarity operations and text matching with optional local embeddings
- **`tool_call_manager.py`**: Manages tool call deduplication and history tracking
- **`vector_store.py`**: Optional Supabase vector storage operations
- **`agent.py`**: Main orchestration logic using the above modules

### Key Benefits
- **No External Dependencies**: Core features work without Supabase or sentence-transformers
- **Graceful Degradation**: Falls back to basic text similarity when embeddings unavailable
- **Clean Separation**: Each module has a single responsibility with no circular dependencies
- **Backward Compatibility**: Existing functionality remains unchanged

## Dataset Structure

The output trace facilitates:

- **Debugging**: Complete visibility into execution flow
- **Performance Analysis**: Detailed timing and token usage metrics
- **Error Analysis**: Comprehensive error information with context
- **Tool Usage Analysis**: Complete tool execution history
- **LLM Comparison**: Detailed comparison of different LLM behaviors
- **Cost Optimization**: Token usage analysis for cost management

Each request trace is uploaded to a HuggingFace dataset.

The dataset contains comprehensive execution traces with the following structure:

### Root Level Fields

```python
{
    "question": str,                    # Original question text
    "file_name": str,                   # Name of attached file (if any)
    "file_size": int,                   # Length of base64 file data (if any)
    "start_time": str,                  # ISO format timestamp when processing started
    "end_time": str,                    # ISO format timestamp when processing ended
    "total_execution_time": float,      # Total execution time in seconds
    "tokens_total": int,                # Total tokens used across all LLM calls
    "debug_output": str,                # Comprehensive debug output as text
}
```

### LLM Traces

```python
"llm_traces": {
    "llm_type": [                      # e.g., "openrouter", "gemini", "groq", "huggingface"
        {
            "call_id": str,             # e.g., "openrouter_call_1"
            "llm_name": str,            # e.g., "deepseek-chat-v3-0324" or "Google Gemini"
            "timestamp": str,           # ISO format timestamp
            
            # === LLM CALL INPUT ===
            "input": {
                "messages": List,       # Input messages (trimmed for base64)
                "use_tools": bool,      # Whether tools were used
                "llm_type": str         # LLM type
            },
            
            # === LLM CALL OUTPUT ===
            "output": {
                "content": str,         # Response content
                "tool_calls": List,     # Tool calls from response
                "response_metadata": dict,  # Response metadata
                "raw_response": dict    # Full response object (trimmed for base64)
            },
            
            # === TOOL EXECUTIONS ===
            "tool_executions": [
                {
                    "tool_name": str,      # Name of the tool
                    "args": dict,          # Tool arguments (trimmed for base64)
                    "result": str,         # Tool result (trimmed for base64)
                    "execution_time": float, # Time taken for tool execution
                    "timestamp": str,      # ISO format timestamp
                    "logs": List           # Optional: logs during tool execution
                }
            ],
            
            # === TOOL LOOP DATA ===
            "tool_loop_data": [
                {
                    "step": int,           # Current step number
                    "tool_calls_detected": int,  # Number of tool calls detected
                    "consecutive_no_progress": int,  # Steps without progress
                    "timestamp": str,      # ISO format timestamp
                    "logs": List           # Optional: logs during this step
                }
            ],
            
            # === EXECUTION METRICS ===
            "execution_time": float,       # Time taken for this LLM call
            "total_tokens": int,           # Estimated token count (fallback)
            
            # === TOKEN USAGE TRACKING ===
            "token_usage": {               # Detailed token usage data
                "prompt_tokens": int,      # Total prompt tokens across all calls
                "completion_tokens": int,  # Total completion tokens across all calls
                "total_tokens": int,       # Total tokens across all calls
                "call_count": int,         # Number of calls made
                "calls": [                 # Individual call details
                    {
                        "call_id": str,   # Unique call identifier
                        "timestamp": str,  # ISO format timestamp
                        "prompt_tokens": int,     # This call's prompt tokens
                        "completion_tokens": int, # This call's completion tokens
                        "total_tokens": int,      # This call's total tokens
                        "finish_reason": str,     # How the call finished (optional)
                        "system_fingerprint": str, # System fingerprint (optional)
                        "input_token_details": dict,  # Detailed input breakdown (optional)
                        "output_token_details": dict  # Detailed output breakdown (optional)
                    }
                ]
            },
            
            # === ERROR INFORMATION ===
            "error": {                     # Only present if error occurred
                "type": str,              # Exception type name
                "message": str,           # Error message
                "timestamp": str          # ISO format timestamp
            },
            
            # === LLM-SPECIFIC LOGS ===
            "logs": List,                 # Logs specific to this LLM call
            
            # === FINAL ANSWER ENFORCEMENT ===
            "final_answer_enforcement": [  # Optional: logs from _force_final_answer for this LLM call
                {
                    "timestamp": str,     # ISO format timestamp
                    "message": str,       # Log message
                    "function": str       # Function that generated the log (always "_force_final_answer")
                }
            ]
        }
    ]
}
```

### Per-LLM Stdout Capture

```python
"per_llm_stdout": [
    {
        "llm_type": str,            # LLM type
        "llm_name": str,            # LLM name (model ID or provider name)
        "call_id": str,             # Call ID
        "timestamp": str,           # ISO format timestamp
        "stdout": str               # Captured stdout content
    }
]
```

### Question-Level Logs

```python
"logs": [
    {
        "timestamp": str,           # ISO format timestamp
        "message": str,             # Log message
        "function": str             # Function that generated the log
    }
]
```

### Final Results

```python
"final_result": {
    "submitted_answer": str,        # Final answer (consistent with code)
    "similarity_score": float,      # Similarity score (0.0-1.0)
    "llm_used": str,               # LLM that provided the answer
    "reference": str,               # Reference answer used
    "question": str,                # Original question
    "file_name": str,               # File name (if any)
    "error": str                    # Error message (if any)
}
```

## Key Features

### Intelligent Fallback System

The agent automatically tries multiple LLM providers in sequence:

- **OpenRouter** (Primary): Fast, reliable, good tool support, has tight daily limits on free tiers
- **Google Gemini** (Fallback): High token limits, excellent reasoning
- **Groq** (Second Fallback): Fast inference, good for simple tasks, has tight token limits per request
- **HuggingFace** (Final Fallback): Local models, no API costs, does not support tools typically

### Advanced Tool Management

- **Automatic Tool Selection**: LLM chooses appropriate tools based on question
- **Tool Deduplication**: Prevents duplicate tool calls using vector similarity
- **Usage Limits**: Prevents excessive tool usage (e.g., max 3 web searches per question)
- **Error Handling**: Graceful degradation when tools fail

### Sophisticated implementations

- **Recursive Truncation**: Separate methods for base64 and max-length truncation
- **Recursive JSON Serialization**: Ensures the complex objects ar passable as HuggingFace JSON dataset
- **Decorator-Based Print Capture**: Captures all print statements into trace data
- **Multilevel Contextual Logging**: Logs tied to specific execution contexts
- **Per-LLM Stdout Traces**: Stdout captured separately for each LLM attempt in a human-readable form
- **Consistent LLM Schema**: Data structures for consistent model identification, configuring and calling
- **Complete Trace Model**: Hierarchical structure with comprehensive coverage
- **Structured dataset uploads** to HuggingFace datasets
- **Schema validation** against `dataset_config.json`
- **Three data splits**: `init` (initialization), `runs` (legacy aggregated results), and `runs_new` (granular per-question results)
- **Robust error handling** with fallback mechanisms

### Comprehensive Tracing

Every question generates a complete execution trace including:

- **LLM Interactions**: All input/output for each LLM attempt
- **Tool Executions**: Detailed logs of every tool call
- **Performance Metrics**: Token usage, execution times, success rates
- **Error Information**: Complete error context and fallback decisions
- **Stdout Capture**: All debug output from each LLM attempt

### Rate Limiting & Reliability

- **Smart Rate Limiting**: Model-specific and provider-specific rate limits
- **Token Management**: Automatic truncation and summarization
- **Error Recovery**: Automatic retry with different LLMs
- **Graceful Degradation**: Continues processing even if some components fail
- **Smart Rate Limit Handling**: Throttles and retries on 429 errors before falling back to other LLMs

## Usage

### Live Demo

Visit the Gradio interface to test the agent interactively:

<https://localhost/cmw-platform-agent>

### Programmatic Usage

```python
from agent import CmwAgent

# Initialize the agent
agent = CmwAgent()

# Create an entity in CMW Platform
result = agent("Create a template called 'Customer' with attributes: Name (Text), Email (Text), Phone (Text)")

# Access the results
print(f"Answer: {result['submitted_answer']}")
print(f"Similarity: {result['similarity_score']}")
print(f"LLM Used: {result['llm_used']}")
```

### Dataset Access

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("arterm-sedov/agent-course-final-assignment")

# Access initialization data
init_data = dataset["init"]["train"]

# Access evaluation results
runs_data = dataset["runs_new"]["train"]
```

## File Structure

The main agent runtime files are:

```
cmw-platform-agent/
├── agent.py              # Main agent implementation
├── app.py                # Gradio web interface
├── tools.py              # Tool definitions and implementations
├── utils.py              # Core upload functions with validation
├── system_prompt.json    # System prompt configuration
├── login_manager.py      # CMW Platform authentication
├── file_manager.py       # File handling utilities
├── dataset_manager.py    # Dataset management
├── vector_store.py       # Vector storage for embeddings
└── logs/                 # Execution logs and results
```

## CMW Platform Integration

This agent is designed to work with the Comindware Platform, a business process management and workflow automation platform. The agent can:

- **Create Templates**: Define data structures with custom attributes
- **Configure Workflows**: Set up business processes and automation rules
- **Manage Entities**: Create, update, and configure platform objects
- **API Integration**: Interact with CMW Platform APIs for entity management

For more information about the Comindware Platform, see the [CMW Platform Documentation](https://github.com/arterm-sedov/cbap-mkdocs-ru).

## Performance Statistics

The agent has been evaluated on complex entity creation tasks with the following results:

- **Overall Success Rate**: 50-65%, up to 80% with all four LLMs available
- **Tool Usage**: Average 2-8 tools per entity creation request
- **LLM Fallback Rate**: 20-40% of requests require multiple LLMs
- **Response Time**: 30-120 seconds per entity creation request
- **Token Usage**: 1K-100K tokens per request (depending on complexity)

## Contributing

This is an experimental research project. Contributions are welcome in the form of:

- **Bug Reports**: Issues with the agent's reasoning or tool usage
- **Feature Requests**: New tools or capabilities for CMW Platform integration
- **Performance Improvements**: Optimizations for speed or accuracy
- **Documentation**: Improvements to this README or code comments
