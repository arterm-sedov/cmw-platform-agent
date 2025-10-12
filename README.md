---
emoji: 🕵🏻‍♂️
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: agent_ng/app_ng_modular.py
pinned: true
hf_oauth: true
hf_oauth_expiration_minutes: 480
license: mit
title: CMW Copilot
---

# Comindware Analyst Copilot

---

**Authors:** Arte(r)m Sedov & Marat Mutalimov

**Github:** <https://github.com/arterm-sedov/>

**This repo:** <https://github.com/arterm-sedov/cmw-platform-agent>

## 🚀 The Comindware Analyst Copilot

Behold the Comindware Analyst Copilot — a robust and extensible system designed for real-world reliability and performance in creating entities within the Comindware Platform.

### 🆕 **LangChain-Native Modular Architecture**

The system features a **LangChain-native modular Gradio app** (`app_ng_modular.py`) that provides:

- **Modular Tab Architecture**: Separate modules for Chat, Logs, and Stats tabs
- **Multi-turn Conversations**: Reliable conversation memory with tool calls using LangChain's native memory management
- **Pure LangChain Patterns**: Native LangChain conversation chains, memory, and streaming
- **Real-time Streaming**: Live response streaming with tool visualization using `astream()` and `astream_events()`
- **Modern UI**: Comprehensive monitoring, debugging, and statistics with internationalization
- **Multi-LLM Support**: OpenRouter, Gemini, Groq, Mistral, and HuggingFace integration with automatic fallback
- **Session Isolation**: Each user gets isolated agent instances with proper session management
- **Internationalization**: Full i18n support (English/Russian) using Gradio's built-in I18n system
- **Comprehensive Error Handling**: Advanced error classification and recovery with vector similarity matching
- **Native LangChain Streaming**: Token-by-token streaming using LangChain's native streaming capabilities

**Quick Start:**

```bash
python agent_ng/app_ng_modular.py
```

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

## 🏗️ Technical Architecture

### Core Architecture

The Agent NG is a modern, LangChain-native conversational AI agent built with a clean modular architecture. It features multi-turn conversations with tool calls, session isolation, real-time streaming, and comprehensive error handling.

#### Main Components

1. **CmwAgent** (`langchain_agent.py`) - Main agent orchestrator using pure LangChain patterns
2. **NextGenApp** (`app_ng_modular.py`) - Gradio web application with modular tab architecture
3. **LLMManager** (`llm_manager.py`) - Multi-provider LLM management with persistent instances
4. **SessionManager** (`session_manager.py`) - User session isolation and state management
5. **ErrorHandler** (`error_handler.py`) - Comprehensive error handling with vector similarity matching
6. **UI Components** (`tabs/`, `ui_manager.py`) - Modular UI system with internationalization
7. **Memory Management** (`langchain_memory.py`) - LangChain-native memory management
8. **Streaming** (`native_langchain_streaming.py`) - Native LangChain streaming implementation
9. **Statistics** (`stats_manager.py`) - Performance metrics and usage tracking
10. **Tracing** (`trace_manager.py`) - Comprehensive execution tracing and debugging

#### Key Features

- ✅ **LangChain-Native**: Uses pure LangChain patterns for memory, chains, and streaming
- ✅ **Multi-Turn Conversations**: Proper tool call context preservation with LangChain memory
- ✅ **Session Isolation**: Each user gets isolated agent instances with proper cleanup
- ✅ **Real-Time Streaming**: Token-by-token response streaming using `astream()` and `astream_events()`
- ✅ **Modular Architecture**: Clean separation of concerns with dedicated modules
- ✅ **Internationalization**: Full i18n support (English/Russian) using Gradio's I18n system
- ✅ **Error Recovery**: Robust error handling with vector similarity and provider fallback
- ✅ **Tool Integration**: 20+ CMW platform tools + utility tools with proper organization
- ✅ **Comprehensive Tracing**: Complete execution traces with debug output capture
- ✅ **Statistics Tracking**: Real-time performance metrics and usage analytics

### LLM Configuration

The agent uses a sophisticated multi-LLM approach with the following providers in sequence:

1. **OpenRouter** (Primary)
   - Models: `x-ai/grok-4-fast:free`, `qwen/qwen3-coder:free`, `openai/gpt-5-mini`, `x-ai/grok-code-fast-1`, `mistralai/codestral-2508`, `deepseek/deepseek-chat-v3.1:free`, `deepseek/deepseek-r1-0528`, `openai/gpt-oss-120b`, `nvidia/nemotron-nano-9b-v2:free`, `qwen/qwen3-coder-flash`
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

The agent includes 20+ specialized tools organized into categories:

#### CMW Platform Tools

- **Application Tools** (`applications_tools/`): List applications, templates, and platform entities
- **Attribute Tools** (`attributes_tools/`): Create and manage all attribute types:
  - Text attributes (`tools_text_attribute.py`)
  - Boolean attributes (`tools_boolean_attribute.py`)
  - DateTime attributes (`tools_datetime_attribute.py`)
  - Decimal/Numeric attributes (`tools_decimal_attribute.py`)
  - Document attributes (`tools_document_attribute.py`)
  - Drawing attributes (`tools_drawing_attribute.py`)
  - Duration attributes (`tools_duration_attribute.py`)
  - Image attributes (`tools_image_attribute.py`)
  - Record attributes (`tools_record_attribute.py`)
  - Role attributes (`tools_role_attribute.py`)
  - Account attributes (`tools_account_attribute.py`)
  - Enum attributes (`tools_enum_attribute.py`)
- **Template Tools** (`templates_tools/`): List and manage template attributes
- **General Operations**: Delete, archive/unarchive, and retrieve attributes

#### Utility Tools

- **Web Search**: Deep research capabilities using Tavily, Wikipedia, and Arxiv
- **Code Execution**: Python code execution for data processing and analysis
- **File Analysis**: Document processing and analysis (PDF, images, text)
- **Mathematical Operations**: Complex calculations and data analysis
- **Image Processing**: OCR and image analysis capabilities using pytesseract
- **Data Processing**: CSV, JSON, and other data format handling
- **Platform Entity URL**: Generate URLs for Comindware Platform entities

## 🔧 Core Modules

### 1. CmwAgent (langchain_agent.py)

**Purpose**: Main agent orchestrator using pure LangChain patterns

**Key Features**:
- LangChain-native memory management with `ConversationBufferMemory`
- Multi-turn conversation support with tool calls using LangChain chains
- Session-specific agent instances with proper isolation
- File handling with security and session-based storage
- Comprehensive statistics tracking and performance monitoring
- Native LangChain streaming using `astream()` and `astream_events()`
- Tool call context preservation across conversation turns
- LangSmith tracing integration for observability

**Usage**:
```python
agent = CmwAgent(session_id="user_123")
response = agent.process_message("Calculate 5 + 3", "conversation_1")
```

### 2. LLMManager (llm_manager.py)

**Purpose**: Centralized LLM provider management with persistent instances

**Supported Providers**:
- **Gemini** (Google): `gemini-2.5-pro` with 2M token limit
- **OpenRouter**: `deepseek/deepseek-chat-v3.1:free`, `mistralai/mistral-small-3.2-24b-instruct:free`
- **Mistral**: `mistral-small-latest`, `mistral-medium-latest`, `mistral-large-latest`
- **Groq**: `qwen-qwq-32b`, `llama-3.1-8b-instant`, `llama-3.3-70b-8192`
- **HuggingFace**: Various models including `Qwen/Qwen2.5-Coder-32B-Instruct`
- **GigaChat**: Sber's Russian LLM (when available)

**Features**:
- Persistent LLM instances across requests
- Tool binding and management with proper provider adapters
- Provider-specific optimizations and error handling
- Health monitoring and automatic fallback
- Thread-safe operations with proper locking
- Configuration-driven initialization
- Rate limiting and token management

### 3. SessionManager (session_manager.py)

**Purpose**: User session isolation and management

**Features**:
- Session-specific agent instances with proper isolation
- Automatic cleanup and resource management
- Session data isolation with Gradio request handling
- Multi-language support with i18n integration
- Clean session lifecycle management
- Proper Gradio request handling for session identification

### 4. ErrorHandler (error_handler.py)

**Purpose**: Comprehensive error classification and recovery with advanced pattern matching

**Error Types Handled**:
- Rate limiting (429 errors) with retry timing extraction
- Authentication errors (401, 403)
- Token limit exceeded and context too long errors
- Network connectivity issues and timeouts
- Provider-specific errors (Mistral tool call IDs, OpenRouter limits, etc.)
- Service unavailable and internal errors
- Resource exhausted and capacity exceeded errors

**Features**:
- Vector similarity for error pattern matching using TF-IDF and cosine similarity
- Provider failure tracking with session-specific counters
- Automatic retry with exponential backoff and smart timing
- Structured error information with recovery suggestions
- HTTP status code extraction from various error formats
- Provider-specific error classification and handling

### 5. UI System

#### Modular Tab Architecture (tabs/)
- **ChatTab** (`chat_tab.py`): Main conversation interface with quick actions and i18n support
- **LogsTab** (`logs_tab.py`): Debug and initialization logs with real-time updates
- **StatsTab** (`stats_tab.py`): Performance metrics and statistics with live monitoring

#### UI Manager (`ui_manager.py`)
- Centralized UI component management
- Theme and styling with custom CSS
- Component state management and event handling
- Internationalization integration with Gradio's I18n system
- Responsive design and user experience optimization

## 🔄 Memory Management

### LangChain Memory (langchain_memory.py)

**Features**:
- Uses LangChain's native memory management with `ConversationBufferMemory`
- Tool call context preservation across conversation turns
- Session-specific memory instances with proper isolation
- Automatic conversation summarization and context management
- Integration with LangChain conversation chains
- Proper message formatting for different LLM providers

**Memory Types**:
- **Native LangChain Memory**: Stores full conversation history using LangChain patterns
- **Tool-aware memory**: Preserves tool call results and context
- **Session isolation**: Memory per user session with automatic cleanup
- **Conversation chains**: Proper integration with LangChain's chain architecture

## 🌐 Internationalization

### Language Support (i18n_translations.py)

**Supported Languages**:
- **English (en)**: Default language
- **Russian (ru)**: Full translation with comprehensive UI coverage

**Features**:
- Dynamic language switching using Gradio's built-in I18n system
- Complete UI component translations including tabs, buttons, and messages
- Error message localization with context-aware translations
- Quick action button translations
- Welcome messages and help text in both languages
- Seamless integration with Gradio's internationalization framework

**Configuration**:
```bash
# Environment variable
export CMW_DEFAULT_LANGUAGE="ru"

# Command line
python app_ng_modular.py --ru
```

## ⚙️ Configuration

### Agent Configuration (agent_config.py)

**Core Settings**:
```python
@dataclass
class RefreshIntervals:
    status: float = 2.0      # Status updates
    logs: float = 3.0        # Log refresh
    stats: float = 4.0       # Statistics refresh
    progress: float = 2.0    # Progress updates
```

**Environment Variables**:
- `CMW_DEFAULT_LANGUAGE`: Default language (ru/en)
- `CMW_DEFAULT_PORT`: Default port (7860)
- `CMW_DEBUG_MODE`: Enable debug mode
- `AGENT_PROVIDER`: LLM provider selection

### Provider Configuration

**Example Environment Setup**:
```bash
# LLM Provider APIs
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
MISTRAL_API_KEY=your_mistral_key
GROQ_API_KEY=your_groq_key

# Agent Configuration
AGENT_PROVIDER=mistral
CMW_DEFAULT_LANGUAGE=ru
CMW_DEBUG_MODE=true
```

## 🔀 Streaming & Real-Time Features

### Native Streaming (native_langchain_streaming.py)

**Features**:
- Token-by-token streaming using LangChain's native `astream()` and `astream_events()`
- Tool usage visualization with real-time updates
- Real-time progress updates and status monitoring
- Event-based architecture with proper event handling
- No artificial delays - uses LangChain's built-in streaming capabilities
- LangSmith tracing integration at the LLM call level

**Event Types**:
- `content`: Main response content with token-by-token delivery
- `thinking`: Agent reasoning process and decision making
- `tool_use`: Tool execution steps with real-time feedback
- `error`: Error messages and recovery information
- `metadata`: Additional information and context
- `streaming`: Real-time streaming events from LangChain

## 📊 Statistics & Monitoring

### Stats Manager (stats_manager.py)

**Metrics Tracked**:
- LLM usage statistics with success/failure rates
- Response times and performance metrics
- Tool call frequency and usage patterns
- Error rates and failure analysis
- Session statistics and user activity
- Conversation history and question tracking
- Token usage and cost analysis

**Features**:
- Real-time metrics with live updates
- Export capabilities for data analysis
- Performance monitoring and optimization
- Usage analytics and insights
- Comprehensive statistics dashboard
- Historical data tracking and trends

### Debug System (debug_streamer.py)

**Features**:
- Real-time log streaming with live updates
- Categorized logging with proper log levels
- Session-specific debug contexts and isolation
- Performance tracing and execution monitoring
- Comprehensive debug output capture
- Integration with trace management system

**Log Categories**:
- INIT: Initialization events and startup processes
- LLM: LLM operations and API calls
- TOOL: Tool executions and results
- ERROR: Error handling and recovery
- THINKING: Agent reasoning and decision making
- STREAMING: Real-time streaming events
- SESSION: Session management and user activity

## 🚀 Concurrency & Performance

### Queue Management (queue_manager.py)

**Features**:
- Request queuing
- Concurrency control
- Resource management
- Performance optimization

### Concurrency Configuration (concurrency_config.py)

**Settings**:
- Maximum concurrent requests
- Queue limits
- Timeout configurations
- Resource allocation

## 🔒 Security Features

### Session Isolation
- User-specific agent instances
- Session-based file handling
- Secure resource cleanup
- Data privacy protection

### File Security
- Secure file upload handling
- Session-specific file storage
- Automatic cleanup
- Path sanitization

## 🧪 Testing

### Test Coverage (agent_ng/_tests/)

**Test Categories**:
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Full workflow testing
- **Performance Tests**: Load and stress testing

**Key Test Files**:
- `test_agent_functionality.py`: Core agent features
- `test_multi_turn_conversations.py`: Conversation flows
- `test_platform_tools.py`: Tool integration
- `test_error_handler_comprehensive.py`: Error handling

## 📱 Web Application

### NextGenApp (app_ng_modular.py)

**Features**:
- Modular tab architecture
- Real-time UI updates
- Session management
- Internationalization
- Responsive design

**UI Components**:
- Chat interface with streaming
- Debug logs with real-time updates
- Statistics dashboard
- Configuration controls

## 🏗️ Modular Architecture

The codebase follows a clean modular design with clear separation of concerns:

### Core Agent Modules (`agent_ng/`)

- **`langchain_agent.py`**: LangChain-native agent implementation with conversation chains and memory
- **`app_ng_modular.py`**: Main Gradio application with modular tab architecture and i18n support
- **`llm_manager.py`**: Multi-provider LLM management with persistent instances and health monitoring
- **`error_handler.py`**: Comprehensive error handling with vector similarity matching and recovery
- **`message_processor.py`**: Message processing and formatting with proper validation
- **`response_processor.py`**: Response processing and validation with error handling
- **`stats_manager.py`**: Statistics tracking and monitoring with real-time updates
- **`trace_manager.py`**: Trace logging and debugging with comprehensive execution traces
- **`debug_streamer.py`**: Debug system and logging with categorized output
- **`token_counter.py`**: Token usage tracking and optimization across providers
- **`session_manager.py`**: Session management and state handling with proper isolation
- **`queue_manager.py`**: Request queue management and concurrency control
- **`concurrency_config.py`**: Concurrency and threading configuration
- **`ui_manager.py`**: UI state management and updates with internationalization
- **`tool_deduplicator.py`**: Tool call deduplication and optimization
- **`streaming_config.py`**: Streaming configuration and settings
- **`provider_adapters.py`**: LLM provider-specific adapters and optimizations
- **`langchain_memory.py`**: LangChain memory management with conversation chains
- **`native_langchain_streaming.py`**: Native LangChain streaming using astream() and astream_events()
- **`i18n_translations.py`**: Internationalization support with English/Russian translations
- **`agent_config.py`**: Centralized configuration management

### Tab Modules (`agent_ng/tabs/`)

- **`chat_tab.py`**: Main chat interface tab with quick actions and i18n support
- **`logs_tab.py`**: Logs and debugging tab with real-time updates
- **`stats_tab.py`**: Statistics and monitoring tab with live metrics

### Tool Modules (`tools/`)

- **`tools.py`**: Core tool functions and consolidated tool definitions with 20+ tools
- **`applications_tools/`**: Application and template management tools
  - `tool_list_applications.py`: List platform applications
  - `tool_list_templates.py`: List application templates
  - `tool_platform_entity_url.py`: Generate platform entity URLs
- **`attributes_tools/`**: Attribute management tools for all attribute types
  - Text, Boolean, DateTime, Decimal, Document, Drawing, Duration, Image, Record, Role, Account, Enum attributes
  - Delete, archive/unarchive, and retrieve attribute operations
- **`templates_tools/`**: Template-related tools and operations
  - `tool_list_attributes.py`: List template attributes
- **`tool_utils.py`**: Common tool utilities and helpers
- **`models.py`**: Data models and schemas for tools
- **`requests_.py`**: HTTP request utilities and helpers
- **`file_utils.py`**: File handling utilities with security
- **`pdf_utils.py`**: PDF processing utilities with OCR support

### Key Benefits

- **Modular Design**: Clean separation of concerns with dedicated modules
- **LangChain Native**: Pure LangChain patterns and best practices
- **Extensible**: Easy to add new tools and capabilities
- **Maintainable**: Clear module boundaries and responsibilities
- **Testable**: Isolated components for comprehensive testing

## Performance Statistics

The agent has been evaluated on complex entity creation tasks with the following results:

- **Overall Success Rate**: 50-65%, up to 80% with all four LLMs available
- **Tool Usage**: Average 2-8 tools per entity creation request
- **LLM Fallback Rate**: 20-40% of requests require multiple LLMs
- **Response Time**: 30-120 seconds per entity creation request
- **Token Usage**: 1K-100K tokens per request (depending on complexity)

### Performance Expectations

- **Success Rate**: 50-65% entities created successfully
- **Response Time**: 30-100 seconds per entity creation request (depending on complexity and LLM)
- **Tool Usage**: 2-8 tool calls per request on average
- **Fallback Rate**: 20-40% of requests require human clarification

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

## 🚀 Getting Started

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_ng.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export GEMINI_API_KEY="your_key"
   export AGENT_PROVIDER="gemini"
   export CMW_DEFAULT_LANGUAGE="en"
   ```

3. **Run the Application**:
   ```bash
   python agent_ng/app_ng_modular.py
   ```

### Development Setup

1. **Activate Virtual Environment**:
   ```bash
   # Windows
   .venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Run Tests**:
   ```bash
   python -m pytest agent_ng/_tests/
   ```

3. **Debug Mode**:
   ```bash
   export CMW_DEBUG_MODE=true
   python agent_ng/app_ng_modular.py
   ```

## Usage

### Live Demo

Visit the Gradio interface to test the agent interactively:

<https://localhost/cmw-platform-agent>

### Programmatic Usage

```python
from agent_ng import NextGenAgent

# Initialize the agent
agent = NextGenAgent()

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

The main agent runtime files are organized into modular directories:

```text
cmw-platform-agent/
├── agent_ng/                    # Next-generation modular agent
│   ├── app_ng_modular.py       # Main Gradio application with modular tabs
│   ├── langchain_agent.py      # LangChain-native agent implementation
│   ├── llm_manager.py          # Multi-provider LLM management
│   ├── error_handler.py        # Error handling with vector similarity
│   ├── message_processor.py    # Message processing and validation
│   ├── response_processor.py   # Response processing and validation
│   ├── stats_manager.py        # Statistics tracking and monitoring
│   ├── trace_manager.py        # Trace logging and debugging
│   ├── debug_streamer.py       # Debug system and logging
│   ├── token_counter.py        # Token usage tracking
│   ├── session_manager.py      # Session management and isolation
│   ├── queue_manager.py        # Request queue management
│   ├── ui_manager.py           # UI state management
│   ├── tool_deduplicator.py    # Tool call deduplication
│   ├── streaming_config.py     # Streaming configuration
│   ├── provider_adapters.py    # LLM provider adapters
│   ├── langchain_memory.py     # LangChain memory management
│   ├── native_langchain_streaming.py  # Native LangChain streaming
│   ├── concurrency_config.py   # Concurrency configuration
│   ├── agent_config.py         # Agent configuration
│   ├── i18n_translations.py    # Internationalization (EN/RU)
│   ├── system_prompt.json      # System prompt configuration
│   └── tabs/                   # Modular tab components
│       ├── chat_tab.py         # Chat interface with quick actions
│       ├── logs_tab.py         # Logs and debugging tab
│       └── stats_tab.py        # Statistics and monitoring tab
├── tools/                      # Tool modules (20+ tools)
│   ├── tools.py               # Core tool functions and definitions
│   ├── applications_tools/    # Application management tools
│   │   ├── tool_list_applications.py
│   │   ├── tool_list_templates.py
│   │   └── tool_platform_entity_url.py
│   ├── attributes_tools/      # Attribute management tools (12 types)
│   │   ├── tools_text_attribute.py
│   │   ├── tools_boolean_attribute.py
│   │   ├── tools_datetime_attribute.py
│   │   ├── tools_decimal_attribute.py
│   │   ├── tools_document_attribute.py
│   │   ├── tools_drawing_attribute.py
│   │   ├── tools_duration_attribute.py
│   │   ├── tools_image_attribute.py
│   │   ├── tools_record_attribute.py
│   │   ├── tools_role_attribute.py
│   │   ├── tools_account_attribute.py
│   │   ├── tools_enum_attribute.py
│   │   ├── tool_delete_attribute.py
│   │   ├── tool_archive_or_unarchive_attribute.py
│   │   └── tool_get_attribute.py
│   ├── templates_tools/       # Template management tools
│   │   └── tool_list_attributes.py
│   ├── tool_utils.py          # Common tool utilities
│   ├── models.py              # Data models and schemas
│   ├── requests_.py           # HTTP request utilities
│   ├── file_utils.py          # File handling utilities
│   └── pdf_utils.py           # PDF processing utilities
└── docs/                      # Documentation and reports
```

## CMW Platform Integration

This agent is designed to work with the Comindware Platform, a business process management and workflow automation platform. The agent can:

- **Create Templates**: Define data structures with custom attributes
- **Configure Workflows**: Set up business processes and automation rules
- **Manage Entities**: Create, update, and configure platform objects
- **API Integration**: Interact with CMW Platform APIs for entity management

For more information about the Comindware Platform, see the [CMW Platform Documentation](https://github.com/arterm-sedov/cbap-mkdocs-ru).

## 📝 Known Issues & Solutions

### Mistral Tool Call IDs
- **Issue**: Mistral requires 9-character alphanumeric tool call IDs
- **Solution**: Automatic ID conversion in `provider_adapters.py`

### OpenRouter Context Limits
- **Issue**: DeepSeek has 163,840 token limit
- **Solution**: Smart context management and chunking

### Session Data Leakage
- **Issue**: Previous versions had global session state
- **Solution**: Session-specific agent instances

## 📞 Support & Troubleshooting

### Common Issues

1. **LLM Not Loading**:
   - Check API keys in environment variables
   - Verify provider availability
   - Check network connectivity

2. **Tool Calls Failing**:
   - Verify tool permissions
   - Check tool configuration
   - Review error logs

3. **Session Issues**:
   - Clear browser cache
   - Restart application
   - Check session isolation

### Debug Information

**Enable Debug Mode**:
```bash
export CMW_DEBUG_MODE=true
export CMW_VERBOSE_LOGGING=true
```

**Check Logs**:
- Use Logs tab in web interface
- Monitor console output
- Review error traces

## 🔮 Future Enhancements

### Planned Features
1. **LangGraph Integration**: Advanced conversation flows
2. **Vector Database**: Enhanced memory and retrieval
3. **Plugin System**: Dynamic tool loading
4. **Advanced Analytics**: Detailed usage insights
5. **Mobile Optimization**: Better mobile experience

### Extension Points
- Custom LLM providers
- Additional tool integrations
- Custom UI themes
- Advanced memory types
- Workflow automation

## Contributing

This is an experimental research project. Contributions are welcome in the form of:

- **Bug Reports**: Issues with the agent's reasoning or tool usage
- **Feature Requests**: New tools or capabilities for CMW Platform integration
- **Performance Improvements**: Optimizations for speed or accuracy
- **Documentation**: Improvements to this README or code comments

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

---