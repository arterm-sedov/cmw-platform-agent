# Next-Generation LLM Agent App

A clean, modern implementation of an LLM agent application using the latest Gradio features and a modular architecture.

## 🚀 Features

### Modern Architecture
- **Clean separation of concerns** with modular components
- **Async initialization** for better performance
- **Real-time streaming** with metadata support
- **Error handling and fallback** logic
- **Conversation management** with history

### Gradio Integration
- **Latest Gradio features** (ChatInterface, Blocks, ChatMessage)
- **Real-time streaming** with native support
- **Copy to clipboard** functionality
- **Responsive design** with custom CSS
- **Tool usage visualization** with collapsible sections

### LLM Management
- **Persistent LLM instances** that survive page refreshes
- **Automatic provider fallback** when one fails
- **Tool calling support** with real-time feedback
- **Health monitoring** and status reporting

## 📁 File Structure

```
├── agent_ng.py          # Next-generation agent orchestrator
├── app_ng.py            # Modern Gradio application
├── test_app_ng.py       # Test script for verification
├── README_NEXTGEN.md    # This documentation
└── core_agent.py        # Core agent module (existing)
    ├── llm_manager.py   # LLM management (existing)
    ├── error_handler.py # Error handling (existing)
    ├── streaming_manager.py # Streaming support (existing)
    └── ...              # Other modular components
```

## 🛠️ Installation

1. **Install dependencies:**
   ```bash
   pip install gradio langchain langchain-core
   ```

2. **Set up environment variables:**
   ```bash
   # At least one LLM provider API key is required
   export GEMINI_KEY="your_gemini_key"
   export GROQ_API_KEY="your_groq_key"
   export OPENROUTER_API_KEY="your_openrouter_key"
   # ... other providers
   ```

## 🚀 Usage

### Quick Start
```bash
python app_ng.py
```

The app will:
1. Start the Gradio interface
2. Initialize LLM providers in the background
3. Show initialization logs in the "Logs" tab
4. Display agent status in the sidebar
5. Enable chat functionality when ready

### Testing
```bash
python test_app_ng.py
```

This will verify that:
- Agent initializes correctly
- LLM providers are available
- Chat functionality works
- App components are functional

## 🎯 Key Improvements

### Over Legacy App
- **Clean code**: No legacy dependencies or complex workarounds
- **Modern patterns**: Uses latest Gradio features and async/await
- **Better UX**: Real-time streaming with proper metadata
- **Modular design**: Easy to extend and maintain
- **Error handling**: Robust error recovery and user feedback

### Architecture Benefits
- **Separation of concerns**: Each module has a single responsibility
- **Testability**: Easy to unit test individual components
- **Extensibility**: Simple to add new LLM providers or tools
- **Maintainability**: Clear code structure and documentation

## 🔧 Configuration

### Agent Configuration
```python
config = AgentConfig(
    enable_vector_similarity=True,
    max_conversation_history=50,
    max_tool_calls=10,
    similarity_threshold=0.95,
    streaming_chunk_size=100,
    enable_tool_calling=True,
    enable_streaming=True
)
```

### LLM Providers
The app automatically detects and initializes available LLM providers:
- **Google Gemini** (gemini-2.5-pro)
- **Groq** (llama-3.3-70b-versatile)
- **OpenRouter** (deepseek/deepseek-chat-v3.1:free)
- **Mistral AI** (mistral-large-latest)
- **HuggingFace** (Qwen/Qwen2.5-Coder-32B-Instruct)
- **GigaChat** (GigaChat-2)

## 📊 Monitoring

### Real-time Status
- **Agent readiness**: Shows when LLM is available
- **Model information**: Current provider and model details
- **Tool availability**: Number of tools loaded
- **Conversation stats**: Message counts and history

### Logs and Debugging
- **Initialization logs**: Real-time startup progress
- **Error tracking**: Detailed error information
- **Performance metrics**: Response times and usage stats
- **Tool execution**: Step-by-step tool usage

## 🎨 UI Features

### Chat Interface
- **Modern design** with gradient backgrounds
- **Real-time streaming** with typing indicators
- **Tool usage visualization** in collapsible sections
- **Copy to clipboard** for responses
- **Message history** with proper formatting

### Sidebar
- **Agent status** with live updates
- **Model information** and health status
- **Quick action buttons** for common tasks
- **Statistics** and performance metrics

### Tabs
- **Chat**: Main conversation interface
- **Logs**: Initialization and debug information
- **Stats**: Performance and usage statistics

## 🔄 Streaming

### Real-time Updates
The app provides real-time streaming for:
- **LLM responses**: Character-by-character streaming
- **Tool execution**: Step-by-step tool usage
- **Thinking process**: Internal reasoning (when available)
- **Error handling**: Immediate error feedback

### Event Types
- `content`: Main response content
- `thinking`: LLM reasoning process
- `tool_use`: Tool execution steps
- `success`: Operation completion
- `error`: Error messages
- `info`: General information

## 🧪 Testing

### Automated Tests
```bash
python test_app_ng.py
```

### Manual Testing
1. **Start the app**: `python app_ng.py`
2. **Check initialization**: Monitor the "Logs" tab
3. **Test chat**: Send messages in the "Chat" tab
4. **Verify streaming**: Watch real-time responses
5. **Test tools**: Ask questions that require tool usage

## 🚀 Deployment

### Local Development
```bash
python app_ng.py
```

### Production
```bash
# Set environment variables
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT="7860"

# Run the app
python app_ng.py
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "app_ng.py"]
```

## 🤝 Contributing

### Adding New Features
1. **Modular approach**: Add new functionality to appropriate modules
2. **Test coverage**: Include tests for new features
3. **Documentation**: Update this README with new features
4. **Error handling**: Ensure robust error handling

### Code Style
- **Type hints**: Use proper type annotations
- **Docstrings**: Document all public methods
- **Error handling**: Use try/catch with meaningful messages
- **Async patterns**: Use async/await for I/O operations

## 📝 License

This project is part of the CMW Platform Agent system and follows the same licensing terms.

## 🆘 Support

For issues or questions:
1. Check the logs in the "Logs" tab
2. Run the test script: `python test_app_ng.py`
3. Verify environment variables are set correctly
4. Check that at least one LLM provider is available

---

**Next-Gen App**: Clean, modern, and ready for the future! 🚀
