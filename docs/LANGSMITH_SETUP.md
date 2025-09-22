# LangSmith Observability Setup
==============================

This guide shows how to enable LangSmith observability in the CMW Platform Agent for tracing LLM calls and application flows.

## Prerequisites

1. **LangSmith Account**: Sign up at [smith.langchain.com](https://smith.langchain.com)
2. **LangSmith API Key**: Get your API key from the LangSmith dashboard
3. **LangSmith Package**: Install with `pip install langsmith`

## Environment Configuration

Add these environment variables to your `.env` file:

```bash
# LangSmith Observability
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_WORKSPACE_ID=your_workspace_id_here  # Optional
LANGSMITH_PROJECT=cmw-platform-agent  # Optional
```

## Installation

1. Install the LangSmith package:
   ```bash
   pip install langsmith
   ```

2. Set your environment variables in `.env`:
   ```bash
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your_api_key_here
   ```

3. Run the application:
   ```bash
   python agent_ng/app_ng_modular.py
   ```

## What Gets Traced

The lean LangSmith implementation traces:

1. **LLM Calls**: All OpenAI-compatible model calls (OpenRouter, Mistral)
2. **Application Flow**: Main streaming chat method
3. **Tool Calls**: Tool execution within conversations
4. **Session Data**: User session isolation

## Viewing Traces

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Navigate to your project (default: "cmw-platform-agent")
3. View traces in real-time as users interact with the agent

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGSMITH_TRACING` | Enable/disable tracing | `false` |
| `LANGSMITH_API_KEY` | Your LangSmith API key | Required |
| `LANGSMITH_WORKSPACE_ID` | Workspace ID (optional) | Auto-detected |
| `LANGSMITH_PROJECT` | Project name for traces | `cmw-platform-agent` |

## Troubleshooting

### Tracing Not Working
- Check that `LANGSMITH_TRACING=true`
- Verify your API key is correct
- Ensure `langsmith` package is installed

### No Traces Appearing
- Check your LangSmith dashboard
- Verify the project name matches
- Check network connectivity to LangSmith

### Performance Impact
- LangSmith tracing adds minimal overhead
- Traces are sent asynchronously
- Can be disabled by setting `LANGSMITH_TRACING=false`

## Implementation Details

The implementation uses:
- `langsmith.wrappers.wrap_openai` for LLM call tracing
- `@traceable` decorator for application flow tracing
- Automatic detection of OpenAI-compatible models
- Session-aware tracing for multi-user support

## Next Steps

1. Set up your LangSmith account
2. Configure environment variables
3. Run the application
4. View traces in the LangSmith dashboard
5. Use traces for debugging and optimization
