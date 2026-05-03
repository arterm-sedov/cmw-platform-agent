# MCP Integration Implementation Plan

**Plan Date:** 2026-04-22
**Goal:** Connect cmw-platform-agent to external MCP server
**Target Endpoint:** https://ennoia.slickjump.org/gradio_api/mcp/

---

## Summary of Findings

| Approach | Status | Issue |
|----------|--------|-------|
| langchain-mcp-adapters | ❌ Incompatible | Requires langchain-core>=1.0, starlette>=1.0 |
| mcp-use | ⚠️ Works but breaks UI | starlette conflict with Gradio 5.x |
| **Upgrade path (recommended)** | 🔄 Future | Requires full ecosystem upgrade |

---

## Recommended Implementation Path

Follow cmw-rag pattern: upgrade to langchain 1.x + gradio 6.x

### Phase 1: Upgrade Dependencies

- [ ] **1.1** Update requirements.txt with flexible `>=` constraints
- [ ] **1.2** Change `langchain==0.3.27` → `langchain>=1.2.10`
- [ ] **1.3** Change `langchain-core==0.3.79` → `langchain-core>=1.2.17`
- [ ] **1.4** Change `gradio==5.49.1` → `gradio[mcp]>=6.8.0`
- [ ] **1.5** Update all langchain-* packages to 1.x versions

### Phase 2: Code Migration

- [ ] **2.1** Update import paths (langchain.chat_models → langchain_openai)
- [ ] **2.2** Update agent creation (initialize_agent → create_agent)
- [ ] **2.3** Update middleware if used
- [ ] **2.4** Update tool definitions for new schema

### Phase 3: Test & Deploy

- [ ] **3.1** Run `pip install -r requirements.txt` in clean venv
- [ ] **3.2** Run lint: `ruff check agent_ng/`
- [ ] **3.3** Run typecheck: `mypy agent_ng/`
- [ ] **3.4** Run tests: `python -m pytest agent_ng/_tests/`
- [ ] **3.5** Test MCP server connection manually

---

## Implementation Details

### Step 1.1: Update requirements.txt

```diff
- langchain==0.3.27
- langchain-core==0.3.79
- langchain-openai==0.3.35
- langchain-groq==0.3.8
- langchain-google-genai==2.0.10
- langchain-huggingface==0.3.1
- langchain-community==0.3.31
- langchain-text-splitters==0.3.11
- gradio==5.49.1
- gradio_client==1.13.3
+ langchain>=1.2.10
+ langchain-core>=1.2.17
+ langchain-openai>=1.0.0
+ langchain-groq>=1.0.0
+ langchain-google-genai>=0.1.0
+ langchain-huggingface>=1.0.0
+ langchain-community>=0.3.0
+ langchain-text-splitters>=0.3.0
+ gradio[mcp]>=6.8.0
```

### Step 2.1: Import Path Changes

```python
# OLD (0.3.x)
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

# NEW (1.x)
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
```

### Step 2.2: Agent Factory Pattern

```python
# OLD (0.3.x)
agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.CONVERSATION_REACT_DESCRIPTION,
)

# NEW (1.x)
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    llm=model,
    tools=tools,
    middleware=[SummarizationMiddleware(...)],
)
```

---

## Checkpoints

### CP1: Dependencies Install
- [ ] `pip install -r requirements.txt` succeeds
- [ ] No conflicts reported
- [ ] `python -c "import gradio; print(gradio.__version__)"` shows 6.x

### CP2: Imports Work
- [ ] `from langchain_openai import ChatOpenAI` succeeds
- [ ] `from langchain.agents import create_agent` succeeds
- [ ] `from gradio import Col` succeeds (new Component API)

### CP3: MCP Integration
- [ ] Gradio MCP client connects to target endpoint
- [ ] Tools are discovered and callable

### CP4: Full Test Suite
- [ ] All lint checks pass
- [ ] All type checks pass
- [ ] All tests pass

---

## Rollback Plan (if needed)

If upgrade fails:
1. Revert requirements.txt to previous pins
2. Restore venv from backup
3. Document issue in progress_reports/

---

## References

- cmw-rag requirements.txt (working example)
- LangChain Migration Guide: https://docs.langchain.com/oss/python/migrate/langchain-v1
- Gradio MCP: https://www.gradio.app/guides/mcp-models-tools