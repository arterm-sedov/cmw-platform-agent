# LangChain & Gradio Version Analysis Report

**Report Date:** 2026-04-22
**Author:** opencode AI agent
**Purpose:** Document langchain/gradio versions across all cmw-platform repos and migration paths

---

## Executive Summary

This report analyzes the state of langchain and gradio dependencies across multiple repositories. Key findings:

1. **cmw-rag** is the most advanced - successfully running langchain 1.2.10+ and gradio 6.8.0+
2. **cmw-platform-agent** (your repo) is pinned to older 0.3.x ecosystem
3. **cmw-platform-agent-langchain-1** has partial 1.2.7 migration but with exact pins
4. **cmw-platform-agent-gradio-6** has gradio 6.x but keeps langchain 0.3.x
5. Migration from 0.3.x to 1.x is complex and requires code changes

---

## Repository Version Comparison

| Repo | langchain | langchain-core | gradio | Approach |
|------|-----------|----------------|-------|----------|
| **cmw-platform-agent** (your repo) | 0.3.27 | 0.3.79 | 5.49.1 | Exact pins |
| **cmw-rag** | >=1.2.10 | >=1.2.17 | >=6.8.0 | Flexible >= |
| **cmw-rag-old** | >=0.1.0 | >=0.1.0 | >=6.0.0 | Flexible >= |
| **cmw-rag-eu-ai-pack-hygiene** | >=1.2.10 | >=1.2.17 | >=6.8.0 | Same as cmw-rag |
| **cmw-platform-agent-langchain-1** | 1.2.7 | 1.2.7 | 5.49.1 | Exact pins |
| **cmw-platform-agent-gradio-6** | 0.3.27 | 0.3.79 | >=6.4.0 | Mixed (gradio >=) |
| **cmw-platform-agent_old** | No pin | (various) | No pin | Unpinned |

---

## Latest Available Versions (as of 2026-04-22)

### LangChain Ecosystem
| Package | Latest PyPI | Your Repo |
|---------|------------|----------|
| langchain | 1.2.15 | 0.3.27 |
| langchain-core | 1.3.0 | 0.3.79 |
| langchain-openai | 1.1.15 | 0.3.35 |
| langchain-groq | 1.1.15 | 0.3.8 |
| langchain-google-genai | 4.1.0 | 2.0.10 |
| langchain-huggingface | 1.2.1 | 0.3.1 |
| langchain-community | 0.4.1 | 0.3.31 |
| langchain-text-splitters | 1.1.0 | 0.3.11 |

### Gradio Ecosystem
| Package | Latest PyPI | Your Repo |
|---------|------------|----------|
| gradio | 6.12.0 | 5.49.1 |
| gradio_client | (linked to gradio) | 1.13.3 |
| starlette | 1.0.0 | 0.48.0 |

---

## Detailed Analysis

### cmw-rag (Most Advanced)

```
langchain>=1.2.10
langchain-core>=1.2.17
gradio[mcp]>=6.8.0
```

- Uses flexible `>=` constraints which auto-resolve to latest compatible
- Successfully migrated to langchain 1.x ecosystem
- Uses MCP support in gradio (`gradio[mcp]`)
- Includes new integrations: `langchain-openrouter`, `langchain-chroma`

**Code Pattern (from llm_manager.py):**
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
# Uses modern langchain 1.x imports
```

**Agent Factory Pattern (from agent_factory.py):**
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, before_model
# Uses new middleware system in langchain 1.x
```

---

### cmw-platform-agent (Your Repository)

```
langchain==0.3.27
langchain-core==0.3.79
gradio==5.49.1
```

- Uses exact pins (`==`) locking to July 2025 versions
- Stuck on 0.3.x (pre-1.0) ecosystem
- No MCP support
- Many integration packages at 0.3.x versions

**Issues with Current Setup:**
- Cannot use `langchain-mcp-adapters` without upgrading
- Would need to upgrade entire langchain ecosystem at once
- Breaking changes between 0.3.x and 1.x

---

### cmw-platform-agent-langchain-1

```
langchain==1.2.7
langchain-core==1.2.7
gradio==5.49.1
```

- Successfully upgraded to langchain 1.x but with exact pins
- gradio still at 5.x (older than cmw-rag)
- Mix of new and old approaches on exact pins

---

### cmw-platform-agent-gradio-6

```
langchain==0.3.27
langchain-core==0.3.79
gradio>=6.4.0
```

- Hybrid approach: gradio 6.x but langchain 0.3.x
- This combination causes conflicts (as you discovered)
- starlette upgrades from 0.48.0 to 1.0.0 breaks gradio 5.x compatibility

---

## Migration Complexity Analysis

### Version Timeline

```
0.3.x series (your repo)
    │
    ├─► Oct 2025: langchain 1.0 GA (breaks 0.3.x)
    │
    └─► Current: 1.2.x series
              │
              ├─► Latest: 1.2.15 (langchain), 1.3.0 (langchain-core)
              │
              └─► Latest: gradio 6.12.0 (requires starlette 1.0.0)
```

### Breaking Changes from 0.3 to 1.0+

1. **Package namespace reduced** - legacy code moved to `langchain-classic`
2. **Python 3.10+ required** (must drop 3.9 support)
3. **Agent API changed** - `create_agent` replaces `initialize_agent`
4. **Middleware system** - new `SummarizationMiddleware` etc.
5. **Tool definitions** - new tool schema
6. **Chat model return types** - changed response format

### Dependency Conflicts (Why Your Install Failed)

```
langchain-mcp-adapters requires:
├─ langchain-core <1.0 (ERROR: requires >1.0)
├─ mcp >= 1.9.2
└─ starlette >= 1.0 (ERROR: conflicts with gradio 5.x)

gradio 5.x requires:
└─ starlette <1.0

Result: IMPOSSIBLE to satisfy both without upgrading gradio
```

---

## Recommended Paths

### Option 1: Stay on 0.3.x (Conservative)

Keep current pinned versions. Pros:
- Stable, works
- No code changes needed

Cons:
- Cannot use MCP/new features
- Will fall further behind

### Option 2: Follow cmw-rag Pattern (Recommended)

Migrate to langchain 1.x + gradio 6.x like cmw-rag:

1. Change `==` to `>=` for flexibility
2. Upgrade langchain packages together
3. Upgrade gradio to 6.x
4. Update import paths for new API

**Target requirements.txt:**
```
langchain>=1.2.10
langchain-core>=1.2.17
langchain-openai>=1.0.0
langchain-groq>=1.0.0
langchain-google-genai>=0.1.0
gradio[mcp]>=6.8.0
```

### Option 3: Full Latest (Aggressive)

Match cmw-rag exactly:
```
langchain>=1.2.10
langchain-core>=1.2.17
gradio>=6.8.0
```

Same as cmw-rag - they update faster.

---

## Code Migration Notes

### Import Changes

```python
# OLD (0.3.x)
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

# NEW (1.x)
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
```

### Agent Factory Pattern

cmw-rag's new pattern:
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    llm=model,
    tools=tools,
    middleware=[SummarizationMiddleware(...)],
)
```

---

## References

- LangChain Migration Guide: https://docs.langchain.com/oss/python/migrate/langchain-v1
- LangChain 1.0 Announcement: https://www.langchain.com/blog/langchain-langchain-1-0-alpha-releases
- Gradio Changelog: https://www.gradio.app/changelog

---

## Conclusion

The cmw-rag repository has successfully navigated the langchain upgrade path by:
1. Using flexible `>=` constraints (not exact pins)
2. Upgrading all langchain ecosystem packages together
3. Adopting gradio 6.x with MCP support
4. Updating code to use new langchain 1.x API

Your repo can follow the same pattern but will require code changes to support the new API.

**Next Steps (if choosing to upgrade):**
1. Create new requirements.txt with `>=` constraints
2. Test import changes from agent code
3. Update tool definitions if needed
4. Run full test suite for regressions