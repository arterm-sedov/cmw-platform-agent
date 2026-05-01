# Knowledge Base

When needing platform expertise, use the MCP knowledge base tool:

```python
from cmw_platform_knowledge-base_get_knowledge_base_articles import get_knowledge_base_articles

# Search for relevant platform documentation
articles = get_knowledge_base_articles.invoke({
    "query": "attribute schema edit partial update",
    "top_k": 5
})
```

## When to Use

- Uncertain about attribute types, formats, or API behavior
- Need examples of proper attribute configuration
- Exploring platform best practices for specific operations
- Troubleshooting API errors or unexpected behavior

## ⚠️ Do NOT Use `ask_comindware`

`ask_comindware` provides conversational answers. Use `get_knowledge_base_articles` for **programmatic access** to documentation.

---

→ Tool signature and parameters: [tool_inventory.md](tool_inventory.md#knowledge-base-tools)
