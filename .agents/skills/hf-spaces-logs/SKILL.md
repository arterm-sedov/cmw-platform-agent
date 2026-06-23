---
name: hf-spaces-logs
description: >-
  Access Hugging Face Spaces logs via the HF API (SSE streaming).
  Use when the user needs to view, tail, or stream container logs or build logs
  from a Hugging Face Space (including cmw-copilot), debug Space deployments,
  inspect runtime errors, check build progress, inspect or update Space
  variables/secrets, restart a Space, or verify environment-driven integrations
  such as MCP. Triggers on mentions of
  "HF logs", "Space logs", "Hugging Face logs", "cmw-copilot logs",
  "container logs", "build logs", "Space debug", "HF variables",
  "Space variables", "Space secrets", or "HF_TOKEN".
---

# Hugging Face Spaces — Logs and Runtime Config

Access live SSE log streams and runtime config for Hugging Face Spaces via the
authenticated Hugging Face API.

## Prerequisites

- **`HF_TOKEN`** — Hugging Face API token. Read scope is enough for logs;
  write/admin permission is needed to update Space variables/secrets or restart.
  Load from `.env` or environment. Never print token values.
- **Space ID** — `{username}/{space}` (e.g. `arterm-sedov/cmw-copilot`).

## Endpoints

| Log type | Endpoint |
|----------|----------|
| Container (run) logs | `https://huggingface.co/api/spaces/{username}/{space}/logs/run` |
| Build logs | `https://huggingface.co/api/spaces/{username}/{space}/logs/build` |

Both return **SSE** (Server-Sent Events) streams. Connect with `Accept: text/event-stream`.

## curl

```bash
export HF_TOKEN="hf_..."

# Container logs
curl -N -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/arterm-sedov/cmw-copilot/logs/run"

# Build logs
curl -N -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/arterm-sedov/cmw-copilot/logs/build"
```

## Python (requests with SSE streaming)

```python
import os
import requests

token = os.environ["HF_TOKEN"]
space = "arterm-sedov/cmw-copilot"
log_type = "run"  # or "build"

url = f"https://huggingface.co/api/spaces/{space}/logs/{log_type}"
headers = {"Authorization": f"Bearer {token}"}

with requests.get(url, headers=headers, stream=True) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if line:
            print(line)
```

SSE events follow the standard `data:` / `event:` format. Strip the `data:` prefix to get the log payload.

## Browser / UI

Navigate to the Space, append `?logs=container&logs-api=true` to the URL:

```
https://huggingface.co/spaces/arterm-sedov/cmw-copilot?logs=container&logs-api=true
```

This enables the in-browser log viewer for the container stream.

## Token config

Preferred: `HF_TOKEN` in `.env`. Never commit tokens. Validate before calling:

```python
token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN not set")
```

## Inspect Space Runtime Config

Use `huggingface_hub.HfApi` to inspect Space variables/secrets without exposing
secret values. This is useful when logs show an env-gated integration is skipped
even though variables appear in the HF UI.

```python
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo = "arterm-sedov/cmw-agent"

variables = api.get_space_variables(repo_id=repo)
secrets = api.get_space_secrets(repo_id=repo)

for key, item in sorted(variables.items()):
    print(key, repr(item.value), "description=", repr(item.description))

print("secrets:", sorted(secrets))
```

Watch for variables where `value == ""` but `description` contains the intended
runtime value. HF variable descriptions are metadata only; the app receives the
`value`, not the description.

## Update Space Variables and Restart

Set public, non-secret config with `add_space_variable`. Store credentials and
tokens with `add_space_secret` instead. After env changes, restart the Space and
verify logs.

```python
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo = "arterm-sedov/cmw-agent"

updates = {
    "CMW_MCP_ENABLED": "true",
    "CMW_MCP_TOOL_NAME_PREFIX": "true",
    "CMW_MCP_ALLOWED_SERVERS": "comindware_kb",
    "CMW_MCP_ALLOWED_HOSTS": "ennoia.slickjump.org",
}

for key, value in updates.items():
    api.add_space_variable(repo_id=repo, key=key, value=value)

api.restart_space(repo_id=repo)
```

For secrets:

```python
api.add_space_secret(repo_id=repo, key="REMOTE_MCP_BEARER_TOKEN", value=token)
```

## MCP Verification Pattern

After restart, tail run logs and look for positive connection evidence:

```text
Connecting to StreamableHTTP endpoint: https://.../gradio_api/mcp/
Negotiated protocol version: ...
Loaded 1 MCP tool(s) from server(s): comindware_kb
```

If there are no MCP log lines at all, first inspect `CMW_MCP_ENABLED` in Space
variables. Empty values are treated as false by the app.
