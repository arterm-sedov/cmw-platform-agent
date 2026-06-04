---
name: hf-spaces-logs
description: >-
  Access Hugging Face Spaces logs via the HF API (SSE streaming).
  Use when the user needs to view, tail, or stream container logs or build logs
  from a Hugging Face Space (including cmw-copilot), debug Space deployments,
  inspect runtime errors, or check build progress. Triggers on mentions of
  "HF logs", "Space logs", "Hugging Face logs", "cmw-copilot logs",
  "container logs", "build logs", or "Space debug".
---

# Hugging Face Spaces — Logs API

Access live SSE log streams for any HF Space via bearer-token authenticated API.

## Prerequisites

- **`HF_TOKEN`** — Hugging Face API token (read scope is sufficient). Load from `.env` or environment.
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
