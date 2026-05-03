"""
OpenRouter usage/cost probe (LangChain ChatOpenAI)
=================================================

Goal:
- Verify where OpenRouter returns `usage.cost` (and prompt cache details) when called via LangChain.
- Print raw metadata containers so we can adjust parsing if needed.

Prereqs (set in .env or environment):
- OPENROUTER_API_KEY
- OPENROUTER_BASE_URL (optional; defaults to https://openrouter.ai/api/v1)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

from dotenv import load_dotenv
import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Allow running as a standalone script from any CWD
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_ng.token_counter import ConversationTokenTracker


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)

def _extract_cost_like(meta: Any) -> tuple[float | None, str]:
    """Best-effort: find OpenRouter cost in common containers."""
    if not isinstance(meta, dict):
        return (None, "meta:not-dict")
    # OpenRouter via ChatOpenAI frequently exposes: response_metadata["token_usage"]["cost"]
    tu = meta.get("token_usage")
    if isinstance(tu, dict) and "cost" in tu:
        try:
            return (float(tu.get("cost") or 0.0), "response_metadata.token_usage.cost")
        except Exception:
            return (None, "response_metadata.token_usage.cost:bad-float")
    usage = meta.get("usage")
    if isinstance(usage, dict) and "cost" in usage:
        try:
            return (float(usage.get("cost") or 0.0), "response_metadata.usage.cost")
        except Exception:
            return (None, "response_metadata.usage.cost:bad-float")
    return (None, "cost:not-found")

def _query_generation_stats(*, base_url: str, api_key: str, generation_id: str) -> dict[str, Any] | None:
    """Query OpenRouter generation stats by id (gen-...)."""
    if not generation_id:
        return None
    url = base_url.rstrip("/") + "/generation"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"id": generation_id}
    delays_s = [0.0, 0.5, 1.0, 2.0, 4.0]
    with httpx.Client(timeout=30.0) as client:
        for attempt, delay_s in enumerate(delays_s, start=1):
            if delay_s:
                time.sleep(delay_s)
            try:
                resp = client.get(url, headers=headers, params=params)
                if resp.status_code == 404:
                    # Often eventual consistency; retry a few times.
                    print(
                        f"[generation] attempt {attempt}/{len(delays_s)}: 404 Not Found (retrying...)",
                        flush=True,
                    )
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data if isinstance(data, dict) else None
            except Exception as exc:
                body = None
                try:
                    body = resp.text  # type: ignore[name-defined]
                except Exception:
                    body = None
                print(
                    f"[generation] attempt {attempt}/{len(delays_s)} failed for {generation_id!r}: {exc!r} body={body!r}",
                    flush=True,
                )
                return None
    return None

def main() -> None:
    load_dotenv()

    # Keep probe output readable
    logging.getLogger().setLevel(logging.WARNING)
    os.environ.setdefault("LANGSMITH_TRACING", "false")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_PROBE_MODEL", "z-ai/glm-4.7")
    prompt = os.getenv(
        "OPENROUTER_PROBE_PROMPT",
        "Что ты умеешь? Ровно в пятидесяти словах",
    )
    do_stream = os.getenv("OPENROUTER_PROBE_STREAM", "true").lower() in {"1", "true", "yes"}
    include_usage = os.getenv("OPENROUTER_PROBE_STREAM_INCLUDE_USAGE", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    max_tokens = int(os.getenv("OPENROUTER_PROBE_MAX_TOKENS", "200"))

    print(f"base_url={base_url}", flush=True)
    print(f"model={model}", flush=True)
    print(f"stream={do_stream}", flush=True)
    print(f"stream_include_usage={include_usage}", flush=True)
    print(f"max_tokens={max_tokens}", flush=True)
    print(f"prompt={prompt!r}", flush=True)

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        streaming=False,
        max_tokens=max_tokens,
    )

    messages = [HumanMessage(content=prompt)]
    resp = llm.invoke(messages)

    print("\n=== Response type ===")
    print(type(resp))
    print("\n=== Response content (first 300 chars) ===")
    print((getattr(resp, "content", "") or "")[:300])

    # These names are typical for LangChain AIMessage
    usage_metadata = getattr(resp, "usage_metadata", None)
    response_metadata = getattr(resp, "response_metadata", None)
    additional_kwargs = getattr(resp, "additional_kwargs", None)
    usage_attr = getattr(resp, "usage", None)

    print("\n=== usage_metadata (LangChain-normalized) ===")
    print(_safe_json(usage_metadata))

    print("\n=== response_metadata (provider-raw container) ===")
    print(_safe_json(response_metadata))
    cost_val, cost_path = _extract_cost_like(response_metadata)
    print(f"\n=== response_metadata cost probe ===\n{cost_path} = {cost_val}")
    gen_id = response_metadata.get("id") if isinstance(response_metadata, dict) else None
    print(f"\n=== response_metadata id probe ===\nid = {gen_id!r}")

    print("\n=== additional_kwargs (provider extras) ===")
    print(_safe_json(additional_kwargs))

    print("\n=== response.usage (if present) ===")
    print(_safe_json(usage_attr))

    # Our pipeline extraction (what UI uses)
    tracker = ConversationTokenTracker()
    token_count = tracker.track_llm_response(resp, messages)
    print("\n=== Parsed by ConversationTokenTracker ===")
    print(_safe_json(token_count.__dict__))

    stats = tracker.get_cumulative_stats()
    print("\n=== Cumulative stats ===")
    print(_safe_json({k: stats.get(k) for k in (
        "turn_cost",
        "conversation_cost",
        "total_cost",
        "last_input_tokens",
        "last_output_tokens",
        "last_cached_tokens",
        "last_cache_write_tokens",
    )}))

    if isinstance(gen_id, str) and gen_id.startswith("gen-"):
        print("\n\n================ GENERATION STATS (non-streaming) ================\n")
        gen_stats = _query_generation_stats(base_url=base_url, api_key=api_key, generation_id=gen_id)
        print(_safe_json(gen_stats))

    if do_stream:
        print("\n\n================ STREAMING PROBE ================\n")
        try:
            llm_s = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0,
                streaming=True,
                stream_usage=True,
                max_tokens=max_tokens,
                model_kwargs=(
                    {"stream_options": {"include_usage": True}}
                    if include_usage
                    else {}
                ),
            )

            tracker_s = ConversationTokenTracker()
            tracker_s.begin_turn()

            last_chunk = None
            chunks = 0
            first_nonempty_response_meta: dict[str, Any] | None = None
            first_seen_gen_id: str | None = None
            first_seen_usage: dict[str, Any] | None = None
            first_seen_token_usage: dict[str, Any] | None = None
            for chunk in llm_s.stream(messages):
                chunks += 1
                last_chunk = chunk
                rm = getattr(chunk, "response_metadata", None)
                um = getattr(chunk, "usage_metadata", None)

                if isinstance(rm, dict) and rm and first_nonempty_response_meta is None:
                    first_nonempty_response_meta = rm
                if isinstance(rm, dict) and isinstance(rm.get("id"), str) and first_seen_gen_id is None:
                    first_seen_gen_id = rm.get("id")
                if isinstance(rm, dict) and isinstance(rm.get("usage"), dict) and first_seen_usage is None:
                    first_seen_usage = rm.get("usage")
                if isinstance(rm, dict) and isinstance(rm.get("token_usage"), dict) and first_seen_token_usage is None:
                    first_seen_token_usage = rm.get("token_usage")

                cost_here, cost_path_here = _extract_cost_like(rm)
                if cost_here is not None or (isinstance(rm, dict) and ("token_usage" in rm or "usage" in rm)):
                    print(
                        f"[chunk {chunks}] type={type(chunk)} cost_path={cost_path_here} cost={cost_here}"
                    )
                    if isinstance(rm, dict) and "token_usage" in rm:
                        print(_safe_json({"token_usage": rm.get("token_usage")}))
                    if isinstance(rm, dict) and "usage" in rm:
                        print(_safe_json({"usage": rm.get("usage")}))

                if isinstance(um, dict) and chunks <= 2:
                    print(f"[chunk {chunks}] usage_metadata={list(um.keys())}")

                # accumulate per-iteration usage (best-effort)
                try:
                    tracker_s.update_turn_usage_from_api(chunk)
                except Exception:
                    pass

            print(f"\nstreamed_chunks={chunks}")
            print(f"last_chunk_type={type(last_chunk)}")
            print("\n=== first_nonempty_chunk.response_metadata ===")
            print(_safe_json(first_nonempty_response_meta))
            print("\n=== first_seen_gen_id (from chunks) ===")
            print(repr(first_seen_gen_id))
            print("\n=== first_seen_usage (from chunks) ===")
            print(_safe_json(first_seen_usage))
            print("\n=== first_seen_token_usage (from chunks) ===")
            print(_safe_json(first_seen_token_usage))
            if last_chunk is not None:
                print("\n=== last_chunk.response_metadata ===")
                print(_safe_json(getattr(last_chunk, "response_metadata", None)))
                print("\n=== last_chunk.usage_metadata ===")
                print(_safe_json(getattr(last_chunk, "usage_metadata", None)))
                print("\n=== last_chunk.additional_kwargs ===")
                print(_safe_json(getattr(last_chunk, "additional_kwargs", None)))
                print("\n=== last_chunk.id ===")
                print(repr(getattr(last_chunk, "id", None)))
                try:
                    print("\n=== last_chunk.__dict__ keys ===")
                    d = getattr(last_chunk, "__dict__", {}) or {}
                    print(sorted(d.keys()))
                except Exception:
                    pass

            finalized = tracker_s.finalize_turn_usage(last_chunk, messages)
            print("\n=== finalize_turn_usage() result ===")
            print(_safe_json(finalized.__dict__ if finalized else None))
            print("\n=== streaming tracker cumulative stats ===")
            print(_safe_json(tracker_s.get_cumulative_stats()))

            if isinstance(first_seen_gen_id, str) and first_seen_gen_id.startswith("gen-"):
                print("\n\n================ GENERATION STATS (streaming) ================\n")
                gen_stats_s = _query_generation_stats(
                    base_url=base_url, api_key=api_key, generation_id=first_seen_gen_id
                )
                print(_safe_json(gen_stats_s))
            else:
                print(
                    "\n[generation] No gen-... id observed in LangChain streaming chunks; cannot query /generation.\n",
                    flush=True,
                )
        except Exception as exc:
            print("\nSTREAMING PROBE FAILED:")
            print(repr(exc))

if __name__ == "__main__":
    main()

