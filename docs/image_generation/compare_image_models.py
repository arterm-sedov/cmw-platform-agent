"""
Compare registered image models on CMW-relevant prompts.

Runs live calls against every model in
:data:`agent_ng.image_models.IMAGE_MODELS`, saves each output under
``docs/image_generation/progress_reports/YYYYMMDD_<slug>_<prompt>.png``,
and writes a summary table alongside the images.

Use this when:
- Evaluating a new model before adding it to the registry.
- Re-benchmarking after a provider update, to justify the default.
- Documenting Russian-text / Cyrillic rendering quality for audit.

Run:  ``python docs/image_generation/compare_image_models.py``

Reads ``OPENROUTER_API_KEY`` from ``.env``. Each call costs real money —
typical full run $0.50–$1.00 depending on which models are registered.
"""

from __future__ import annotations

import os
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from agent_ng.image_engine import ImageEngine  # noqa: E402
from agent_ng.image_models import get_image_models  # noqa: E402

# Two CMW-relevant prompts: one business graphic, one with Cyrillic text.
_PROMPT_BUSINESS = (
    "Minimalist flat business icon: a workflow with three connected process "
    "boxes and arrows between them, corporate blue palette, clean white "
    "background, suitable for a SaaS product dashboard."
)
_PROMPT_RUSSIAN = (
    "A clean business badge with the Cyrillic text 'Сделка закрыта' "
    "(meaning: 'Deal closed') centered in bold sans-serif, green checkmark "
    "icon above the text, white background, flat corporate style."
)


def _png_dims(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", data[16:24])


def _jpeg_dims(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            height = struct.unpack(">H", data[i + 5 : i + 7])[0]
            width = struct.unpack(">H", data[i + 7 : i + 9])[0]
            return width, height
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        i += 2 + seg_len
    return None


def _dims(data: bytes) -> tuple[int, int] | None:
    return _png_dims(data) or _jpeg_dims(data)


def _run(
    engine: ImageEngine,
    model: str,
    prompt_key: str,
    prompt: str,
    out_dir: Path,
    date_prefix: str,
) -> dict:
    start = time.time()
    try:
        result = engine.generate(prompt=prompt, model=model)
    except Exception as exc:  # noqa: BLE001
        return {
            "model": model,
            "prompt": prompt_key,
            "ok": False,
            "error": f"exception: {exc}",
            "seconds": round(time.time() - start, 2),
        }
    elapsed = round(time.time() - start, 2)
    if not result.success:
        return {
            "model": model,
            "prompt": prompt_key,
            "ok": False,
            "error": result.error,
            "seconds": elapsed,
            "cost": result.cost,
        }
    ext = ".png" if (result.mime_type or "").endswith("png") else ".jpg"
    safe = model.replace("/", "__").replace(":", "_")
    fname = f"{date_prefix}_{safe}_{prompt_key}{ext}"
    path = out_dir / fname
    path.write_bytes(result.image_bytes or b"")
    dims = _dims(result.image_bytes or b"")
    return {
        "model": model,
        "prompt": prompt_key,
        "ok": True,
        "seconds": elapsed,
        "cost": result.cost,
        "size_kb": round(len(result.image_bytes or b"") / 1024, 1),
        "dims": f"{dims[0]}x{dims[1]}" if dims else "?",
        "file": str(path.relative_to(_REPO)).replace("\\", "/"),
    }


def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set; aborting")
        sys.exit(1)

    out_dir = _REPO / "docs" / "image_generation" / "progress_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_prefix = datetime.now().strftime("%Y%m%d")

    engine = ImageEngine()
    rows: list[dict] = []

    for model in get_image_models():
        print(f"\n=== {model} ===")
        for key, prompt in (
            ("business", _PROMPT_BUSINESS),
            ("russian", _PROMPT_RUSSIAN),
        ):
            row = _run(engine, model, key, prompt, out_dir, date_prefix)
            rows.append(row)
            if row["ok"]:
                print(
                    f"  [{key:<8}] OK   {row['seconds']:>6}s  "
                    f"${row.get('cost', 0.0) or 0.0:.4f}  "
                    f"{row.get('size_kb', 0):>6} KB  "
                    f"{row.get('dims', '?'):>10}  -> {row['file']}"
                )
            else:
                print(
                    f"  [{key:<8}] FAIL {row['seconds']:>6}s  "
                    f"cost=${row.get('cost', 0.0) or 0.0:.4f}  "
                    f"err={row.get('error', '?')[:120]}"
                )

    summary = out_dir / f"{date_prefix}_model_comparison.md"
    lines = [
        "# Image model comparison\n",
        f"Run date: {datetime.now().isoformat()}\n",
        "| model | prompt | ok | time(s) | cost($) | size(KB) | dims | file |",
        "|-------|--------|----|---------|---------|----------|------|------|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['model']}` | {r['prompt']} | "
            f"{'✓' if r['ok'] else '✗'} | "
            f"{r.get('seconds', '-')} | "
            f"{r.get('cost', '-')} | "
            f"{r.get('size_kb', '-')} | "
            f"{r.get('dims', '-')} | "
            f"{r.get('file', '-') if r['ok'] else (r.get('error', '-') or '')[:80]} |"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
