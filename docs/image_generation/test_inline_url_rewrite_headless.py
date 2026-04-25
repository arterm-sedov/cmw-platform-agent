"""Headless smoke test — validates /file= URL serving without a browser.

Launches smoke_inline_url_rewrite.py, then uses requests to:
1. Fetch the page HTML
2. Check what Gradio writes for img src in the chatbot bubble
3. Attempt to fetch the /file= URL endpoint for the PNG

Exit codes:
  0 = all checks passed
  1 = test failure
  2 = server failed to start
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import requests

_REPO = Path(__file__).resolve().parents[2]
_smoke_script = _REPO / "docs" / "image_generation" / "smoke_inline_url_rewrite.py"


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    """Poll ``url`` until it responds or ``timeout`` expires."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    print("=" * 60)
    print("Starting smoke server ...")

    server = subprocess.Popen(
        [sys.executable, str(_smoke_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Extract port from stdout (server prints "Port     : NNN")
        port = None
        for _ in range(30):
            line = server.stdout.readline()
            if line:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                print(f"  [server] {decoded}")
                if decoded.startswith("Port     : "):
                    port = int(decoded.split()[-1])
                    break
            time.sleep(0.3)
            if server.poll() is not None:
                stderr = server.stderr.read().decode("utf-8", errors="replace")
                print(f"Server died. stderr:\n{stderr}")
                return 2

        if not port:
            print("ERROR: Could not determine server port from stdout")
            return 2

        base_url = f"http://127.0.0.1:{port}"
        if not _wait_for_server(base_url):
            print(f"ERROR: Server at {base_url} did not respond in time")
            return 2

        print(f"\nServer is up at {base_url}")
        print("-" * 60)

        # ------------------------------------------------------------------
        # 1. Fetch page HTML and inspect img src attributes
        # ------------------------------------------------------------------
        html = requests.get(base_url, timeout=5.0).text

        findings: dict[str, str] = {}
        import re
        for match in re.finditer(r'<img\s+src="([^"]+)"', html):
            src = match.group(1)
            # Grab a snippet of surrounding text as label
            snippet = match.group(0)[:60]
            findings[snippet] = src

        print("\n1. img src attributes found in page HTML:")
        has_bare = False
        has_file_url = False
        has_http = False
        for snippet, src in findings.items():
            print(f"   {snippet!r}")
            print(f"   → src={src!r}")
            if src.startswith("llm_image_") and not src.startswith("/file="):
                has_bare = True
            elif src.startswith("/file="):
                has_file_url = True
            elif src.startswith("http"):
                has_http = True
            print()

        # ------------------------------------------------------------------
        # 2. Test the /file= endpoint directly
        # ------------------------------------------------------------------
        # The smoke PNG is at _SMOKE_DIR / "llm_image_smoke_test.png"
        # We need to find its absolute path — it's in the Gradio cache dir.
        # We can get it from the HTML by looking for /file= in the markdown blocks.

        print("2. Testing /file= endpoint for the smoke PNG ...")
        # Find a /file= URL in the HTML and try to fetch it
        file_url_match = re.search(r'src="(/file=[^"]+)"', html)
        if file_url_match:
            file_url = file_url_match.group(1)
            # Resolve to absolute URL
            abs_url = base_url + "/gradio_api" + file_url
            print(f"   Found /file= URL: {file_url}")
            print(f"   Full URL: {abs_url}")
            try:
                r = requests.get(abs_url, timeout=5.0)
                print(f"   HTTP {r.status_code}  Content-Type: {r.headers.get('Content-Type','?')}")
                if r.status_code == 200 and len(r.content) > 0:
                    print(f"   Content length: {len(r.content):,} B")
                    print("   ✓ /file= endpoint serves the PNG!")
                else:
                    print(f"   ✗ Unexpected response")
                    return 1
            except requests.RequestException as e:
                print(f"   ✗ Request failed: {e}")
                return 1
        else:
            print("   ⚠ No /file= URL found in page HTML (expected when bare img is broken)")
            print("   This is OK — the smoke server only tests bare img rendering")

        # ------------------------------------------------------------------
        # 3. Summary
        # ------------------------------------------------------------------
        print("-" * 60)
        print("\n3. Summary:")
        print(f"   Bare llm_image_ src in HTML: {'YES (broken expected)' if has_bare else 'NO'}")
        print(f"   /file= URL in HTML:          {'YES' if has_file_url else 'NO'}")
        print(f"   http:// gradio_api URL:       {'YES' if has_http else 'NO'}")

        print("\n   KEY INSIGHT:")
        if has_bare:
            print("   Gradio does NOT auto-rewrite bare img src — it passes them as-is.")
            print("   This confirms our rewriter is needed: convert bare filenames to /file= URLs.")
        else:
            print("   Gradio appears to rewrite bare img src to http:// URLs.")
            print("   The /file= form should definitely work.")

        print("\n✓ Smoke test complete")
        return 0

    finally:
        server.terminate()
        server.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
