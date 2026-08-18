"""
Start the web control panel.

    python scripts/start_web.py

Opens the studio UI at http://127.0.0.1:8000 — the FastAPI bridge serves the
frontend (web/) and exposes /api/* endpoints (see web/API.md).

Optional env:
    OPC_OUTPUT_DIR=<abs path>   override the output directory
    OPC_PORT=8080               override the port (default 8000)
"""
import os
import sys
import webbrowser
from pathlib import Path

# make sure we run from the project root so relative paths (input/, output/) work
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("OPC_PORT", "8000"))
    print("=" * 60)
    print("  OPC AI Video Studio — Web Control Panel")
    print(f"  Open:  http://127.0.0.1:{port}")
    print("  API contract: web/API.md")
    print("=" * 60)
    webbrowser.open(f"http://127.0.0.1:{port}")
    uvicorn.run("api.server:app", host="127.0.0.1", port=port, log_level="info")
