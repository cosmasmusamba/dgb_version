"""
main_api_server.py
====================
Production entry point for the DGB API server.

Usage
-----
  python main_api_server.py
  python main_api_server.py --host 0.0.0.0 --port 8080 --reload
  uvicorn main_api_server:app --host 0.0.0.0 --port 8000

Environment
-----------
  DGB_SECRET_KEY   JWT signing secret (required in production)
  BRAVE_API_KEY    Web search grounding (optional)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from configs.loader import get_config
from api_manager.app import create_app

logger = logging.getLogger(__name__)

# ASGI app — importable by uvicorn directly
app = create_app()


def main() -> None:
    import uvicorn
    parser = argparse.ArgumentParser(description="DGB API server")
    parser.add_argument("--host",   default=None)
    parser.add_argument("--port",   type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    cfg  = get_config()
    host = args.host  or cfg.api.host
    port = args.port  or cfg.api.port

    print(f"\n  DGB AI Platform v{cfg.project.version}")
    print(f"  API:  http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}{cfg.api.docs_url}")
    print(f"  UI:   http://{host}:{port}/ui\n")

    uvicorn.run(
        "main_api_server:app",
        host=host,
        port=port,
        reload=args.reload or cfg.api.reload,
        workers=args.workers,
        log_level=cfg.api.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
