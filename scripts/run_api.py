"""
CLI script to start the FastAPI anti-spoofing API server.

Usage:
    python scripts/run_api.py [--host HOST] [--port PORT]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config


def main():
    parser = argparse.ArgumentParser(
        description="Start the Audio Anti-Spoofing API server"
    )
    parser.add_argument("--host", type=str, default=config.API_HOST)
    parser.add_argument("--port", type=int, default=config.API_PORT)
    parser.add_argument("--reload", action="store_true", default=False,
                        help="Enable auto-reload for development")
    args = parser.parse_args()

    import uvicorn

    print("=" * 60)
    print("AUDIO ANTI-SPOOFING API SERVER")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Docs: http://localhost:{args.port}/docs")
    print(f"Health: http://localhost:{args.port}/api/v1/health")
    print("=" * 60)

    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
