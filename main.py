#!/usr/bin/env python3
"""Entry point for transcriber CLI."""

import sys
from pathlib import Path

# Add src to path so we can import transcriber
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcriber.cli import app

if __name__ == "__main__":
    app()
