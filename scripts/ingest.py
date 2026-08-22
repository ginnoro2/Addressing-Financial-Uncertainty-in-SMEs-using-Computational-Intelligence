#!/usr/bin/env python3
"""Ingest the research proposal PDF into the vector store."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_engine import RAGEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest research proposal PDF")
    parser.add_argument("--force", action="store_true", help="Rebuild the index")
    args = parser.parse_args()

    engine = RAGEngine()
    result = engine.ingest(force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
