#!/usr/bin/env python3
"""CLI interface for querying the research proposal RAG system."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_engine import RAGEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the research proposal RAG system")
    parser.add_argument("question", help="Question to ask about the research proposal")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--json", action="store_true", help="Output JSON response")
    args = parser.parse_args()

    engine = RAGEngine()
    response = engine.ask(args.question, top_k=args.top_k)

    if args.json:
        print(
            json.dumps(
                {
                    "question": response.question,
                    "answer": response.answer,
                    "backend": response.backend,
                    "sources": response.sources,
                },
                indent=2,
            )
        )
        return

    print(f"\nQuestion: {response.question}")
    print(f"Backend: {response.backend}\n")
    print("Answer:")
    print(response.answer)
    print("\nSources:")
    for index, source in enumerate(response.sources, start=1):
        print(
            f"  [{index}] Page {source['page']} | {source['section']} "
            f"(score={source['score']})"
        )


if __name__ == "__main__":
    main()
