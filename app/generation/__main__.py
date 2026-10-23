"""Run one citation-validated RAG question from the command line."""

import argparse
import sys

from app.generation.service import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(answer_question(args.question).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
