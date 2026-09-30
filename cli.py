import argparse
import logging
import os
import sys

from rag import build_index_from_path, index_pages, retrieve_matches

logging.getLogger("pypdf").setLevel(logging.CRITICAL)

COMMANDS = """
Commands:
  /pages     list the page numbers that were indexed
  /chunks    show how many chunks the PDF was split into
  /help      show this message
  /exit      leave the session
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Ask questions about a local PDF and read back the "
                    "matching passages.",
    )
    parser.add_argument("pdf", help="path to the PDF file")
    parser.add_argument(
        "-k",
        "--chunks",
        type=int,
        default=4,
        help="how many passages to show per question (default: 4)",
    )
    parser.add_argument(
        "-q",
        "--question",
        help="ask one question and exit, instead of starting a session",
    )
    return parser


def render_matches(matches) -> str:
    blocks = []
    for rank, (doc, distance) in enumerate(matches, start=1):
        page = doc.metadata.get("page", "?")
        blocks.append(
            f"[{rank}] page {page}  (distance {distance:.3f}, lower is closer)\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def answer(vectorstore, question: str, k: int) -> str:
    matches = retrieve_matches(vectorstore, question, k=k)
    return render_matches(matches)


def build_index(pdf_path: str):
    vectorstore, chunks = build_index_from_path(pdf_path)
    return vectorstore, len(chunks), index_pages(chunks)


def main() -> int:
    args = build_parser().parse_args()

    if args.chunks < 1:
        print("error: --chunks must be at least 1", file=sys.stderr)
        return 2

    if not os.path.isfile(args.pdf):
        print(f"error: no such file: {args.pdf}", file=sys.stderr)
        return 1

    print(f"Indexing {args.pdf} ...")
    try:
        vectorstore, chunk_count, pages = build_index(args.pdf)
    except FileNotFoundError:
        print(f"error: no such file: {args.pdf}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: could not read {args.pdf}: {exc}", file=sys.stderr)
        return 1

    print(
        f"Ready. {chunk_count} chunks from {len(pages)} pages "
        f"({pages[0]}-{pages[-1]})."
    )
    print("This version retrieves passages; it does not generate answers.\n")

    if args.question:
        print(answer(vectorstore, args.question, args.chunks))
        return 0

    print(f"Ask a question about the PDF, or type /help.\n{COMMANDS}\n")

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            continue

        if question in {"/exit", "/quit"}:
            print("bye")
            return 0
        if question == "/help":
            print(f"\n{COMMANDS}\n")
            continue
        if question == "/pages":
            print(f"\npages: {pages}\n")
            continue
        if question == "/chunks":
            print(f"\nchunks: {chunk_count}\n")
            continue
        if question.startswith("/"):
            print(f"unknown command: {question} (try /help)\n")
            continue

        print(f"\n{answer(vectorstore, question, args.chunks)}\n")


if __name__ == "__main__":
    raise SystemExit(main())
