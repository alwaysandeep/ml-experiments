"""
ReadWithAI - Interactive paper reading tool using local Ollama models.
Based on: https://aayushmnit.com/posts/2026-01-31_ReadWithAI/ReadWithAI.html

Usage:
    python read_paper.py <path_to_pdf>

Requirements:
    - Ollama running locally (ollama serve)
    - pip install pymupdf4llm lisette

Default local model: qwen3-vl:8b (as recommended in the original post)
Context window: 64000 tokens (as recommended in the original post)
"""

import sys
import os
import signal
import time
import threading
import argparse
import urllib.request
import pymupdf4llm
from lisette import Chat, contents


def _sigint_handler(sig, frame):
    print("\nExiting.")
    os._exit(0)

signal.signal(signal.SIGINT, _sigint_handler)


def check_ollama():
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
    except Exception:
        print("Error: Ollama is not running. Start it first:\n")
        print('  PowerShell: $env:OLLAMA_CONTEXT_LENGTH=64000; ollama serve')
        print('  Bash:       OLLAMA_CONTEXT_LENGTH=64000 ollama serve\n')
        sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "ollama/qwen3-vl:8b"


def estimate_num_ctx(paper_text: str) -> int:
    """Estimate tokens needed: paper chars / 4 + system prompt overhead + generation headroom."""
    paper_tokens = len(paper_text) // 4
    overhead = 1024   # system prompt boilerplate
    headroom = 2048   # generation output
    needed = paper_tokens + overhead + headroom
    # Round up to next power of 2, min 4096
    ctx = 4096
    while ctx < needed:
        ctx *= 2
    print(f"[ctx] ~{paper_tokens:,} paper tokens → num_ctx={ctx:,}")
    return ctx

SYSTEM_PROMPT_TEMPLATE = """You are an expert tutor helping me understand a research paper deeply.

Here is the full paper text:
<paper>
{paper_text}
</paper>

Your teaching philosophy:
- Build intuition FIRST, math second
- Use concrete examples and analogies
- Connect new ideas to things I already know
- Use code/numpy examples when helpful for algorithms

CRITICAL RULES:
- NEVER explain everything at once
- Take ONE small concept, explain it clearly, then STOP and wait
- Keep responses SHORT (2-4 paragraphs max)
- Always end with a question to check understanding OR ask what to cover next
- If I seem confused, back up and try a different angle
- Format math with backticks, e.g. `loss = sum(w_i * log(p_i))`

Start by asking about my background and what I want to get out of this paper."""


def convert_pdf(pdf_path: str) -> str:
    """Convert PDF to markdown, caching result as .md next to the PDF."""
    md_path = pdf_path.rsplit(".", 1)[0] + ".md"
    if os.path.exists(md_path):
        print(f"[cache] Using existing markdown: {md_path}")
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    print(f"[convert] Converting PDF to markdown...", end=" ", flush=True)
    t0 = time.time()
    text = pymupdf4llm.to_markdown(pdf_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"done ({time.time()-t0:.1f}s) — saved: {md_path}")
    return text


def make_chat(paper_text: str, model: str) -> Chat:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(paper_text=paper_text)
    return Chat(model=model, sp=system_prompt)


def _spinner(stop_event: threading.Event, t0: float):
    """Print elapsed time while waiting for first token."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        elapsed = time.time() - t0
        print(f"\r[loading context... {frames[i % len(frames)]} {elapsed:.0f}s]", end="", flush=True)
        i += 1
        time.sleep(0.1)
    # Clear the spinner line
    print("\r" + " " * 40 + "\r", end="", flush=True)


def call(chat: Chat, msg: str, num_ctx: int):
    """Stream response tokens to stdout, return full text."""
    t0 = time.time()
    first_token_time = None

    # Start spinner to show context-loading progress
    stop_spinner = threading.Event()
    spinner = threading.Thread(target=_spinner, args=(stop_spinner, t0), daemon=True)
    spinner.start()

    stream = chat(msg, stream=True, num_ctx=num_ctx)
    full = ""
    try:
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
                if delta:
                    if first_token_time is None:
                        first_token_time = time.time()
                        stop_spinner.set()
                        spinner.join()
                        print("Assistant: ", end="", flush=True)
                    print(delta, end="", flush=True)
                    full += delta
            except (AttributeError, IndexError):
                pass
    except KeyboardInterrupt:
        stop_spinner.set()
        print("\n[interrupted]")
        raise
    elapsed = time.time() - t0
    ttft = f"ttft {first_token_time - t0:.1f}s, " if first_token_time else ""
    words = len(full.split())
    print(f"\n[{ttft}total {elapsed:.1f}s, ~{words} words]\n")
    return full


def repl(chat: Chat, num_ctx: int):
    """Simple REPL with checkpoint support."""
    print("\n" + "="*60)
    print("Commands: /exit  /checkpoint  /rewind  /history")
    print("="*60 + "\n")

    checkpoint = 0

    # Kick off the conversation
    greeting = "Hello, let's start."
    try:
        call(chat, greeting, num_ctx)
    except KeyboardInterrupt:
        print("Exiting.")
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("Goodbye!")
            break
        elif user_input == "/checkpoint":
            checkpoint = len(chat.hist)
            print(f"[checkpoint saved at message {checkpoint}]")
            continue
        elif user_input == "/rewind":
            if checkpoint == 0:
                print("[no checkpoint set — use /checkpoint first]")
            else:
                chat.hist = chat.hist[:checkpoint]
                print(f"[rewound to message {checkpoint}]")
            continue
        elif user_input == "/history":
            for i, msg in enumerate(chat.hist):
                role = msg.get("role", "?")
                text = str(msg.get("content", ""))[:120]
                print(f"  [{i}] {role}: {text}")
            print()
            continue

        try:
            call(chat, user_input, num_ctx)
        except KeyboardInterrupt:
            print("Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Read a research paper interactively with AI.")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: file not found: {args.pdf}")
        sys.exit(1)

    check_ollama()
    print(f"[model] {args.model}")
    paper_text = convert_pdf(args.pdf)
    print(f"[paper] {len(paper_text):,} characters")

    num_ctx = estimate_num_ctx(paper_text)
    chat = make_chat(paper_text, args.model)
    repl(chat, num_ctx)


if __name__ == "__main__":
    main()
