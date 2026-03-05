# ReadWithAI — Interactive Paper Reader

Interactive learning from research papers via local LLM. Based on:
https://aayushmnit.com/posts/2026-01-31_ReadWithAI/ReadWithAI.html

## Setup

### 1. Install dependencies
```bash
pip install pymupdf4llm lisette
```

### 2. Pull the recommended model (~5 GB)
```bash
ollama pull qwen3-vl:8b
```

### 3. Start Ollama with expanded context (required — default is only 2048)
```bash
# Windows PowerShell
$env:OLLAMA_CONTEXT_LENGTH=64000; ollama serve

# Or set it permanently in Ollama's settings
```

## Usage
```bash
python read_paper.py path/to/paper.pdf
```

The PDF is converted to markdown on first run and cached as `.md` next to the PDF.

## REPL commands
| Command | Action |
|---------|--------|
| `/checkpoint` | Save current position (before exploring a tangent) |
| `/rewind` | Restore to last checkpoint |
| `/history` | Show conversation history |
| `/exit` | Quit |

## Switching models
```bash
# Use a different local model
python read_paper.py paper.pdf --model ollama/phi4:latest

# Use Claude (cloud)
ANTHROPIC_API_KEY=... python read_paper.py paper.pdf --model anthropic/claude-haiku-4-5-20251001
```

## Notes
- `qwen3-vl:8b` is a vision-language model; it handles text-heavy papers well
- Context of 64k tokens is important — most papers are 10k–30k chars as markdown
- The system prompt enforces "one concept at a time" teaching — don't disable it
