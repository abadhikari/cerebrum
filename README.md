# cerebrum

An LLM-connected semantic map that organizes and connects your thoughts and concepts.
Capture ideas, embed them, search them, and let a language model reason over your past thinking.
A memory system.

---

## Requirements

- Python **3.10+**
- macOS/Linux recommended  
  (Windows may require extra setup for audio + FAISS)
- A working C toolchain (FAISS may compile on some platforms)

---

## Installation

Use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -e .
```

This installs the cerebrum entrypoints.

---

## Usage

Run the CLI:

```bash
cerebrum-cli
```

This opens the interactive menu for adding thoughts, querying, and asking Cerebrum.

---

## Linting & Formatting (For Development Only)

Install the dev dependencies:

```
pip install -e ".[dev]"
```

Then use Ruff to lint and auto-fix, and Black to format:

```bash
ruff check . --fix && black .
```
