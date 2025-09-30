# lecproc

Tools for turning lecture materials into usable text:
- PDF → Markdown with Typst math extraction, validation, and auto-fixing
- Audio → JSONL transcripts via faster-whisper

## Why this is useful
- Typst-first Markdown: Parses Typst code blocks, validates syntax, and iteratively fixes errors with an LLM so fewer broken formulas make it to your notes.
- Bulk, resumable conversion: Skips PDFs that already have non-empty outputs, deletes empty outputs, and resumes from an intermediate `.phase1.md` when present.
- Concurrency and progress: Converts many PDFs in parallel with a clear progress bar.
- Context-aware: If `context.{md,txt,json}` exists beside your PDFs, it’s included to steer the conversion.
- Zero-setup runs with uv: `uv run` resolves and installs dependencies automatically on first use.

## Quick start
Prerequisites:
- Install uv: `pip install uv`
- Create a `.env` in the repo root with your LLM provider key(s), e.g.:
  - `OPENROUTER_API_KEY=...`

### Convert PDFs → Markdown
PowerShell:
```powershell
uv run --directory src -m pdf2md 'D:\OneDrive - National University of Singapore\Academic\Current-Modules\ST2132\Lecture'
```

Command Prompt (cmd.exe):
```bat
uv run --directory src -m pdf2md "D:\OneDrive - National University of Singapore\Academic\Current-Modules\ST2132\Lecture"
```
Notes:
- Output `.md` files are written next to each PDF.
- Add `--overwrite` to regenerate existing outputs.
- `uv run` will auto-install dependencies before executing the module.

### Transcribe audio → JSONL
```powershell
uv run --directory src -m audio2text "C:\path\to\lecture.mp3" "C:\path\to\out.jsonl" --language en
```
- Transcripts stream to JSONL (one segment per line).
- Models are downloaded to `hf_downloads/` (see `src/path_settings.py`).

## Configuration
Key options (env or defaults):
- LLM routing and retries via LiteLLM (see `src/pdf2md/settings.py`).
- Typst validation/fixing can be toggled; concurrency, timeouts, and retry counts are configurable.
- A `context.{md,txt,json}` file (if present) is loaded once and applied to all PDFs.

## Development
- Format: `just format`
- Checks: `just check`

No manual `pip install` needed unless you prefer a local venv; `uv run` handles resolution and caching automatically.