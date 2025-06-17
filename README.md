# Audio Transcription Tool

A modern async audio transcription tool using OpenAI's Whisper model with rich console output and parallel processing.

## Features

- Async/concurrent transcription of multiple audio files
- Rich console output with progress tracking
- Supports multiple audio formats: MP3, M4A, WAV, FLAC, AAC
- Saves transcriptions as `.txt` files alongside original audio files
- Structured logging with rich formatting

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Create virtual environment:
```bash
uv venv --python 3.12
```

2. Activate virtual environment:
```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

3. Install dependencies:
```bash
uv sync
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Usage

Place your audio files in the `input-audio/` directory (can be organized in subdirectories) and run:

```bash
python transcribe.py
```

Transcriptions will be saved as `.txt` files in the same directory as each audio file.

## Development

### Code Quality

This project uses `ruff` for linting and formatting. Pre-commit hooks will automatically run these checks.

To manually run checks:
```bash
ruff check .
ruff format .
```