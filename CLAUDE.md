# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Modern async audio transcription tool using OpenAI's Whisper model with rich console output and parallel processing. Built with Python 3.12+ and modern tooling including uv, ruff, and pre-commit hooks.

## Architecture

- `transcribe.py`: Async main script that processes multiple audio formats concurrently
- `input-audio/`: Directory containing audio files (supports .mp3, .m4a, .wav, .flac, .aac)
- `pyproject.toml`: Project configuration with dependencies and ruff settings
- `.pre-commit-config.yaml`: Pre-commit hooks for code quality
- Output: `.txt` files saved alongside each audio file

## Setup Commands

### Initial Setup
```bash
# Create virtual environment
uv venv --python 3.12

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Install dependencies
uv sync

# Install pre-commit hooks
pre-commit install
```

### Development Commands

```bash
# Run transcription
python transcribe.py

# Code quality checks
ruff check .
ruff format .

# Run pre-commit hooks manually
pre-commit run --all-files
```

## Key Features

- Async/concurrent processing of multiple audio files
- Rich console output with progress tracking and logging
- Saves transcriptions as `.txt` files in same directory as audio
- Comprehensive error handling and logging
- Pre-commit hooks with ruff for code quality