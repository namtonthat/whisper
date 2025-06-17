import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import whisper
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
logger = logging.getLogger(__name__)

AUDIO_INPUT_PATH = Path("input-audio")
TRANSCRIPTION_LOG_PATH = Path("transcription_log.json")


def load_transcription_log():
    """Load the transcription log from file"""
    if TRANSCRIPTION_LOG_PATH.exists():
        try:
            with open(TRANSCRIPTION_LOG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load transcription log: {e}. Starting fresh.")
    return {}


def save_transcription_log(log_data):
    """Save the transcription log to file"""
    try:
        with open(TRANSCRIPTION_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.error(f"Failed to save transcription log: {e}")


def is_file_already_transcribed(file_path: Path, log_data: dict) -> bool:
    """Check if a file has already been transcribed based on path and modification time"""
    output_path = file_path.with_suffix(".txt")
    file_key = str(file_path)

    # First check if output file exists
    if not output_path.exists():
        logger.info(f"Output file for {file_path.name} not found")
        return False

    # If no log data exists but output file does, check modification times
    if file_key not in log_data:
        # If output is newer than input, consider it transcribed
        if output_path.stat().st_mtime >= file_path.stat().st_mtime:
            logger.info(f"Found existing transcription for {file_path.name} (no log entry)")
            return True
        else:
            logger.info(f"Output file for {file_path.name} is older than audio file")
            return False

    # Check if the file has been modified since last transcription
    current_mtime = file_path.stat().st_mtime
    logged_mtime = log_data[file_key].get("mtime", 0)

    if current_mtime > logged_mtime:
        logger.info(f"File {file_path.name} has been modified since last transcription")
        return False

    return True


def split_into_sentences(text):
    """Split text into sentences, preserving original punctuation and spacing"""
    # Split on sentence endings but keep the punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def transcribe_file(model, audio_path: Path, log_data: dict):
    """Transcribe a single audio file with its own progress timer"""
    try:
        logger.info(f"Starting transcription for {audio_path.name}")

        # Create a new progress context for this file with its own timer
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as file_progress:
            task_id = file_progress.add_task(f"Transcribing {audio_path.name}...", total=None)

            # Transcribe with settings to preserve filler words and pauses
            result = model.transcribe(
                str(audio_path),
                verbose=False,
                fp16=False,  # Use FP32 on CPU
                word_timestamps=True,  # Get word-level timestamps
                prepend_punctuations="\"'¿([{-",
                append_punctuations="\"'.,!?:;)]}",
                no_speech_threshold=0.6,  # Lower threshold to catch more speech
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True,  # Better context understanding
                temperature=0.0,  # More deterministic output
                beam_size=5,  # Better accuracy
                patience=1.0,  # Allow more processing time
                length_penalty=1.0,
                suppress_blank=False,  # Don't suppress blanks/pauses
                suppress_tokens=[],  # Don't suppress any tokens
            )

            # Get the raw transcription
            raw_text = result["text"].strip()

            # Split into sentences and format with newlines
            sentences = split_into_sentences(raw_text)
            formatted_transcription = "\n".join(sentences)

            # Save transcription to text file in same directory as audio
            output_path = audio_path.with_suffix(".txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_transcription)

            # Update log data with successful transcription
            log_data[str(audio_path)] = {
                "mtime": audio_path.stat().st_mtime,
                "transcribed_at": datetime.now().isoformat(),
                "output_path": str(output_path),
                "detected_language": result.get("language", "unknown"),
            }

            file_progress.update(task_id, description=f"✓ Completed {audio_path.name}")

        logger.info(f"Completed transcription for {audio_path.name} -> {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to transcribe {audio_path.name}: {e}")
        return False


def transcribe_directory(audio_directory: Path):
    """Transcribe all supported audio files in directory structure sequentially"""
    # Load transcription log
    log_data = load_transcription_log()
    logger.info("Loading Whisper model...")
    model = whisper.load_model("base")

    # Find all supported audio files
    supported_extensions = {".mp3", ".m4a", ".wav", ".flac", ".aac"}
    audio_files = []
    skipped_files = []

    for root, _dirs, files in os.walk(audio_directory):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in supported_extensions:
                if is_file_already_transcribed(file_path, log_data):
                    skipped_files.append(file_path)
                else:
                    audio_files.append(file_path)

    if skipped_files:
        logger.info(f"Skipping {len(skipped_files)} already transcribed files")
        for file in skipped_files:
            logger.debug(f"  - {file.name}")

    if not audio_files:
        if skipped_files:
            logger.info("All audio files have already been transcribed")
        else:
            logger.warning(f"No supported audio files found in {audio_directory}")
        return

    logger.info(f"Found {len(audio_files)} audio files to transcribe")

    successful = 0

    # Process files sequentially
    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file.name}")
        result = transcribe_file(model, audio_file, log_data)
        if result:
            successful += 1
            # Save log after each successful transcription
            save_transcription_log(log_data)

    logger.info(f"Transcription complete: {successful}/{len(audio_files)} files processed successfully")

    if skipped_files:
        logger.info(f"Previously transcribed: {len(skipped_files)} files")


if __name__ == "__main__":
    if not AUDIO_INPUT_PATH.exists():
        logger.error(f"Audio input directory '{AUDIO_INPUT_PATH}' not found")
        exit(1)

    transcribe_directory(AUDIO_INPUT_PATH)
