import asyncio
import logging
import os
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


async def transcribe_file(model, audio_path: Path, progress, task_id):
    """Transcribe a single audio file asynchronously"""
    try:
        logger.info(f"Starting transcription for {audio_path.name}")
        progress.update(task_id, description=f"Transcribing {audio_path.name}...")
        
        # Run transcription in thread pool since whisper is CPU-bound
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, model.transcribe, str(audio_path))
        transcription = result["text"]
        
        # Save transcription to text file in same directory as audio
        output_path = audio_path.with_suffix(".txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        
        progress.update(task_id, description=f"✓ Completed {audio_path.name}")
        logger.info(f"Completed transcription for {audio_path.name} -> {output_path}")
        return True
        
    except Exception as e:
        progress.update(task_id, description=f"✗ Failed {audio_path.name}")
        logger.error(f"Failed to transcribe {audio_path.name}: {e}")
        return False


async def transcribe_directory(audio_directory: Path):
    """Transcribe all supported audio files in directory structure"""
    logger.info(f"Loading Whisper model...")
    model = whisper.load_model("base")
    
    # Find all supported audio files
    supported_extensions = {".mp3", ".m4a", ".wav", ".flac", ".aac"}
    audio_files = []
    
    for root, dirs, files in os.walk(audio_directory):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in supported_extensions:
                audio_files.append(file_path)
    
    if not audio_files:
        logger.warning(f"No supported audio files found in {audio_directory}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files to transcribe")
    
    # Create progress tracker
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        # Create tasks for each file
        tasks = []
        for audio_file in audio_files:
            task_id = progress.add_task(f"Queued {audio_file.name}", total=None)
            task = transcribe_file(model, audio_file, progress, task_id)
            tasks.append(task)
        
        # Run transcriptions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes
        successful = sum(1 for result in results if result is True)
        logger.info(f"Transcription complete: {successful}/{len(audio_files)} files processed successfully")


if __name__ == "__main__":
    if not AUDIO_INPUT_PATH.exists():
        logger.error(f"Audio input directory '{AUDIO_INPUT_PATH}' not found")
        exit(1)
    
    asyncio.run(transcribe_directory(AUDIO_INPUT_PATH))