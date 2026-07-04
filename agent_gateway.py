"""
agent_gateway.py
----------------
LangChain tool wrapper that exposes the Auvia audio enhancement pipeline
(noise reduction + spectral shaping) as an agent-callable tool.

The tool follows the "airlock" pattern:
  - The FastAPI endpoint decodes base64 → temp file on disk
  - The agent receives the file path and passes it to this tool
  - This tool reads the file, processes it, and writes the output
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import tool

from dsp_pipeline import apply_timbre_enhancement

logger = logging.getLogger("auvia_agent_gateway")


@tool
def enhance_audio(file_path: str, output_path: str = "output/enhanced.wav") -> str:
    """
    Apply Auvia's two-stage vocal enhancement to an audio file on disk.

    Stage 1 — Noise Reduction: spectral gating via noisereduce removes room
    noise, mic hiss, AC hum, and breath noise.

    Stage 2 — Spectral Shaping: single STFT pass that warms low-mids
    (150–500 Hz × 1.4), lifts presence (2–5 kHz × 1.15), and rolls off
    sub-bass rumble (< 80 Hz × 0.3).

    The result is peak-normalised to −1 dBFS and written as a 16-bit PCM WAV.

    Args:
        file_path: Path to the source audio file on disk (WAV, MP3, FLAC,
                   OGG, or M4A).
        output_path: Filesystem path where the enhanced WAV file should be
                     saved (default: "output/enhanced.wav").

    Returns:
        A human-readable summary string that includes the output path and
        the size of the processed file in kilobytes.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: If the decoded audio produces silence after processing.
        Exception: Propagates any librosa / soundfile decode errors so the
                   agent can surface them to the caller.
    """
    logger.info("enhance_audio tool invoked | file_path=%s | output_path=%s",
                file_path, output_path)

    # Verify the source file exists
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"Source audio file not found: {file_path}")

    # Read the file from disk
    audio_bytes: bytes = src.read_bytes()
    logger.info("Read input file: %d bytes", len(audio_bytes))

    # Delegate to the core DSP pipeline
    processed_bytes: bytes = apply_timbre_enhancement(audio_bytes)

    # Persist the result
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(processed_bytes)

    size_kb = len(processed_bytes) / 1024
    summary = (
        f"Audio enhancement complete. "
        f"Output written to '{dest}' ({size_kb:.1f} KB, 16-bit PCM WAV)."
    )
    logger.info(summary)
    return summary