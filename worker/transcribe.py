"""
transcribe.py

Performs speech-to-text transcription on single-speaker audio
using a pretrained Whisper model.

Key assumptions:
- Input audio contains ONE dominant speaker
- Audio has already been denoised and separated
- Inference-only (no training or fine-tuning)
"""

import os
import whisper


# -------------------------------------------------
# MODEL LOADING (GLOBAL, ONCE)
# -------------------------------------------------
# Whisper models are large.
# Loading once is critical for performance.

WHISPER_MODEL = whisper.load_model("base")  # CPU-friendly


def transcribe_audio(audio_path):
    """
    Transcribe a single-speaker audio file.

    Parameters
    ----------
    audio_path : str
        Path to single-speaker .wav file

    Returns
    -------
    result : dict
        {
            "text": full transcript,
            "segments": [
                {
                    "start": float,
                    "end": float,
                    "text": str,
                    "avg_logprob": float
                }
            ],
            "language": detected language
        }
    """

    # -------------------------------------------------
    # 1. Validation
    # -------------------------------------------------
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # -------------------------------------------------
    # 2. Run Whisper transcription
    # -------------------------------------------------
    # fp16=False is REQUIRED for CPU inference
    result = WHISPER_MODEL.transcribe(
        audio_path,
        fp16=False
    )

    # -------------------------------------------------
    # 3. Extract relevant outputs
    # -------------------------------------------------
    transcript_data = {
        "text": result.get("text", "").strip(),
        "segments": [],
        "language": result.get("language", "unknown")
    }

    for segment in result.get("segments", []):
        transcript_data["segments"].append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip(),
            "avg_logprob": segment["avg_logprob"]
        })

    return transcript_data
