"""
transcribe.py

Performs speech-to-text transcription on single-speaker audio
using a pretrained Whisper model.

Key assumptions:
- Input audio contains ONE dominant speaker
- Audio has already been denoised and separated
- Inference-only (no training or fine-tuning)
"""

import json
import os
import sys
import whisper


# -------------------------------------------------
# MODEL LOADING (GLOBAL, ONCE)
# -------------------------------------------------
# Whisper models are large.
# Loading once is critical for performance.

_WHISPER_MODEL = None


def _get_whisper_model(model_size="base"):
    """
    Lazy-load and cache the Whisper model.
    """
    global _WHISPER_MODEL
    
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model(model_size)
    
    return _WHISPER_MODEL


def transcribe_audio(audio_path, model_size="base"):
    """
    Transcribe a single-speaker audio file.

    Parameters
    ----------
    audio_path : str
        Path to single-speaker .wav file
        
    model_size : str
        Whisper model size (tiny, base, small, medium, large)

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
    # 2. Load Whisper model
    # -------------------------------------------------
    model = _get_whisper_model(model_size)

    # -------------------------------------------------
    # 3. Run Whisper transcription
    # -------------------------------------------------
    # fp16=False is REQUIRED for CPU inference
    result = model.transcribe(
        audio_path,
        fp16=False
    )

    # -------------------------------------------------
    # 4. Extract relevant outputs
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


# -------------------------------------------------
# STANDALONE EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    """
    Run transcription as a standalone script.
    
    Usage:
        python transcribe.py <audio_path> [model_size] [output_dir]
    
    Example:
        python transcribe.py data/separated/audio_speaker_1.wav base data/transcripts
    """
    
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_path> [model_size] [output_dir]")
        print("Model sizes: tiny, base, small, medium, large")
        print("Example: python transcribe.py data/separated/audio_speaker_1.wav base data/transcripts")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "base"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/transcripts"
    
    print(f"Transcribing: {audio_file}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    
    try:
        result = transcribe_audio(audio_file, model)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
        json_path = os.path.join(output_dir, f"{base_name}_transcript.json")
        
        # Save plain text transcript
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result['text'])
        
        # Save full JSON with segments
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nSUCCESS!")
        print(f"Language: {result['language']}")
        print(f"Segments: {len(result['segments'])}")
        print(f"\nSaved to:")
        print(f"  Text: {txt_path}")
        print(f"  JSON: {json_path}")
        print(f"\nFull transcript:")
        print("-" * 40)
        print(result['text'])
        print("-" * 40)
        
        if result['segments']:
            print(f"\nSegment details:")
            for seg in result['segments'][:5]:  # Show first 5
                print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
            if len(result['segments']) > 5:
                print(f"  ... and {len(result['segments']) - 5} more segments")
                
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
