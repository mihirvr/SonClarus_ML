"""
engine.py

Final ML pipeline orchestrator for SonClarus.

Responsibilities:
- Enforce correct ML stage ordering
- Integrate confidence estimation
- Generate audit artifacts
- Surface errors explicitly

This file contains NO model logic.
"""

import os
import sys

from denoise import denoise_audio
from separate import separate_speakers
from transcribe import transcribe_audio
from confidence import compute_overall_confidence
from audit import generate_audit_artifacts


def process_audio_job(audio_path):
    """
    Run the complete SonClarus ML pipeline.

    Parameters
    ----------
    audio_path : str
        Path to raw input audio (.wav)

    Returns
    -------
    result : dict
        Comprehensive ML output with confidence and audit data
    """

    result = {
        "status": "FAILED",
        "input_audio": audio_path,
        "denoised_audio": None,
        "separated_audio": [],
        "transcripts": [],
        "confidence": {},
        "audit": {},
        "errors": []
    }

    try:
        # -------------------------------------------------
        # 1. DENOISING
        # -------------------------------------------------
        print(f"[1/5] Denoising audio: {audio_path}")
        denoised_audio_path = denoise_audio(audio_path)
        result["denoised_audio"] = denoised_audio_path
        print(f"      Denoised: {denoised_audio_path}")

        # -------------------------------------------------
        # 2. SPEAKER SEPARATION
        # -------------------------------------------------
        print(f"[2/5] Separating speakers...")
        speaker_audio_paths = separate_speakers(denoised_audio_path)
        result["separated_audio"] = speaker_audio_paths
        print(f"      Separated into {len(speaker_audio_paths)} tracks")

        # -------------------------------------------------
        # 3. TRANSCRIPTION (PER SPEAKER)
        # -------------------------------------------------
        print(f"[3/5] Transcribing speakers...")
        transcription_results = []

        for idx, speaker_path in enumerate(speaker_audio_paths):
            print(f"      Transcribing speaker {idx + 1}...")
            transcript = transcribe_audio(speaker_path)
            transcription_results.append({
                "audio": speaker_path,
                "transcript": transcript
            })

        result["transcripts"] = transcription_results

        # -------------------------------------------------
        # 4. CONFIDENCE COMPUTATION
        # -------------------------------------------------
        print(f"[4/5] Computing confidence scores...")
        confidence_report = compute_overall_confidence(
            denoised_audio_path=denoised_audio_path,
            speaker_audio_paths=speaker_audio_paths,
            transcription_results=transcription_results
        )

        result["confidence"] = confidence_report
        print(f"      Overall confidence: {confidence_report.get('overall_confidence', 'N/A')}")

        # -------------------------------------------------
        # 5. AUDIT ARTIFACTS
        # -------------------------------------------------
        print(f"[5/5] Generating audit artifacts...")
        audit_report = generate_audit_artifacts(
            raw_audio_path=audio_path,
            denoised_audio_path=denoised_audio_path,
            transcripts=transcription_results
        )

        result["audit"] = audit_report

        # -------------------------------------------------
        # PIPELINE COMPLETED SUCCESSFULLY
        # -------------------------------------------------
        result["status"] = "SUCCESS"
        print(f"\nPipeline completed successfully!")

    except Exception as e:
        result["status"] = "FAILED"
        result["errors"].append(str(e))
        print(f"\nPipeline FAILED: {e}")

    return result


# -------------------------------------------------
# STANDALONE EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    """
    Run the complete audio processing pipeline as a standalone script.
    
    Usage:
        python engine.py <input_audio_path>
    
    Example:
        python engine.py data/raw/meeting_audio.wav
    """
    
    if len(sys.argv) < 2:
        print("Usage: python engine.py <input_audio_path>")
        print("Example: python engine.py data/raw/meeting_audio.wav")
        sys.exit(1)
    
    input_audio = sys.argv[1]
    
    if not os.path.exists(input_audio):
        print(f"ERROR: Audio file not found: {input_audio}")
        sys.exit(1)
    
    print(f"=" * 60)
    print(f"SonClarus Audio Processing Pipeline")
    print(f"=" * 60)
    print(f"Input: {input_audio}")
    print(f"=" * 60)
    print()
    
    result = process_audio_job(input_audio)
    
    print()
    print(f"=" * 60)
    print(f"RESULTS")
    print(f"=" * 60)
    print(f"Status: {result['status']}")
    print(f"Denoised audio: {result['denoised_audio']}")
    print(f"Separated tracks: {len(result['separated_audio'])}")
    print(f"Transcripts: {len(result['transcripts'])}")
    
    if result['transcripts']:
        print()
        print("TRANSCRIPTIONS:")
        for idx, t in enumerate(result['transcripts'], 1):
            text = t['transcript'].get('text', '')[:200]
            print(f"  Speaker {idx}: {text}...")
    
    if result['confidence']:
        print()
        print(f"CONFIDENCE SCORES:")
        for key, value in result['confidence'].items():
            print(f"  {key}: {value}")
    
    if result['errors']:
        print()
        print(f"ERRORS:")
        for error in result['errors']:
            print(f"  - {error}")
