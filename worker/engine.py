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
        denoised_audio_path = denoise_audio(audio_path)
        result["denoised_audio"] = denoised_audio_path

        # -------------------------------------------------
        # 2. SPEAKER SEPARATION
        # -------------------------------------------------
        speaker_audio_paths = separate_speakers(denoised_audio_path)
        result["separated_audio"] = speaker_audio_paths

        # -------------------------------------------------
        # 3. TRANSCRIPTION (PER SPEAKER)
        # -------------------------------------------------
        transcription_results = []

        for speaker_path in speaker_audio_paths:
            transcript = transcribe_audio(speaker_path)
            transcription_results.append({
                "audio": speaker_path,
                "transcript": transcript
            })

        result["transcripts"] = transcription_results

        # -------------------------------------------------
        # 4. CONFIDENCE COMPUTATION
        # -------------------------------------------------
        confidence_report = compute_overall_confidence(
            denoised_audio_path=denoised_audio_path,
            speaker_audio_paths=speaker_audio_paths,
            transcription_results=transcription_results
        )

        result["confidence"] = confidence_report

        # -------------------------------------------------
        # 5. AUDIT ARTIFACTS
        # -------------------------------------------------
        audit_report = generate_audit_artifacts(
            raw_audio_path=audio_path,
            denoised_audio_path=denoised_audio_path,
            transcripts=transcription_results
        )

        result["audit"] = audit_report

        # -------------------------------------------------
        # PIPELINE COMPLETED SUCCESSFULLY
        # -------------------------------------------------
        result["status"] = "
