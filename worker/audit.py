"""
audit.py

Generates transparency and audit artifacts for SonClarus.

Artifacts include:
- Spectrogram comparisons (before vs after denoising)
- Confidence-aware transcript text
- Processing logs

Purpose:
Enable explainability, review, and trust.
"""

import os
import sys
import datetime

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt


# -------------------------------------------------
# SPECTROGRAM GENERATION
# -------------------------------------------------
def generate_spectrogram(audio_path, output_path, title):
    """
    Generate and save a spectrogram image for an audio file.
    """

    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    spectrogram = torchaudio.transforms.Spectrogram()(waveform)

    spectrogram_db = 10 * torch.log10(spectrogram + 1e-10)

    plt.figure(figsize=(10, 4))
    plt.imshow(
        spectrogram_db.squeeze().numpy(),
        origin="lower",
        aspect="auto",
        cmap="magma"
    )
    plt.colorbar(label="Intensity (dB)")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


# -------------------------------------------------
# CONFIDENCE-AWARE TRANSCRIPT
# -------------------------------------------------
def generate_confidence_transcript(transcripts, output_path, threshold=0.5):
    """
    Create a human-readable transcript with confidence markers.

    Low-confidence segments are clearly flagged.
    """

    lines = []
    lines.append("=" * 60)
    lines.append("SONCLARUS TRANSCRIPT WITH CONFIDENCE MARKERS")
    lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
    lines.append("=" * 60)

    for idx, item in enumerate(transcripts, start=1):
        speaker_header = f"\n--- Speaker {idx} ---"
        lines.append(speaker_header)

        segments = item["transcript"].get("segments", [])

        for seg in segments:
            text = seg["text"]
            logprob = seg.get("avg_logprob", -10)

            # Normalize logprob similar to confidence logic
            confidence = min(max(logprob + 1.0, 0.0), 1.0)

            if confidence < threshold:
                # Flag low confidence
                line = f"[LOW CONF {confidence:.2f}] {text}"
            else:
                line = f"[{confidence:.2f}] {text}"

            lines.append(line)

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


# -------------------------------------------------
# AUDIT ARTIFACTS GENERATION
# -------------------------------------------------
def generate_audit_artifacts(raw_audio_path, denoised_audio_path, transcripts):
    """
    Generate all audit artifacts for a processing job.

    Parameters
    ----------
    raw_audio_path : str
        Path to original raw audio
    denoised_audio_path : str
        Path to denoised audio
    transcripts : list
        List of transcription results

    Returns
    -------
    audit_report : dict
        Paths to generated artifacts
    """

    audit_dir = "data/audit"
    os.makedirs(audit_dir, exist_ok=True)

    # Extract base name for file naming
    base_name = os.path.basename(raw_audio_path)
    name, _ = os.path.splitext(base_name)

    audit_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "raw_spectrogram": None,
        "denoised_spectrogram": None,
        "transcript_file": None
    }

    # Generate raw audio spectrogram
    try:
        raw_spec_path = os.path.join(audit_dir, f"{name}_raw_spectrogram.png")
        generate_spectrogram(raw_audio_path, raw_spec_path, f"Raw Audio: {name}")
        audit_report["raw_spectrogram"] = raw_spec_path
    except Exception as e:
        print(f"Warning: Failed to generate raw spectrogram: {e}")

    # Generate denoised audio spectrogram
    try:
        denoised_spec_path = os.path.join(audit_dir, f"{name}_denoised_spectrogram.png")
        generate_spectrogram(denoised_audio_path, denoised_spec_path, f"Denoised Audio: {name}")
        audit_report["denoised_spectrogram"] = denoised_spec_path
    except Exception as e:
        print(f"Warning: Failed to generate denoised spectrogram: {e}")

    # Generate confidence-aware transcript
    try:
        transcript_path = os.path.join(audit_dir, f"{name}_transcript.txt")
        generate_confidence_transcript(transcripts, transcript_path)
        audit_report["transcript_file"] = transcript_path
    except Exception as e:
        print(f"Warning: Failed to generate transcript file: {e}")

    return audit_report


# -------------------------------------------------
# STANDALONE EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    """
    Run audit artifact generation as a standalone script.
    
    Usage:
        python audit.py <raw_audio_path> <denoised_audio_path>
    
    Example:
        python audit.py data/raw/audio.wav data/denoised/audio_denoised.wav
    """
    
    if len(sys.argv) < 3:
        print("Usage: python audit.py <raw_audio_path> <denoised_audio_path>")
        print("Example: python audit.py data/raw/audio.wav data/denoised/audio_denoised.wav")
        sys.exit(1)
    
    raw_audio = sys.argv[1]
    denoised_audio = sys.argv[2]
    
    print(f"Generating audit artifacts...")
    print(f"Raw audio: {raw_audio}")
    print(f"Denoised audio: {denoised_audio}")
    
    # Create mock transcripts for standalone testing
    mock_transcripts = [{
        "audio": denoised_audio,
        "transcript": {
            "text": "Sample transcript for testing",
            "segments": [
                {"text": "Sample segment", "avg_logprob": -0.3}
            ]
        }
    }]
    
    try:
        result = generate_audit_artifacts(raw_audio, denoised_audio, mock_transcripts)
        print(f"\nSUCCESS: Audit artifacts generated:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
