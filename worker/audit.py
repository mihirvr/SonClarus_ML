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
import datetime

import numpy as np
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

    spectrogram_db = 10 * torch_log10_safe(spectrogram)

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


def torch_log10_safe(x):
    """
    Safe log10 operation to avoid log(0).
    """
    return torch_safe_log10(x + 1e-10)


def torch_safe_log10(x):
    import torch
    return torch.log10(x)


# -------------------------------------------------
# CONFIDENCE-AWARE TRANSCRIPT
# -------------------------------------------------
def generate_confidence_transcript(transcripts, output_path, threshold=0.5):
    """
    Create a human-readable transcript with confidence markers.

    Low-confidence segments are clearly flagged.
    """

    lines = []

    for idx, item in enumerate(transcripts, start=1):
        speaker_header = f"\n--- Speaker {idx} ---"
        lines.append(speaker_header)

        segments = item["transcript"].get("segments", [])

        for seg in segments:
            text = seg["text"]
            logprob = seg.get("avg_logprob", -10)

            # Normalize logprob similar to confidence logic
            confi
