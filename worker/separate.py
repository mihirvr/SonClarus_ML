"""
separate.py

Performs blind source separation (BSS) on denoised audio
using a pretrained SepFormer model.

This module assumes:
- Mono audio input
- At most two dominant speakers
- Inference-only usage
"""

import os
import torch
import torchaudio

from speechbrain.pretrained import SepformerSeparation


# -------------------------------------------------
# MODEL LOADING (GLOBAL, ONCE)
# -------------------------------------------------
# Loading the separation model is expensive.
# We do it once per worker process.

SEPARATION_MODEL = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/separation"
)


def separate_speakers(audio_path, output_dir="data/separated"):
    """
    Separate overlapping speakers from a denoised audio file.

    Parameters
    ----------
    audio_path : str
        Path to denoised mono audio file

    output_dir : str
        Directory to store separated speaker audio files

    Returns
    -------
    speaker_paths : list[str]
        List of file paths for separated speakers
    """

    # -------------------------------------------------
    # 1. Validation
    # -------------------------------------------------
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------
    # 2. Load audio
    # -------------------------------------------------
    waveform, sample_rate = torchaudio.load(audio_path)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Model expects [batch, time]
    waveform = waveform.squeeze(0).unsqueeze(0)

    # -------------------------------------------------
    # 3. Run separation model
    # -------------------------------------------------
    with torch.no_grad():
        separated_sources = SEPARATION_MODEL.separate_batch(waveform)

    # Output shape:
    # [batch, time, num_speakers]
    separated_sources = separated_sources.squeeze(0)

    # -------------------------------------------------
    # 4. Save separated audio files
    # -------------------------------------------------
    base_name = os.path.basename(audio_path)
    name, _ = os.path.splitext(base_name)

    speaker_paths = []

    for idx in range(separated_sources.shape[1]):
        speaker_waveform = separated_sources[:, idx].unsqueeze(0)

        speaker_path = os.path.join(
            output_dir,
            f"{name}_speaker_{idx + 1}.wav"
        )

        torchaudio.save(
            speaker_path,
            speaker_waveform,
            sample_rate
        )

        speaker_paths.append(speaker_path)

    return speaker_paths
