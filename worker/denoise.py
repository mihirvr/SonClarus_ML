"""
denoise.py

This module performs intelligent noise reduction on audio signals
using a pretrained deep learning model.

Key principles:
- Inference only (no training)
- Preserve speech content
- Improve signal-to-noise ratio
- CPU-first execution
"""

import os
import torch
import torchaudio

from speechbrain.pretrained import SpectralMaskEnhancement


# -----------------------------
# MODEL LOADING (ONCE)
# -----------------------------
# We load the model at module level so it is reused across jobs.
# This avoids reloading weights for every audio file.
# This is critical for performance and memory efficiency.

ENHANCEMENT_MODEL = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir="pretrained_models/denoising"
)


def denoise_audio(audio_path, output_dir="data/denoised"):
    """
    Perform noise reduction on a raw audio file.

    Parameters
    ----------
    audio_path : str
        Path to the noisy input audio (.wav)

    output_dir : str
        Directory where denoised audio will be saved

    Returns
    -------
    denoised_audio_path : str
        Path to the denoised audio file
    """

    # -----------------------------
    # 1. Basic validation
    # -----------------------------
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # 2. Load audio
    # -----------------------------
    # torchaudio loads audio as:
    # waveform shape -> [channels, samples]
    waveform, sample_rate = torchaudio.load(audio_path)

    # -----------------------------
    # 3. Convert to mono if needed
    # -----------------------------
    # Most enhancement models expect mono audio.
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # -----------------------------
    # 4. Model expects batch dimension
    # -----------------------------
    # Shape becomes: [batch, time]
    waveform = waveform.squeeze(0).unsqueeze(0)

    # -----------------------------
    # 5. Apply denoising model
    # -----------------------------
    # This performs spectral masking internally.
    with torch.no_grad():
        enhanced_waveform = ENHANCEMENT_MODEL.enhance_batch(
            waveform,
            lengths=torch.tensor([1.0])
        )

    # -----------------------------
    # 6. Prepare output filename
    # -----------------------------
    base_name = os.path.basename(audio_path)
    name, _ = os.path.splitext(base_name)

    denoised_audio_path = os.path.join(
        output_dir,
        f"{name}_denoised.wav"
    )

    # -----------------------------
    # 7. Save denoised audio
    # -----------------------------
    torchaudio.save(
        denoised_audio_path,
        enhanced_waveform.squeeze(0),
        sample_rate
    )

    return denoised_audio_path
