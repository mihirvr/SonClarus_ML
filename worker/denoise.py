"""
denoise.py

This module performs *soft* speech-preserving noise reduction
using DeepFilterNet3 for real-time audio enhancement.

Design goals:
- Inference only (no training)
- Preserve speech intelligibility
- Avoid aggressive over-suppression
- CPU-first execution
- Stable behavior for transcription pipelines

Uses the official DeepFilterNet3 model.
"""

import os
import sys

import numpy as np

# DeepFilterNet imports
from df.enhance import enhance, init_df, load_audio, save_audio


# -----------------------------
# CONFIGURATION
# -----------------------------
_TARGET_SAMPLE_RATE = 48000  # DeepFilter requires 48kHz

# Global model cache
_DEEPFILTER_MODEL = None
_DEEPFILTER_STATE = None


def _get_deepfilter_model():
    """
    Lazy-load and cache the DeepFilterNet3 model.
    """
    global _DEEPFILTER_MODEL, _DEEPFILTER_STATE
    
    if _DEEPFILTER_MODEL is None:
        # Initialize DeepFilterNet3 model
        # Downloads model automatically on first run
        _DEEPFILTER_MODEL, _DEEPFILTER_STATE, _ = init_df()
    
    return _DEEPFILTER_MODEL, _DEEPFILTER_STATE


def denoise_audio(audio_path, output_dir="data/denoised", atten_lim_db=None):
    """
    Perform speech-preserving noise reduction on an audio file.

    Uses DeepFilterNet3 for high-quality noise suppression.

    Parameters
    ----------
    audio_path : str
        Path to the noisy input audio (.wav)

    output_dir : str
        Directory where denoised audio will be saved

    atten_lim_db : float, optional
        Maximum attenuation in dB. If provided, limits noise reduction.
        Default: None (use DeepFilter defaults)

    Returns
    -------
    denoised_audio_path : str
        Path to the denoised audio file
    """

    # -----------------------------
    # 1. Validation
    # -----------------------------
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # 2. Load DeepFilter model
    # -----------------------------
    model, df_state = _get_deepfilter_model()

    # -----------------------------
    # 3. Load audio using DeepFilter's loader
    # -----------------------------
    # DeepFilter handles resampling to 48kHz internally
    audio, _ = load_audio(audio_path, sr=df_state.sr())

    # -----------------------------
    # 4. Apply DeepFilter enhancement
    # -----------------------------
    if atten_lim_db is not None:
        enhanced_audio = enhance(model, df_state, audio, atten_lim_db=atten_lim_db)
    else:
        enhanced_audio = enhance(model, df_state, audio)

    # -----------------------------
    # 5. Prepare output filename
    # -----------------------------
    base_name = os.path.basename(audio_path)
    name, _ = os.path.splitext(base_name)

    denoised_audio_path = os.path.join(
        output_dir,
        f"{name}_denoised.wav"
    )

    # -----------------------------
    # 6. Save denoised audio
    # -----------------------------
    save_audio(denoised_audio_path, enhanced_audio, df_state.sr())

    return denoised_audio_path


# -----------------------------
# STANDALONE EXECUTION
# -----------------------------
if __name__ == "__main__":
    """
    Run denoising as a standalone script.
    
    Usage:
        python denoise.py <input_audio_path> [output_directory] [atten_lim_db]
    
    Example:
        python denoise.py data/raw/noisy_audio.wav data/denoised
        python denoise.py data/raw/noisy_audio.wav data/denoised -20
    """
    
    if len(sys.argv) < 2:
        print("Usage: python denoise.py <input_audio_path> [output_directory] [atten_lim_db]")
        print("Example: python denoise.py data/raw/noisy_audio.wav data/denoised")
        print()
        print("Arguments:")
        print("  input_audio_path  : Path to noisy audio file")
        print("  output_directory  : Where to save denoised audio (default: data/denoised)")
        print("  atten_lim_db      : Optional max attenuation in dB (e.g., -20)")
        sys.exit(1)
    
    input_audio = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else "data/denoised"
    atten_db = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    print(f"DeepFilterNet3 Denoising")
    print(f"=" * 40)
    print(f"Input: {input_audio}")
    print(f"Output directory: {output_directory}")
    if atten_db:
        print(f"Attenuation limit: {atten_db} dB")
    print()
    
    try:
        result_path = denoise_audio(input_audio, output_directory, atten_db)
        print(f"SUCCESS: Denoised audio saved to: {result_path}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
