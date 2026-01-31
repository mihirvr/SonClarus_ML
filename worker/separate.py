"""
separate.py

Performs speaker separation on denoised audio
using spectral clustering and energy-based segmentation.

This module assumes:
- Mono audio input
- At most two dominant speakers
- Inference-only usage

Note: This is a simplified separation approach.
For production use, consider dedicated speaker diarization libraries.
"""

import os
import sys
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from scipy.ndimage import median_filter


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
_TARGET_SAMPLE_RATE = 16000
_HOP_LENGTH = 512
_N_FFT = 2048


def separate_speakers(audio_path, output_dir="data/separated", num_speakers=2):
    """
    Separate overlapping speakers from a denoised audio file.

    Uses spectral analysis and energy-based clustering to
    separate audio into distinct speaker tracks.

    Parameters
    ----------
    audio_path : str
        Path to denoised mono audio file

    output_dir : str
        Directory to store separated speaker audio files

    num_speakers : int
        Expected number of speakers (default: 2)

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

    # Convert to numpy for librosa processing
    audio_np = waveform.squeeze().numpy()

    # Resample if needed
    if sample_rate != _TARGET_SAMPLE_RATE:
        audio_np = librosa.resample(
            audio_np, 
            orig_sr=sample_rate, 
            target_sr=_TARGET_SAMPLE_RATE
        )
        sample_rate = _TARGET_SAMPLE_RATE

    # -------------------------------------------------
    # 3. Compute STFT for spectral analysis
    # -------------------------------------------------
    stft = librosa.stft(audio_np, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # -------------------------------------------------
    # 4. Perform spectral separation using NMF
    # -------------------------------------------------
    # Non-negative Matrix Factorization for source separation
    from sklearn.decomposition import NMF

    # Reshape magnitude for NMF
    nmf = NMF(n_components=num_speakers, init='random', random_state=42, max_iter=300)
    W = nmf.fit_transform(magnitude.T)  # Time x Components
    H = nmf.components_  # Components x Frequency

    # Reconstruct separated spectrograms
    separated_magnitudes = []
    for i in range(num_speakers):
        # Reconstruct each component's spectrogram
        component_mag = np.outer(W[:, i], H[i, :]).T
        separated_magnitudes.append(component_mag)

    # -------------------------------------------------
    # 5. Apply soft masking for better separation
    # -------------------------------------------------
    total_magnitude = np.sum(separated_magnitudes, axis=0) + 1e-10
    
    separated_audio = []
    for i, sep_mag in enumerate(separated_magnitudes):
        # Soft mask
        mask = sep_mag / total_magnitude
        mask = median_filter(mask, size=3)  # Smooth the mask
        
        # Apply mask to original STFT
        masked_stft = mask * stft
        
        # Inverse STFT
        audio_reconstructed = librosa.istft(masked_stft, hop_length=_HOP_LENGTH)
        separated_audio.append(audio_reconstructed)

    # -------------------------------------------------
    # 6. Save separated audio files
    # -------------------------------------------------
    base_name = os.path.basename(audio_path)
    name, _ = os.path.splitext(base_name)

    speaker_paths = []

    for idx, speaker_audio in enumerate(separated_audio):
        speaker_path = os.path.join(
            output_dir,
            f"{name}_speaker_{idx + 1}.wav"
        )

        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(speaker_audio))
        if max_val > 0:
            speaker_audio = speaker_audio / max_val * 0.95

        sf.write(speaker_path, speaker_audio, sample_rate)
        speaker_paths.append(speaker_path)

    return speaker_paths


# -------------------------------------------------
# STANDALONE EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    """
    Run speaker separation as a standalone script.
    
    Usage:
        python separate.py <input_audio_path> [output_directory] [num_speakers]
    
    Example:
        python separate.py data/denoised/audio_denoised.wav data/separated 2
    """
    
    if len(sys.argv) < 2:
        print("Usage: python separate.py <input_audio_path> [output_directory] [num_speakers]")
        print("Example: python separate.py data/denoised/audio_denoised.wav data/separated 2")
        sys.exit(1)
    
    input_audio = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else "data/separated"
    num_spk = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    print(f"Separating speakers from: {input_audio}")
    print(f"Output directory: {output_directory}")
    print(f"Number of speakers: {num_spk}")
    
    try:
        result_paths = separate_speakers(input_audio, output_directory, num_spk)
        print(f"SUCCESS: Separated audio saved to:")
        for path in result_paths:
            print(f"  - {path}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
