"""
confidence.py

Computes confidence scores for the SonClarus ML pipeline.

Confidence is NOT accuracy.
It is an estimate of reliability based on model-internal signals.
"""

import os
import sys
import numpy as np
import torchaudio


# -------------------------------------------------
# DENOISING CONFIDENCE
# -------------------------------------------------
def denoise_confidence(audio_path):
    """
    Estimate confidence of denoising stage.

    This is a conservative proxy based on signal stability.
    """

    waveform, _ = torchaudio.load(audio_path)
    waveform = waveform.numpy()

    # Signal energy
    energy = np.mean(waveform ** 2)

    # Heuristic normalization
    confidence = min(max(energy * 10, 0.0), 1.0)

    return confidence


# -------------------------------------------------
# SEPARATION CONFIDENCE
# -------------------------------------------------
def separation_confidence(speaker_audio_paths):
    """
    Estimate confidence of speaker separation.

    Uses energy dominance between separated speakers.
    """

    energies = []

    for path in speaker_audio_paths:
        waveform, _ = torchaudio.load(path)
        waveform = waveform.numpy()
        energies.append(np.mean(waveform ** 2))

    if len(energies) < 2:
        return 0.0

    max_energy = max(energies)
    total_energy = sum(energies)

    if total_energy == 0:
        return 0.0

    dominance_ratio = max_energy / total_energy

    # Normalize dominance ratio
    confidence = min(max(dominance_ratio, 0.0), 1.0)

    return confidence


# -------------------------------------------------
# TRANSCRIPTION CONFIDENCE
# -------------------------------------------------
def transcription_confidence(transcript_segments):
    """
    Estimate confidence from Whisper transcription segments.
    """

    if not transcript_segments:
        return 0.0

    avg_logprobs = [
        seg["avg_logprob"]
        for seg in transcript_segments
        if "avg_logprob" in seg
    ]

    if not avg_logprobs:
        return 0.0

    mean_logprob = np.mean(avg_logprobs)

    # Normalize log-probability
    confidence = min(max(mean_logprob + 1.0, 0.0), 1.0)

    return confidence


# -------------------------------------------------
# AGGREGATE CONFIDENCE
# -------------------------------------------------
def compute_overall_confidence(
    denoised_audio_path,
    speaker_audio_paths,
    transcription_results
):
    """
    Combine confidence scores from all ML stages.
    """

    d_conf = denoise_confidence(denoised_audio_path)
    s_conf = separation_confidence(speaker_audio_paths)

    t_confs = []
    for t in transcription_results:
        segs = t["transcript"].get("segments", [])
        t_confs.append(transcription_confidence(segs))

    t_conf = np.mean(t_confs) if t_confs else 0.0

    overall_confidence = (
        0.2 * d_conf +
        0.3 * s_conf +
        0.5 * t_conf
    )

    return {
        "denoise_confidence": round(d_conf, 3),
        "separation_confidence": round(s_conf, 3),
        "transcription_confidence": round(t_conf, 3),
        "overall_confidence": round(overall_confidence, 3)
    }


# -------------------------------------------------
# STANDALONE EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    """
    Run confidence computation as a standalone script.
    
    Usage:
        python confidence.py <denoised_audio_path> [speaker_audio_1] [speaker_audio_2] ...
    
    Example:
        python confidence.py data/denoised/audio_denoised.wav data/separated/audio_speaker_1.wav data/separated/audio_speaker_2.wav
    """
    
    if len(sys.argv) < 2:
        print("Usage: python confidence.py <denoised_audio_path> [speaker_audio_1] [speaker_audio_2] ...")
        print("Example: python confidence.py data/denoised/audio_denoised.wav data/separated/audio_speaker_1.wav")
        sys.exit(1)
    
    denoised_audio = sys.argv[1]
    speaker_audios = sys.argv[2:] if len(sys.argv) > 2 else []
    
    print(f"Computing confidence scores...")
    print(f"Denoised audio: {denoised_audio}")
    print(f"Speaker audios: {speaker_audios}")
    
    try:
        # Compute individual confidences
        d_conf = denoise_confidence(denoised_audio)
        print(f"\nDenoise confidence: {d_conf:.3f}")
        
        if speaker_audios:
            s_conf = separation_confidence(speaker_audios)
            print(f"Separation confidence: {s_conf:.3f}")
        
        # If we have all inputs, compute overall
        if speaker_audios:
            # Create mock transcription results for standalone testing
            mock_transcripts = [
                {"audio": path, "transcript": {"segments": []}}
                for path in speaker_audios
            ]
            
            result = compute_overall_confidence(
                denoised_audio,
                speaker_audios,
                mock_transcripts
            )
            
            print(f"\nOverall confidence report:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        
        print("\nSUCCESS!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
