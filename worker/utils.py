"""
utils.py

Utility helpers shared across ML modules.
"""

import os
import sys
import librosa
import soundfile as sf


def validate_audio(audio_path):
    """
    Validates format, duration, and readability of an audio file.
    
    Parameters
    ----------
    audio_path : str
        Path to the audio file to validate
        
    Returns
    -------
    dict
        Validation result with details about the audio file
    """
    result = {
        "valid": False,
        "path": audio_path,
        "exists": False,
        "readable": False,
        "duration": None,
        "sample_rate": None,
        "channels": None,
        "error": None
    }
    
    # Check if file exists
    if not os.path.exists(audio_path):
        result["error"] = f"File not found: {audio_path}"
        return result
    
    result["exists"] = True
    
    try:
        # Try to load audio info
        info = sf.info(audio_path)
        result["readable"] = True
        result["duration"] = info.duration
        result["sample_rate"] = info.samplerate
        result["channels"] = info.channels
        result["valid"] = True
    except Exception as e:
        result["error"] = str(e)
        
    return result


def get_audio_duration(audio_path):
    """
    Get the duration of an audio file in seconds.
    
    Parameters
    ----------
    audio_path : str
        Path to the audio file
        
    Returns
    -------
    float
        Duration in seconds
    """
    info = sf.info(audio_path)
    return info.duration


def resample_audio(audio_path, target_sr, output_path=None):
    """
    Resample an audio file to a target sample rate.
    
    Parameters
    ----------
    audio_path : str
        Path to the input audio file
    target_sr : int
        Target sample rate
    output_path : str, optional
        Output path. If None, overwrites the input file.
        
    Returns
    -------
    str
        Path to the resampled audio file
    """
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    if output_path is None:
        output_path = audio_path
        
    sf.write(output_path, audio, target_sr)
    return output_path


# -------------------------------------------------
# STANDALONE EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    """
    Run audio validation as a standalone script.
    
    Usage:
        python utils.py <audio_path>
    
    Example:
        python utils.py data/raw/audio.wav
    """
    
    if len(sys.argv) < 2:
        print("Usage: python utils.py <audio_path>")
        print("Example: python utils.py data/raw/audio.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print(f"Validating audio file: {audio_file}")
    print("-" * 40)
    
    result = validate_audio(audio_file)
    
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    if result["valid"]:
        print("")
        print("Audio file is VALID")
    else:
        print("")
        print(f"Audio file is INVALID: {result['error']}")
        sys.exit(1)
