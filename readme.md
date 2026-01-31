# 🌊 SonClarus
**Intelligence in Every Wave**

---

## 🚀 Running the Scripts (Terminal Commands)

> ⚠️ **Important**: Don't use the IDE's "Run" button. Run scripts from the terminal with arguments.

### Full Pipeline (Recommended)
```bash
# Process audio through all stages: denoise → separate → transcribe → audit
python worker/engine.py data/raw/noisy_audio.wav
```

### Individual Modules

**1. Denoise Audio** - Remove background noise
```bash
python worker/denoise.py <input_audio> [output_dir] [atten_lim_db]

# Examples:
python worker/denoise.py data/raw/noisy_audio.wav
python worker/denoise.py data/raw/noisy_audio.wav data/denoised
python worker/denoise.py data/raw/noisy_audio.wav data/denoised -20
```

**2. Separate Speakers** - Split audio by speaker
```bash
python worker/separate.py <input_audio> [output_dir] [num_speakers]

# Examples:
python worker/separate.py data/denoised/noisy_audio_denoised.wav
python worker/separate.py data/denoised/noisy_audio_denoised.wav data/separated 2
```

**3. Transcribe Audio** - Speech to text
```bash
python worker/transcribe.py <audio_path> [model_size] [output_dir]

# Model sizes: tiny, base, small, medium, large
# Examples:
python worker/transcribe.py data/separated/audio_speaker_1.wav
python worker/transcribe.py data/separated/audio_speaker_1.wav base
python worker/transcribe.py data/separated/audio_speaker_1.wav base data/transcripts
```
Output: Saves `<filename>_transcript.txt` and `<filename>_transcript.json` to output_dir (default: `data/transcripts/`)

**4. Generate Audit Artifacts** - Spectrograms & confidence reports
```bash
python worker/audit.py <raw_audio> <denoised_audio>

# Example:
python worker/audit.py data/raw/noisy_audio.wav data/denoised/noisy_audio_denoised.wav
```

---

## ⚠️ Setup Notes (READ FIRST)

### DeepFilterNet Installation on Windows

DeepFilterNet requires the `deepfilterlib` Rust library. On Windows, **pre-built wheels are available** but pip may fail to find them automatically.

**What was done:**
1. `pip install deepfilternet` initially failed because it tried to compile `deepfilterlib` from source (requires Rust)
2. Solution: Manually download and install the pre-built wheel first:
   ```bash
   pip download deepfilterlib --only-binary :all: --dest temp_wheels
   pip install temp_wheels/deepfilterlib-0.5.6-cp311-none-win_amd64.whl
   pip install deepfilternet
   ```

**For future setup:**
```bash
# 1. Create conda environment
conda create -n sonclarus python=3.11 -y
conda activate sonclarus

# 2. Install PyTorch first
pip install torch==2.1.2 torchaudio==2.1.2

# 3. Install DeepFilterLib wheel manually (Windows)
pip download deepfilterlib --only-binary :all: --dest temp_wheels
pip install temp_wheels/deepfilterlib-*.whl
rmdir /s /q temp_wheels

# 4. Install remaining requirements
pip install -r requirements.txt
```

**Note:** On Linux/Mac, `pip install deepfilternet` should work directly.

---

## 📁 Project Structure

### Worker Modules (`worker/`)

| File | Purpose | Key Library |
|------|---------|-------------|
| `denoise.py` | Removes background noise from audio | **DeepFilterNet3** |
| `separate.py` | Separates overlapping speakers into individual tracks | **scikit-learn (NMF)** |
| `transcribe.py` | Converts speech to text | **OpenAI Whisper** |
| `confidence.py` | Computes reliability scores for each ML stage | numpy, torchaudio |
| `audit.py` | Generates spectrograms and confidence-marked transcripts | matplotlib, torchaudio |
| `engine.py` | Orchestrates the full ML pipeline | All above modules |
| `utils.py` | Audio validation and utility helpers | soundfile, librosa |
| `config.py` | Configuration settings (currently empty) | - |

### Data Directories

| Directory | Purpose |
|-----------|---------|
| `data/raw/` | Input audio files |
| `data/denoised/` | Noise-reduced audio output |
| `data/separated/` | Speaker-separated audio tracks |
| `data/audit/` | Spectrograms and transcript artifacts |
| `audio_cache/` | Temporary audio cache |
| `pretrained_models/` | Downloaded model checkpoints |

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `deepfilternet` | 0.5.6 | **Denoising** - State-of-the-art deep learning noise suppression |
| `torch` | 2.1.2 | Deep learning framework (CPU) |
| `torchaudio` | 2.1.2 | Audio loading and transforms |
| `openai-whisper` | latest | Speech-to-text transcription |
| `librosa` | latest | Audio analysis and resampling |
| `scikit-learn` | latest | NMF for speaker separation |
| `soundfile` | latest | Audio file I/O |
| `scipy` | latest | Signal processing utilities |
| `numpy` | 1.26.4 | Numerical operations |
| `matplotlib` | latest | Spectrogram visualization |
| `huggingface_hub` | 0.19.4 | Model downloading |
| `tqdm` | latest | Progress bars |

---

## 🔧 How It Works

### Pipeline Flow

```
Raw Audio → Denoise → Separate Speakers → Transcribe → Confidence → Audit
    │           │            │               │             │          │
    │           ▼            ▼               ▼             ▼          ▼
    │      DeepFilterNet  NMF/Spectral    Whisper      Energy +    Spectrograms
    │      (48kHz)        Clustering      (16kHz)      LogProb     + Transcripts
    │                                                  Analysis
    ▼
  data/raw/    data/denoised/   data/separated/              data/audit/
```

### 1. Denoising (`denoise.py`)
- Uses **DeepFilterNet3** (deep learning model)
- Processes at 48kHz sample rate
- Removes background noise while preserving speech
- Model auto-downloads on first run (~100MB)

### 2. Speaker Separation (`separate.py`)
- Uses **Non-negative Matrix Factorization (NMF)**
- Spectral analysis to separate overlapping voices
- Configurable number of speakers (default: 2)
- Outputs individual speaker tracks

### 3. Transcription (`transcribe.py`)
- Uses **OpenAI Whisper** (base model by default)
- Runs on CPU with fp16=False
- Returns text, timestamps, and confidence (log probabilities)
- Supports multiple languages

### 4. Confidence Scoring (`confidence.py`)
- Computes reliability estimates for each stage
- Denoising: Signal energy analysis
- Separation: Speaker energy dominance ratio
- Transcription: Average log probability from Whisper
- Weighted overall score: 20% denoise + 30% separation + 50% transcription

### 5. Audit Artifacts (`audit.py`)
- Generates before/after spectrograms
- Creates confidence-marked transcripts
- Enables human review and verification

---

## 🚀 Running the Scripts

**All scripts are self-sufficient and can be run directly.**

Each module in `worker/` has a `if __name__ == "__main__"` block, so you can run any script standalone without importing or calling from another file.

### Full Pipeline
```bash
cd worker
python engine.py ../data/raw/your_audio.wav
```

### Individual Modules (All Standalone)
```bash
cd worker

# Denoise audio
python denoise.py <input.wav> [output_dir] [atten_limit_db]

# Separate speakers
python separate.py <input.wav> [output_dir] [num_speakers]

# Transcribe audio
python transcribe.py <input.wav> [model_size]

# Compute confidence scores
python confidence.py <denoised.wav> [speaker1.wav] [speaker2.wav]

# Generate audit artifacts
python audit.py <raw.wav> <denoised.wav>

# Validate audio file
python utils.py <input.wav>
```

### Examples
```bash
# Denoise a noisy recording
python denoise.py ../data/raw/meeting.wav ../data/denoised

# Separate into 2 speakers
python separate.py ../data/denoised/meeting_denoised.wav ../data/separated 2

# Transcribe speaker 1
python transcribe.py ../data/separated/meeting_denoised_speaker_1.wav base

# Run full pipeline on a file
python engine.py ../data/raw/interview.wav
```

---

## What This Project Is
SonClarus is an asynchronous audio intelligence system designed to help analysts work with long, noisy recordings that contain overlapping human speech.

The system focuses on **processing quality, explainability, and trust**, not real-time performance.

---

## Core ML Capabilities
The ML subsystem is responsible for:
- Removing background noise from audio
- Separating overlapping speakers without prior identity knowledge
- Transcribing speech into text
- Estimating confidence for all AI outputs
- Producing transparency artifacts for audit and review

All ML processing is **offline, batch-based, and CPU-oriented**.

---

## ML Design Constraints
- Pretrained models only
- No training or fine-tuning
- Maximum two dominant speakers
- Latency is acceptable
- Explainability is mandatory

---

## Architectural Philosophy
The system separates:
- Control logic (API)
- Heavy computation (ML worker)

This prevents blocking and ensures scalability.

---

## ML Principles Followed
- Modular pipeline design
- Explicit intermediate outputs
- Confidence-aware AI
- Auditability over automation

---

## Intended Audience
- Intelligence analysts
- Academic evaluators
- Researchers working with noisy audio data

---

## Status
This repository is under active development.
The ML subsystem is built incrementally and deliberately.
