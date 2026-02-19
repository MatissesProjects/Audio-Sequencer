# Tech Stack: AudioSequencer AI

## Language & Core
- **Primary Language:** Python 3.10+ (Required for AI model compatibility).
- **Environment Management:** `venv` or `conda`.

## Audio Analysis & DSP (MIR)
- **Feature Extraction:** `librosa` (BPM, Chromagram, onset detection).
- **High-End DSP:** `pedalboard` (Spotify's wrapper for JUCE - used for VST hosting, EQs, and high-quality stretching).
- **Time/Pitch Scaling:** `pyrubberband` (Interface for Rubber Band Library).
- **Export/Mixing:** `pydub` (Glue for final rendering).

## AI & Vector Search
- **Audio Embeddings:** `laion-clap` (Contrastive Language-Audio Pretraining).
- **Vector Database:** `ChromaDB` (Local-first, simple vector storage).
- **Generation (Phase 5):** `AudioCraft` (MusicGen) via `torch`.

## Frontend / GUI
- **Framework:** `PyQt6` or `PySide6` for a native desktop experience with complex canvas interactions.
- **Visualizations:** `matplotlib` or `pyqtgraph` for waveform rendering.

## Storage
- **Relational Data:** `SQLite` (for file metadata, paths, and user tags).
- **Vector Data:** `ChromaDB` (for high-dimensional embeddings).
