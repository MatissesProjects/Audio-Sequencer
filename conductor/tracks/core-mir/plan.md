# Track Plan: core-mir (Analysis Engine)

## Objective
Build the foundational "Ingestion Pipeline" that watches a directory, analyzes audio files using `librosa`, and stores the mathematical metadata in a local SQLite database.

## User Stories
- As a user, I want my audio files to be automatically analyzed for BPM and Key when I add them to my library.
- As a user, I want this analysis to be persistent so it doesn't re-run every time I open the app.

## Tasks
1. [x] **Project Scaffolding:** Set up Python environment and `requirements.txt`.
2. [x] **Database Schema:** Design and implement a SQLite schema for audio metadata (BPM, Key, Energy, File Path, Duration).
3. [x] **Analysis Module:** Implement a `librosa` wrapper that extracts:
    - Tempo (BPM)
    - Chromagram / Harmonic Key
    - Loudness (LUFS/RMS)
4. [x] **Ingestion Watcher:** Simple script to scan a folder and process all supported audio files (.wav, .mp3, .flac).
5. [x] **Validation:** Create a test suite with known audio samples to verify BPM/Key accuracy.

## Definition of Done
- A CLI script can scan a folder and populate a SQLite database with accurate musical metadata.
- Unit tests pass for the analysis extraction logic.
