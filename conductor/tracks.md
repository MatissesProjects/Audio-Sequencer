# Tracks Registry

This file tracks the epic-level progress of the AudioSequencer AI project.

## Core Epics

| ID | Title | Status | Target | Description |
|---|---|---|---|---|
| `sys-data` | **Data Persistence & State** | ✅ Completed | MVP | SQLite schema for file metadata and ChromaDB setup for vector storage. |
| `core-mir` | **Analysis Engine (MIR)** | ✅ Completed | MVP | `librosa`-based ingestion: BPM, Key, and LUFS extraction. |
| `ai-embed` | **AI Embedding Engine** | ✅ Completed | v1.0 | Implement local `laion-clap` to convert audio to 512-d vectors for semantic search. |
| `logic-seq` | **Sequencing & Scoring Logic** | ✅ Completed | MVP | The algorithm calculating compatibility scores (Tempo, Harmonic, Vector similarity). |
| `audio-dsp` | **Audio Manipulation (DSP)** | ✅ Completed | MVP | `pedalboard` implementation for artifact-free time-stretching and pitch-shifting. |
| `audio-rndr`| **Playback & Export Engine** | ✅ Completed | MVP | `pydub` engine for applying crossfades, EQ ducking, and rendering the final mix file. |
| `gui-core` | **UI Canvas & Browser** | 🏗️ In Progress | MVP | PyQt6 library browser, drag-and-drop timeline, and real-time state visualization. |
| `ai-gen` | **Generative Audio Expansion** | ⏳ Blocked | v2.0 | Local `AudioCraft` generation for custom transition sweeps/risers. |

## Dependency Graph
- `sys-data` is foundational.
- `ai-embed` depends on `sys-data` and `core-mir`.
- `logic-seq` depends on `sys-data` and `core-mir`.
- `audio-dsp` runs independently but is fed parameters by `logic-seq`.
- `audio-rndr` depends on `audio-dsp`.
- `gui-core` depends on all MVP tracks to function fully.

## Active Sprint / Next Actions
1. Spin up the SQLite database schema (`sys-data`) to store the outputs from `core-mir`.
2. Write unit tests to ensure `core-mir` accurately captures transient data for seamless looping.
