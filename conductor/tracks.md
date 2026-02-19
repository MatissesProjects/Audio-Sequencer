# Tracks Registry

This file tracks the high-level progress of the AudioSequencer AI project.

| ID | Title | Status | Description |
|---|---|---|---|
| `core-mir` | **Analysis Engine (Core MIR)** | 🏗️ In Progress | librosa-based ingestion and feature extraction. |
| `vector-search` | **AI Recommendation System** | ⏳ Blocked | CLAP embeddings and ChromaDB integration. |
| `seq-engine` | **Sequencing & Layering** | ⏳ Blocked | Horizontal stitching and vertical layering logic. |
| `gui-canvas` | **UI & Workflow** | ⏳ Blocked | PyQt6 interface and the "Flow" canvas. |
| `ai-gen` | **Advanced AI Expansion** | ⏳ Blocked | Local AudioCraft integration for gap filling. |

## Dependencies
- `vector-search` depends on `core-mir`.
- `seq-engine` depends on `core-mir`.
- `gui-canvas` depends on all core engines.
