# Track Plan: ai-gen (Generative Audio Expansion)

## Objective
Use Google libraries and Gemini to generate custom audio transitions (risers, sweeps, fills) that bridge two structurally incompatible tracks.

## Tasks
1. [ ] **Gemini Orchestrator:** Integrate `google-generativeai` to analyze the "vibe" of Track A and Track B and generate a specialized prompt for a transition.
2. [ ] **Local Audio Generation:** Set up a local high-fidelity generator (preferring Magenta or MusicGen) optimized for the RTX 4090.
3. [ ] **Smart Bridging:** Implement logic to determine when a transition is needed (e.g., Score < 60%) and trigger the generation.
4. [ ] **Stitching Logic:** Automatically insert the generated transition between the two tracks in the renderer.

## Dependencies
- `google-generativeai` (for Gemini orchestration).
- `magenta` or `audiocraft` (for local high-fidelity audio output).
- `ffmpeg` (required for complex audio stitching).

## Hardware Strategy
- **RTX 4090:** Dedicated to the local audio generation model (High VRAM usage).
- **RTX 3070:** Handles GUI, Analysis, and CLAP embeddings.
