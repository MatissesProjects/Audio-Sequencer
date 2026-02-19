# Product Guidelines: AudioSequencer AI

## Audio Standards
- **Sample Rate Consistency:** Internal processing should maintain a consistent sample rate (e.g., 44.1kHz or 48kHz) to avoid aliasing.
- **Artifact-Free Stretching:** Use `pedalboard` or `rubberband` for time-stretching. Avoid basic linear interpolation which causes "chipmunking" or "granularity" artifacts.
- **Gain Staging:** Ensure crossfades and layering don't clip. Implement automatic peak normalization or LUFS matching before mixing.

## AI & UX Integration
- **Transparent AI:** When a recommendation is made, show the "Confidence Score" and the reason (e.g., "Semantic Match: 94%", "Key Match: Perfect").
- **Local Priority:** Never block the UI for AI analysis. All CLAP/MusicGen operations should run in background threads with progress indicators.
- **Non-Destructive Editing:** All sequencing operations should be metadata-driven until the final "Export" phase.

## Visual Aesthetic
- **Waveform-Centric:** The UI should center around high-fidelity waveform visualizations.
- **Node-Based Flow:** Explore a node-based or fluid timeline rather than a rigid DAW-style grid to emphasize the "flow" aspect.
- **Responsive Feedback:** UI components should react to audio levels or "vibe" (e.g., color shifts based on harmonic key).
