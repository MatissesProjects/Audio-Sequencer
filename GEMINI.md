## 🛠 Environment & Tooling

### Shell & Commands (Windows PowerShell)
- **Command Chaining:** When running multiple commands in a single `run_shell_command` call, use the semicolon `;` as a separator instead of `&&`. 
    - *Correct:* `git status; git log`
    - *Incorrect:* `git status && git log` (Causes a ParserError in PowerShell).

## General rules

Always take the chunks we have worked on and break them into logical git commits

Always attempt to build anything that needs to be built

## 🧠 Lessons Learned & Technical Insights

### Audio Analysis & MIR
- **Beat Onsets are Critical:** Accurate looping depends on using detected beat onsets as loop points. Simple duration-based looping causes rhythmic drift and "jumping."
- **BPM/Key Accuracy:** `librosa` is highly reliable for 30s clips, but storing these mathematical properties in SQLite is essential to avoid redundant analysis.

### DSP & Rendering Refinements
- **Eliminating "Jumping":** Crossfading is mandatory at loop points. A **500ms equal-power crossfade** (using sin/cos curves) provides a seamless transition without volume dips or clicks.
- **Buttery Smooth Transitions:** Linear ramps feel unnatural. **S-Curve (Sinusoidal) Easing** follows human loudness perception more closely and eliminates the sensation of "skipping" during track changes.
- **The "Bass Swap" Technique:** To prevent muddy mixes, use parallel filter blending. Progressively High-pass the outgoing track and Low-pass the incoming one during overlaps.
- **Gain Staging:** Always normalize tracks to a consistent RMS level before mixing. Use a professional **Limiter** (e.g., `pedalboard.Limiter`) on the master bus to prevent clipping during complex layers.

### AI Integration
- **Semantic Discovery (CLAP):** Vector-based similarity using local CLAP embeddings allows for "vibe-based" sequencing that math-only analysis (BPM/Key) misses.
- **Gemini Orchestration:** The `google-genai` library is superior for orchestrating transition parameters (noise types, filter cutoffs) based on track metadata.
- **Multi-Dimensional Scoring:** Combining BPM proximity, Harmonic compatibility (Circle of Fifths), and Semantic vibe into a single weighted score creates the most musical sequences.

### Architectural Best Practices
- **Parallel Filter Blending:** Instead of chunked processing (which can click), apply constant filters to entire overlapping segments and blend the *wet/dry signals* using automation curves.
- **Foundation + Layers:** Professional "flow" is often better achieved by extending a core "Foundation" track and intelligently ducking it to let shorter "Interlude" layers breathe.