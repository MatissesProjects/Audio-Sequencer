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
- **Smart Loop Detection:** Automatically finding the most energetic 4-bar section (using RMS energy over a sliding window) provides a better default starting point for loops than the absolute start of a file.
- **Rhythmic Density (Groove):** Onset density (beats per second) is a critical factor for "Groove Lock." Matching tracks with similar densities prevents rhythmic clashing.

### DSP & Rendering Refinements
- **Dynamic Sidechain Ducking:** Implementing "Fake Sidechain" by calculating the RMS envelope of a lead track and applying the inverse to background tracks creates professional "pumping" and clarity.
- **Spectral Masking (Auto-EQ):** Automatically applying a 300Hz-400Hz High-Pass Filter to background/ambient tracks clears critical room for the lead track's bass and kick.
- **Dual-Edge Trimming:** When trimming the start (left edge) of a clip, you must adjust both the `start_ms` and the internal `offset_ms` in inverse directions to keep the audio anchored in place while changing its boundary.
- **S-Curve (Sinusoidal) Easing:** Essential for natural loudness perception. Sinusoidal curves prevent the "linear volume dip" feel at the center of crossfades.

### AI Integration
- **Neural Grain Clouds:** Granular synthesis (chopping into 150ms grains with randomization) is a powerful way to transform rhythmic material into cinematic atmospheric pads for intros and outros.
- **Stochastic Orchestration:** Purely deterministic AI arrangements feel repetitive. Introducing randomness in section lengths, panning distributions, and track selection (weighted by vibe) makes automated music feel "human."
- **Stochastic Production Edits:** Applying micro-chopping (4-bar -> 2-bar -> 1-bar) with rising pitch and high-pass filters generates genuine tension for builds.

### UI/UX & Workflow
- **Zero-Wait Startup:** Heavy AI libraries (`torch`, `laion-clap`) must be initialized in a background thread *after* the UI is shown. Use **Lazy-Import Isolation** (moving imports inside methods) to prevent the main thread from stalling.
- **Visual Feedback:** Drag-and-drop operations need a visual "ghost" (pixmap) of the item being moved to feel responsive.
- **Silence Guard:** Real-time "Energy Scanning" (sampling combined volume every 500ms) is more effective than simple block-checking for detecting "dead air" or quiet transitions.
- **Draggable Splitters:** In high-density layouts (Library + Mixer + Timeline), using `QSplitter` is mandatory to prevent automatic layout "shrinking" and allow users to manually balance their workspace.

### Architectural Best Practices
- **Robust Data Handling:** When retrieving numeric values from SQLite (which may return `NULL`), always use the `(value or 0)` pattern (e.g., `track.get('energy') or 0`) instead of `get('energy', 0)`. The latter only applies if the key is missing entirely, whereas `NULL` in the database results in a `None` value which will crash sorting and numeric comparisons.
- **Signal/Slot Decoupling:** Large UI components (like a Timeline) should never call `self.window()`. Instead, they should emit signals (`undoRequested`, `trackDropped`) that the Controller/Main Window handles. This enables modular testing.
- **Model/View Separation:** Keep audio data structures (`TrackSegment`) in a dedicated `models.py` separate from the painting logic to allow for headless rendering or CLI automation.
- **Data Robustness:** When adding new analysis fields to a database, always use the `value or 0` (fallback) pattern when loading to ensure compatibility with older tracks that have `NULL` values.
- **Identity-Based Comparison:** When dealing with dictionaries that contain Numpy arrays, never use `==` (which triggers expensive sample-wise broadcasting). Always use `is` or ID checks.
