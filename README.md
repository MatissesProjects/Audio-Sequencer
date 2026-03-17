# AudioSequencer AI: The Pro Flow 🎧✨

AudioSequencer AI is a professional-grade visual arrangement environment that leverages local and remote AI models to automate the technical hurdles of music production. It enables seamless track discovery, tempo matching, and harmonic alignment, allowing creators to focus on the semantic context and "journey" of their sound.

## 🚀 Key Features

*   **🎹 Multi-Lane Visual Timeline:** Arrange and layer parallel audio lanes with high-resolution waveforms and detected beat markers.
*   **🪄 AI Arrangement Intelligence:**
    *   **Semantic Vibe Search:** Find tracks by describing a sound (e.g., "dark heavy bass") using local **CLAP** embeddings.
    *   **Smart Bridge Search:** Instantly find the best musical "glue" to connect two existing segments.
    *   **Auto-Orchestration:** Generate complex, multi-lane layered journeys from a single seed track.
*   **✨ AI Generative Transitions:** Procedural risers and sweeps, orchestrated by **Gemini** to match the unique vibe of your arrangement.
*   **🔪 AI Stem Separation:** Deep-learning based isolation of vocals, drums, bass, and instruments (requires remote 4090 server for Pro Scan).
*   **📏 Professional Precision Tools:** 
    *   **Beat-Grid & Bar Snapping:** Align loops perfectly to the project's rhythm.
    *   **Visual Slip Tool:** `Alt + Drag` to slide audio within a clip for frame-perfect sync.
    *   **Interactive Volume Envelopes:** Adjust clip gain and view lead-focus ducking in real-time.
    *   **Undo/Redo System:** Full state-based history for all arrangement actions.
*   **🔊 Pro-Grade DSP Engine:** 
    *   **Master Bus Chain:** Integrated Compression and Limiting for a polished, radio-ready mix.
    *   **S-Curve Fades:** Sinusoidal crossfades for natural transitions.
*   **📦 DAW-Ready Export:** High-fidelity Master Mixdown or individual Stem export for all lanes.

## 🛠 Tech Stack

*   **Analysis:** `librosa` (BPM, Key, Onsets)
*   **DSP:** `pedalboard` (Spotify), `pyrubberband`
*   **AI:** `laion-clap` (Local Embeddings), `google-genai` (Orchestration), `Demucs` (via remote server)
*   **Database:** `SQLite` (Metadata), `ChromaDB` (Vectors)
*   **UI:** `PyQt6`

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/AudioSequencer.git
    cd AudioSequencer
    ```

2.  **Set up a virtual environment:**
    ```powershell
    python -m venv .venv
    # Windows:
    .venv\Scripts\Activate.ps1
    # Linux/Mac:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    Create a `.env` file in the root directory and add your Google Gemini API key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

## 🎮 Usage

### Launch the GUI
The main interface can be launched directly from the root:
```powershell
python src/main.py
```

### Remote AI Server (Optional)
For advanced features like Stem Separation and Pro Scan, run the remote server on a machine with a powerful GPU:
```powershell
cd remote_server
python server.py
```

### Pro Navigation Tips:
*   **`Ctrl + Mouse Wheel`**: Zoom timeline (Horizontal).
*   **`Space`**: Play/Pause journey.
*   **`Alt + Drag`**: Slip audio inside a clip (change start/end without moving position).
*   **`Shift + Drag`**: Adjust clip volume.
*   **`Delete / Backspace`**: Remove selected clip.
*   **`M / S`**: Mute/Solo selected lane.
*   **`Ctrl + B`**: Blade Tool (Split selected clip at cursor).

## 🏗 Project Structure

*   `src/main.py`: Application entry point.
*   `src/ui/`: All PyQt6 interface components.
    *   `main_window.py`: Primary application logic and layout.
    *   `widgets.py`: Custom UI components (Timeline, Waveforms).
*   `src/core/`: Foundation logic.
    *   `models.py`: Data structures for segments and metadata.
    *   `config.py`: Global paths and audio settings.
*   `src/renderer.py`: Pro DSP engine and Master Bus.
*   `src/orchestrator.py`: AI pathfinding and layered sequencing.
*   `src/database.py`: Hybrid SQLite/ChromaDB manager.

## 📜 License
This project is licensed under the MIT License.
