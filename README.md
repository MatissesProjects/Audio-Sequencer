# AudioSequencer AI: The Flow 🎧✨

AudioSequencer AI is a professional-grade audio sequencing environment that leverages local AI models to automate the technical hurdles of music production. It enables seamless track discovery, tempo matching, and harmonic alignment, allowing creators to focus on the semantic context and "flow" of their sound.

## 🚀 Key Features

*   **Semantic Discovery:** Find audio files based on "vibe," instrument, or emotional quality using local **CLAP** (Contrastive Language-Audio Pretraining) embeddings.
*   **Intelligent Sequencing:** Automatically generate continuous mixes using a multi-dimensional "Compatibility Score" (BPM + Harmonic Key + Semantic Similarity).
*   **Professional DJ Rendering:**
    *   **Dynamic Cross-Ducking:** Progressively swap frequency bands (Bass/Highs) between tracks for seamless transitions.
    *   **Harmonic Sync:** Automatic pitch-shifting using Circle of Fifths logic.
    *   **Rhythmic Looping:** Automatically extend clips using detected beat onsets to ensure perfect transition "tails."
*   **Local-First Privacy:** All AI analysis and audio manipulation run locally on your hardware.
*   **Desktop GUI:** A modern PyQt6 interface for library browsing, real-time recommendations, and one-click mixing.

## 🛠 Tech Stack

*   **Analysis:** `librosa` (BPM, Key, Onsets)
*   **DSP:** `pedalboard` (Spotify), `pyrubberband`
*   **AI:** `laion-clap` (Embeddings), `google-genai` (Orchestration)
*   **Database:** `SQLite` (Metadata), `ChromaDB` (Vectors)
*   **UI:** `PyQt6`

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/AudioSequencer.git
    cd AudioSequencer
    ```

2.  **Set up a virtual environment:**
    ```bash
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
The primary way to use the application is through the desktop interface:
```bash
python src/main.py --gui
```

### CLI Commands
You can also run the core engine directly from the command line:

*   **Scan a directory for audio:**
    ```bash
    python src/main.py --scan "path/to/your/music"
    ```
*   **Generate AI embeddings:**
    ```bash
    python src/main.py --embed
    ```
*   **Generate a full continuous mix:**
    ```bash
    python src/main.py --full-mix
    ```
*   **View library statistics:**
    ```bash
    python src/main.py --stats
    ```

## 🏗 Project Structure

*   `src/analysis.py`: Core MIR (Music Information Retrieval) engine.
*   `src/embeddings.py`: CLAP vector generation.
*   `src/orchestrator.py`: Pathfinding and mix sequencing logic.
*   `src/renderer.py`: Advanced DSP and audio stitching.
*   `src/gui.py`: PyQt6 desktop application.

## 📜 License
This project is licensed under the MIT License.
