Audio Sequencer & AI Recommendation Tool
This document outlines the architecture and development roadmap for a purpose-built application designed to analyze, stitch, and layer 30-second audio loops into a constant flow. It automates tempo and harmonic matching, while introducing an AI-driven recommendation engine for track discovery and sequencing.

1. System Architecture & Tech Stack
To build a robust, highly performant application, we will use a modular Python backend coupled with a modern frontend.

Audio Analysis: librosa (for BPM, key estimation, and transient detection).

Audio Manipulation: pedalboard (by Spotify) or pyrubberband (for high-quality time-stretching and pitch-shifting without artifacts).

Audio Processing & Export: pydub (for crossfading, splicing, and volume normalization).

AI Embedding & Recommendation: laion-clap (Contrastive Language-Audio Pretraining) running locally to ensure privacy and offline capability.

Database (Metadata & Vectors): ChromaDB or FAISS (for storing and querying high-dimensional audio embeddings), alongside SQLite for standard file metadata.

Frontend GUI: PyQt6 (for a native, responsive desktop app) or a FastAPI backend with a React frontend.

2. Phase 1: The Analysis Engine (Core MIR)
Before the app can sequence anything, it needs to extract the exact mathematical properties of your library.

Ingestion Pipeline:

Watch a designated local directory for new .wav, .mp3, or .flac files.

Extract standard file metadata (filename, duration, sample rate).

Musical Feature Extraction (librosa):

Tempo (BPM): Calculate the exact beats per minute.

Harmonic Key: Run a chromagram analysis to estimate the root note and scale (e.g., C Minor).

Energy/Loudness: Calculate LUFS (Loudness Units relative to Full Scale) to ensure quiet ambient tracks aren't layered directly over aggressive, heavily compressed beats without proper gain staging.

Local Database Synchronization: * Store these features locally to prevent re-analyzing the same 30-second clips every time the application boots.

3. Phase 2: The AI Recommendation System
This phase moves beyond raw math and allows the application to understand the acoustic and semantic context of your audio files.

Audio Embeddings (CLAP):

Implement the open-source CLAP model.

When a track is ingested, CLAP converts the 30-second audio into a dense 512-dimensional vector (an embedding) that represents its acoustic characteristics.

Vector Search (ChromaDB):

Store these embeddings in the local vector database.

Audio-to-Audio Search: When you drop a beat into the timeline, the app queries the vector database using cosine similarity to recommend other beats in your library that share the exact same acoustic "vibe" or recording style.

Text-to-Audio Semantic Search: Because CLAP aligns audio and text in a shared latent space, you can include a search bar in the UI. You can type "heavy distorted synth bass" or "chill ambient piano", and it will filter your local clips accordingly, eliminating the need for manual tagging.

4. Phase 3: Sequencing & Layering Logic
With the metadata and AI embeddings ready, the engine automates the construction of the mix.

Compatibility Scoring:

The engine evaluates potential track pairings based on a weighted score of BPM Proximity (within 5%), Harmonic Compatibility (using Camelot Wheel logic—e.g., matching a track in 8A with one in 8B or 9A), and AI Vector Similarity.

Horizontal Stitching (Transitions):

Automatically line up compatible tracks end-to-end.

Apply beat-matched crossfades. If Track A is 120 BPM and Track B is 122 BPM, the engine slightly time-stretches Track B to 120 BPM during the transition window to prevent the kick drums from clashing.

Vertical Layering (Stems/Overlays):

If two beats share the exact same BPM and compatible keys, the engine can stack them vertically.

Implement automatic EQ ducking (e.g., if both tracks have heavy low-end frequencies, apply a high-pass filter to one so the mix doesn't become muddy).

5. Phase 4: User Interface & Workflow
The UI must abstract the complex MIR and vector operations into a clean, creative workspace.

Library Browser: Displays all ingested loops, filterable by BPM, Key, Energy, and text-based semantic search.

The "Flow" Canvas:

A simplified, node-based or linear timeline where you drop a starting track.

Smart Suggestions Sidebar: As soon as a track is placed on the canvas, this sidebar dynamically populates with the top 5 AI-recommended subsequent tracks, highlighting why they fit (e.g., "Matched Key: C Minor", "High Semantic Similarity").

Export Module: Render the final sequenced arrangement as a single continuous audio file.

6. Phase 5: Advanced AI Expansion (Optional)
Once the core application is stable, it can be expanded to generate new assets to fill in gaps.

Local Audio Generation: Integrate open-source generation models like AudioCraft (MusicGen). If you have two beats that are structurally incompatible but you want to force a transition, you can prompt the local model to generate a custom 4-bar riser or transition sweep in the exact BPM and Key needed to bridge the gap seamlessly.


---

Next steps

1. Unblock Phase 5: True Generative AI Transitions
Currently, your TransitionGenerator in src/generator.py relies on Gemini to orchestrate a procedural noise sweep using white/pink noise and filters. You can unblock the ai-gen track by fully integrating audiocraft (MusicGen) to run locally on your RTX 4090. Instead of just noise sweeps, you can prompt MusicGen to synthesize custom 4-bar instrumental risers in the exact BPM and key needed to seamlessly bridge two structurally incompatible tracks.

2. Stream Interaction Engine
You can make the sequencer highly interactive by connecting it directly to your Twitch and YouTube streams. You could build a listener module that reads chat messages and feeds them into the EmbeddingEngine. If chat types "give me dark heavy bass" or "switch to chill piano," the app can convert that text into a 512-dimensional CLAP embedding and dynamically inject the most relevant track from your library into the FullMixOrchestrator's queue. Viewers could also use channel point redemptions to trigger the "Hyper-Mix" drops.

3. Deep Learning Stem Separation
In src/processor.py, your separate_stems method currently uses basic Harmonic/Percussive separation (HPSS) and frequency bandpassing. Upgrading this to use a deep learning model like Demucs or Spleeter would drastically improve the audio quality of isolated vocals and drums. This cleaner separation would allow your dynamic sidechain ducking and spectral masking to sound truly professional.

4. Stochastic Production Edits (Auto-Chopping)
To fix the issue of purely deterministic arrangements feeling repetitive, you can program genuine tension-building mechanics into your AudioProcessor. Implement a method that takes a 4-bar loop and automatically applies "micro-chopping"—slicing it into 2-bar, then 1-bar, then 1/4-beat stutters. By pairing this automated stutter with a rising pitch shift and a sweeping high-pass filter, the engine can autonomously generate massive EDM-style build-ups right before a drop.

5. "Groove Lock" Smart Syncing
While you are currently calculating the onset_density (beats per second) and using it in your scoring logic, you can utilize the actual onsets_json timestamps stored in the database to align the transients of different tracks. By slightly warping a background track so its transients lock perfectly onto the kick drum onsets of your foundation track, you eliminate rhythmic clashing entirely.