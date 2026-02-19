import librosa
import numpy as np
import os

class AnalysisModule:
    """Wrapper for librosa to extract musical features from audio files."""
    
    def __init__(self, sample_rate=44100):
        self.target_sr = sample_rate

    def analyze_file(self, file_path):
        """Analyzes a single audio file and returns a dictionary of features."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load audio (mono for analysis)
        y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # 1. BPM Detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        # librosa 0.10+ returns tempo as a scalar or array; handle both
        bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)

        # 2. Harmonic Key Detection (Simplified Chromagram)
        # Using a simple chromagram-based approach for root note estimation
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mean_chroma = np.mean(chroma, axis=1)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_index = np.argmax(mean_chroma)
        harmonic_key = notes[key_index]

        # 3. Energy / Loudness (RMS)
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))
        
        # Note: LUFS calculation usually requires a dedicated library like pyloudnorm,
        # but for Phase 1 we use RMS energy as a proxy.

        return {
            "file_path": os.path.abspath(file_path),
            "filename": os.path.basename(file_path),
            "duration": duration,
            "sample_rate": sr,
            "bpm": round(bpm, 2),
            "harmonic_key": harmonic_key,
            "energy": energy
        }

if __name__ == "__main__":
    # Quick test logic
    import sys
    if len(sys.argv) > 1:
        analyzer = AnalysisModule()
        results = analyzer.analyze_file(sys.argv[1])
        print(results)
