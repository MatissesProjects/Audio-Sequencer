import librosa
import numpy as np
import os
from typing import List, Dict, Optional, Any, Union, Tuple

class AnalysisModule:
    """Wrapper for librosa to extract musical features from audio files."""
    
    def __init__(self, sample_rate: int = 44100):
        self.target_sr: int = sample_rate

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyzes a single audio file and returns a dictionary of features."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm: float = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
        
        beat_times: np.ndarray = librosa.frames_to_time(beat_frames, sr=sr)
        onsets_json: str = ",".join([str(round(t, 4)) for t in beat_times])
        onset_density: float = len(beat_times) / duration if duration > 0 else 0.0

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mean_chroma = np.mean(chroma, axis=1)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        harmonic_key: str = notes[np.argmax(mean_chroma)]

        rms = librosa.feature.rms(y=y)
        energy: float = float(np.mean(rms))
        
        loop_start, loop_dur = self.detect_best_loop(y, sr, beat_times, bars=4)
        vocal_energy: float = self.detect_vocal_prominence(y, sr)

        sections: List[Dict[str, Any]] = []
        try:
            from src.core.config import AppConfig
            import requests
            with open(file_path, 'rb') as f:
                r = requests.post(AppConfig.REMOTE_SECTIONS_URL, files={'file': f}, timeout=15)
            if r.status_code == 200:
                sections = r.json().get("sections", [])
        except: pass

        return {
            "file_path": os.path.abspath(file_path),
            "filename": os.path.basename(file_path),
            "duration": duration,
            "sample_rate": sr,
            "bpm": round(bpm, 2),
            "onsets_json": onsets_json,
            "harmonic_key": harmonic_key,
            "energy": energy,
            "vocal_energy": vocal_energy,
            "onset_density": onset_density,
            "loop_start": loop_start,
            "loop_duration": loop_dur,
            "sections": sections
        }

    def detect_vocal_prominence(self, y: np.ndarray, sr: int) -> float:
        """Estimates how prominent vocals/mid-range leads are."""
        S = np.abs(librosa.stft(y)); freqs = librosa.fft_frequencies(sr=sr)
        vocal_mask = (freqs >= 300) & (freqs <= 3000)
        v_energy = np.mean(S[vocal_mask, :]); total_energy = np.mean(S)
        return float(v_energy / (total_energy + 1e-9))

    def detect_best_loop(self, y: np.ndarray, sr: int, beat_times: np.ndarray, bars: int = 4) -> Tuple[float, float]:
        """Finds the most energetic and rhythmic 'n' bar section."""
        if len(beat_times) < 4 * bars: return 0.0, librosa.get_duration(y=y, sr=sr)
        beats_per_loop = 4 * bars; max_energy = -1.0; best_start = 0.0; best_end = 0.0
        for i in range(0, len(beat_times) - beats_per_loop, 4):
            start_t, end_t = beat_times[i], beat_times[i + beats_per_loop]
            segment = y[int(start_t * sr):int(end_t * sr)]
            if len(segment) == 0: continue
            seg_rms = np.mean(librosa.feature.rms(y=segment))
            if seg_rms > max_energy:
                max_energy = float(seg_rms); best_start = float(start_t); best_end = float(end_t)
        if max_energy == -1.0: return 0.0, librosa.get_duration(y=y, sr=sr)
        return best_start, (best_end - best_start)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyzer = AnalysisModule()
        print(analyzer.analyze_file(sys.argv[1]))
