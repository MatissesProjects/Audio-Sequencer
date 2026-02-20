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
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        # librosa 0.10+ returns tempo as a scalar or array; handle both
        bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
        
        # 1b. Beat Onsets (for seamless looping/alignment)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        onsets_json = ",".join([str(round(t, 4)) for t in beat_times])
        
        # 1c. Onset Density (Activity)
        onset_density = len(beat_times) / duration if duration > 0 else 0

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
        
        # 4. Best Loop Detection (Smart Clip Extraction)
        # Find 16-beat window with stable energy
        loop_start, loop_dur = self.detect_best_loop(y, sr, beat_times, bars=4)

        return {
            "file_path": os.path.abspath(file_path),
            "filename": os.path.basename(file_path),
            "duration": duration,
            "sample_rate": sr,
            "bpm": round(bpm, 2),
            "onsets_json": onsets_json,
            "harmonic_key": harmonic_key,
            "energy": energy,
            "onset_density": onset_density,
            "loop_start": loop_start,
            "loop_duration": loop_dur
        }

    def detect_best_loop(self, y, sr, beat_times, bars=4):
        """Finds the most energetic and rhythmic 'n' bar section."""
        if len(beat_times) < 4 * bars: # Assuming 4/4
            return 0.0, librosa.get_duration(y=y, sr=sr)
            
        # Estimate beats per bar = 4
        beats_per_loop = 4 * bars
        max_energy = -1
        best_start = 0
        best_end = 0
        
        # Slide a window of 'beats_per_loop'
        # We check every downbeat (every 4th beat)
        for i in range(0, len(beat_times) - beats_per_loop, 4):
            start_t = beat_times[i]
            end_t = beat_times[i + beats_per_loop]
            
            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            
            segment = y[start_sample:end_sample]
            if len(segment) == 0: continue
            
            seg_rms = np.mean(librosa.feature.rms(y=segment))
            
            # Simple heuristic: Loudest 4 bars is usually the 'drop' or main loop
            if seg_rms > max_energy:
                max_energy = seg_rms
                best_start = start_t
                best_end = end_t
                
        if max_energy == -1: # Fallback
            return 0.0, librosa.get_duration(y=y, sr=sr)
            
        return best_start, (best_end - best_start)

if __name__ == "__main__":
    # Quick test logic
    import sys
    if len(sys.argv) > 1:
        analyzer = AnalysisModule()
        results = analyzer.analyze_file(sys.argv[1])
        print(results)
