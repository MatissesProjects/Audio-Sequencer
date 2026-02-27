import librosa
import soundfile as sf
import os
import pedalboard
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple

class AudioProcessor:
    """Handles high-quality time-stretching and pitch-shifting using Pedalboard."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sr: int = sample_rate

    def stretch_to_bpm(self, input_path: str, current_bpm: float, target_bpm: float, output_path: Optional[str] = None) -> np.ndarray:
        """Stretches audio to a target BPM using Pedalboard."""
        y, sr = librosa.load(input_path, sr=self.sr)
        return self.stretch_numpy(y, sr, current_bpm, target_bpm, output_path)

    def stretch_numpy(self, y: np.ndarray, sr: int, current_bpm: float, target_bpm: float, output_path: Optional[str] = None) -> np.ndarray:
        """Core stretching logic on numpy arrays."""
        stretch_factor = target_bpm / current_bpm
        if len(y.shape) == 1:
            y_in = y.reshape(1, -1)
        else:
            y_in = y
        y_in = y_in.astype(np.float32)
        y_stretched = pedalboard.time_stretch(y_in, float(sr), stretch_factor=float(stretch_factor))
        y_out = y_stretched.flatten()
        if output_path:
            sf.write(output_path, y_out, sr)
        return y_out

    def shift_pitch(self, input_path: str, steps: float, output_path: Optional[str] = None) -> np.ndarray:
        """Shifts pitch using librosa."""
        y, sr = librosa.load(input_path, sr=self.sr)
        return self.shift_pitch_numpy(y, sr, steps, output_path)

    def shift_pitch_numpy(self, y: np.ndarray, sr: int, steps: float, output_path: Optional[str] = None) -> np.ndarray:
        """Core pitch shifting on numpy arrays."""
        if steps == 0: return y
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        if output_path:
            sf.write(output_path, y_shifted, sr)
        return y_shifted

    def loop_track(self, input_path: str, target_duration: float, onsets: List[float], output_path: Optional[str] = None) -> np.ndarray:
        """Rhythmically loops a track to a target duration."""
        y, sr = librosa.load(input_path, sr=self.sr)
        return self.loop_numpy(y, sr, target_duration, onsets, output_path)

    def loop_numpy(self, y: np.ndarray, sr: int, target_duration: float, onsets: List[float], output_path: Optional[str] = None) -> np.ndarray:
        """Core looping logic on numpy arrays."""
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration >= target_duration:
            final = y[:int(target_duration * sr)]
        else:
            start_sample = int(onsets[0] * sr) if onsets else 0
            end_sample = int(onsets[-1] * sr) if onsets else len(y)
            loop_segment = y[start_sample:end_sample]
            fade_len = int(sr * 0.5) 
            if fade_len > len(loop_segment) // 2: fade_len = len(loop_segment) // 2
            if fade_len < 10: fade_len = 0
            extended = y[:end_sample]
            if fade_len > 0:
                t = np.linspace(0, 1, fade_len)
                fade_out = np.cos(0.5 * np.pi * t); fade_in = np.sin(0.5 * np.pi * t)
                current_dur = len(extended) / sr
                while current_dur < target_duration + 2.0:
                    overlap_end = extended[-fade_len:]; overlap_start = loop_segment[:fade_len]
                    blended = (overlap_end * fade_out) + (overlap_start * fade_in)
                    extended = np.concatenate([extended[:-fade_len], blended, loop_segment[fade_len:]])
                    current_dur = len(extended) / sr
            else:
                while (len(extended)/sr) < target_duration + 2.0:
                    extended = np.concatenate([extended, loop_segment])
            final = extended[:int(target_duration * sr)]
        if output_path: sf.write(output_path, final, sr)
        return final

    def apply_rhythmic_gate(self, y: np.ndarray, sr: int, bpm: float, pattern: str = "1/8") -> np.ndarray:
        """Applies a rhythmic volume gate (stutter effect) to a numpy array."""
        if len(y.shape) > 1:
            channels = []
            for i in range(y.shape[0]):
                channels.append(self.apply_rhythmic_gate(y[i], sr, bpm, pattern))
            return np.stack(channels)
        beat_dur = (60.0 / bpm) * sr
        if pattern == "1/4": division = 1.0
        elif pattern == "1/8": division = 0.5
        elif pattern == "1/16": division = 0.25
        elif pattern == "triplet": division = 1.0/3.0
        else: division = 0.5
        gate_dur = int(beat_dur * division)
        num_gates = int(len(y) / gate_dur) + 1
        on_samples = int(gate_dur * 0.7)
        off_samples = gate_dur - on_samples
        gate_cycle = np.concatenate([np.ones(on_samples), np.zeros(off_samples)])
        full_gate = np.tile(gate_cycle, num_gates)[:len(y)]
        return y * full_gate.astype(np.float32)

    def get_waveform_envelope(self, input_path: str, num_points: int = 500) -> List[float]:
        """Returns a low-res amplitude envelope for waveform display."""
        try:
            y, sr = librosa.load(input_path, sr=22050)
            hop_length = max(1, len(y) // num_points)
            envelope = []
            for i in range(0, len(y), hop_length):
                chunk = y[i : i + hop_length]
                if chunk.size > 0: envelope.append(float(np.max(np.abs(chunk))))
                else: envelope.append(0.0)
            return envelope
        except: return []

    def generate_grain_cloud(self, input_path: str, output_path: str, duration: float = 10.0, pitch_shift: int = 0) -> str:
        """Creates an atmospheric textural pad using granular synthesis."""
        y, sr = librosa.load(input_path, sr=self.sr)
        grain_size = int(sr * 0.15) 
        overlap = 0.75; hop = int(grain_size * (1 - overlap)); num_grains = int((duration * sr) / hop)
        output = np.zeros(int(duration * sr) + grain_size)
        window = np.hanning(grain_size)
        for i in range(num_grains):
            start = np.random.randint(0, len(y) - grain_size)
            grain = y[start : start + grain_size] * window
            if pitch_shift != 0:
                grain = librosa.effects.pitch_shift(grain, sr=sr, n_steps=float(pitch_shift))
                if len(grain) != grain_size: grain = np.resize(grain, grain_size) * window
            pos = i * hop; output[pos : pos + grain_size] += grain
        peak = np.max(np.abs(output))
        if peak > 0: output /= peak
        board = pedalboard.Pedalboard([
            pedalboard.Reverb(room_size=0.9, wet_level=0.8, dry_level=0.2),
            pedalboard.Delay(delay_seconds=0.5, feedback=0.6, mix=0.4),
            pedalboard.LowpassFilter(cutoff_frequency_hz=3000)
        ])
        output_float = output.astype(np.float32); output_p = board(output_float, sr)
        sf.write(output_path, output_p, sr)
        return output_path

    def generate_spectral_pad_remote(self, source_path: str, output_path: str, duration: float = 20.0) -> str:
        """Calls remote 4090 for high-fidelity spectral blurring."""
        from src.core.config import AppConfig
        url = AppConfig.REMOTE_PAD_URL
        try:
            import requests
            with open(source_path, 'rb') as f:
                response = requests.post(url, files={'file': f}, data={'duration': duration}, timeout=60)
            if response.status_code == 200:
                with open(output_path, 'wb') as f: f.write(response.content)
                return output_path
        except: pass
        return self.generate_grain_cloud(source_path, output_path, duration=duration)

    def separate_stems(self, input_path: str, output_dir: str) -> Dict[str, str]:
        """Extracts stems using high-quality remote Demucs OR local HPSS fallback."""
        os.makedirs(output_dir, exist_ok=True)
        from src.core.config import AppConfig
        remote_url = AppConfig.REMOTE_SEP_URL
        try:
            import requests; import zipfile; import io
            with open(input_path, 'rb') as f:
                response = requests.post(remote_url, files={'file': f}, timeout=120)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf: zf.extractall(output_dir)
                paths = {}
                for s in ["vocals", "drums", "bass", "other"]:
                    p = os.path.join(output_dir, f"{s}.wav")
                    if os.path.exists(p): paths[s] = p
                if paths: return paths
        except: pass
        y, sr = librosa.load(input_path, sr=self.sr)
        harmonic, percussive = librosa.effects.hpss(y)
        import pedalboard
        board = pedalboard.Pedalboard([pedalboard.HighpassFilter(cutoff_frequency_hz=300), pedalboard.LowpassFilter(cutoff_frequency_hz=3000)])
        vocal_candidate = board(harmonic.astype(np.float32), sr); instr_harmonic = harmonic - vocal_candidate
        stems = {"drums": percussive, "vocals": vocal_candidate, "other": instr_harmonic, "bass": np.zeros_like(percussive)}
        paths = {}
        for name, data in stems.items():
            path = os.path.join(output_dir, f"{name}.wav"); sf.write(path, data, sr); paths[name] = path
        return paths

    def calculate_sidechain_keyframes(self, source_path: str, duration_ms: float, depth: float = 0.8, sensitivity: float = 0.1) -> List[Tuple[float, float]]:
        """Analyzes audio energy and returns a list of (ms, volume) keyframes."""
        try:
            y, sr = librosa.load(source_path, sr=22050, duration=duration_ms/1000.0)
            hop_length = 512
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
            if rms.size > 0:
                max_val = np.max(rms)
                if max_val > 0: rms = rms / max_val
            keyframes = []
            for i in range(0, len(rms), 4):
                t_ms = times[i] * 1000.0; energy = rms[i]
                val = 1.0 - (min(1.0, energy / sensitivity) * depth)
                val = max(0.15, val)
                keyframes.append((t_ms, float(val)))
            return keyframes
        except: return []

    def generate_gender_swap_remote(self, source_path: str, output_path: str, target: str = "female", steps: float = 0) -> Optional[str]:
        """Calls remote 4090 for gender transformation (formant shifting)."""
        from src.core.config import AppConfig
        url = AppConfig.REMOTE_GENDER_URL
        try:
            import requests
            with open(source_path, 'rb') as f:
                response = requests.post(url, files={'file': f}, data={'target': target, 'steps': steps}, timeout=60)
            if response.status_code == 200:
                with open(output_path, 'wb') as f: f.write(response.content)
                return output_path
        except: pass
        return None
