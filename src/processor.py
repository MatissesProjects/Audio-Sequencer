import librosa
import soundfile as sf
import os
import pedalboard
import numpy as np

class AudioProcessor:
    """Handles high-quality time-stretching and pitch-shifting using Pedalboard."""
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def stretch_to_bpm(self, input_path, current_bpm, target_bpm, output_path=None):
        """Stretches audio to a target BPM using Pedalboard."""
        y, sr = librosa.load(input_path, sr=self.sr)
        return self.stretch_numpy(y, sr, current_bpm, target_bpm, output_path)

    def stretch_numpy(self, y, sr, current_bpm, target_bpm, output_path=None):
        """Core stretching logic on numpy arrays."""
        stretch_factor = target_bpm / current_bpm
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
        y = y.astype(np.float32)
        y_stretched = pedalboard.time_stretch(y, float(sr), stretch_factor=float(stretch_factor))
        y_out = y_stretched.flatten()
        if output_path:
            sf.write(output_path, y_out, sr)
        return y_out

    def shift_pitch(self, input_path, steps, output_path=None):
        """Shifts pitch using librosa."""
        y, sr = librosa.load(input_path, sr=self.sr)
        return self.shift_pitch_numpy(y, sr, steps, output_path)

    def shift_pitch_numpy(self, y, sr, steps, output_path=None):
        """Core pitch shifting on numpy arrays."""
        if steps == 0: return y
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        if output_path:
            sf.write(output_path, y_shifted, sr)
        return y_shifted

    def loop_track(self, input_path, target_duration, onsets, output_path=None):
        """Rhythmically loops a track to a target duration."""
        y, sr = librosa.load(input_path, sr=self.sr)
        return self.loop_numpy(y, sr, target_duration, onsets, output_path)

    def loop_numpy(self, y, sr, target_duration, onsets, output_path=None):
        """Core looping logic on numpy arrays."""
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration >= target_duration:
            final = y[:int(target_duration * sr)]
        else:
            # Use loop points based on onsets
            start_sample = int(onsets[0] * sr) if onsets else 0
            end_sample = int(onsets[-1] * sr) if onsets else len(y)
            
            loop_segment = y[start_sample:end_sample]
            fade_len = int(sr * 0.5) # 500ms crossfade
            
            extended = y[:end_sample]
            
            # Crossfade curves (Equal Power)
            t = np.linspace(0, 1, fade_len)
            fade_out = np.cos(0.5 * np.pi * t)
            fade_in = np.sin(0.5 * np.pi * t)
            
            current_dur = len(extended) / sr
            while current_dur < target_duration + 2.0:
                overlap_end = extended[-fade_len:]
                overlap_start = loop_segment[:fade_len]
                blended = (overlap_end * fade_out) + (overlap_start * fade_in)
                extended = np.concatenate([extended[:-fade_len], blended, loop_segment[fade_len:]])
                current_dur = len(extended) / sr
            final = extended[:int(target_duration * sr)]
        
        if output_path:
            sf.write(output_path, final, sr)
        return final

    def get_waveform_envelope(self, input_path, num_points=500):
        """Returns a low-res amplitude envelope for waveform display."""
        try:
            y, sr = librosa.load(input_path, sr=22050) # Use lower sr for speed
            # Compute RMSE or just absolute mean in blocks
            hop_length = max(1, len(y) // num_points)
            envelope = []
            for i in range(0, len(y), hop_length):
                chunk = y[i : i + hop_length]
                if len(chunk) > 0:
                    envelope.append(float(np.max(np.abs(chunk))))
            return envelope
        except:
            return []

    def generate_grain_cloud(self, input_path, output_path, duration=10.0, pitch_shift=0):
        """
        Creates an atmospheric textural pad using granular synthesis.
        Perfect for intros, outros, and glue layers.
        """
        y, sr = librosa.load(input_path, sr=self.sr)
        
        # Grain parameters
        grain_size = int(sr * 0.15) # 150ms grains
        overlap = 0.75
        hop = int(grain_size * (1 - overlap))
        num_grains = int((duration * sr) / hop)
        
        output = np.zeros(int(duration * sr) + grain_size)
        
        # Window function (Hanning)
        window = np.hanning(grain_size)
        
        # Create cloud
        for i in range(num_grains):
            # Random start point in source file
            start = np.random.randint(0, len(y) - grain_size)
            grain = y[start : start + grain_size] * window
            
            # Pitch shift grain if needed (slow but effective)
            if pitch_shift != 0:
                grain = librosa.effects.pitch_shift(grain, sr=sr, n_steps=pitch_shift)
                if len(grain) != grain_size: # Handle length change
                    grain = np.resize(grain, grain_size) * window
            
            # Placement
            pos = i * hop
            output[pos : pos + grain_size] += grain
            
        # Normalize
        peak = np.max(np.abs(output))
        if peak > 0: output /= peak
        
        # Apply intense reverb/blur using pedalboard
        board = pedalboard.Pedalboard([
            pedalboard.Reverb(room_size=0.9, wet_level=0.8, dry_level=0.2),
            pedalboard.Delay(delay_seconds=0.5, feedback=0.6, mix=0.4),
            pedalboard.LowpassFilter(cutoff_frequency_hz=3000)
        ])
        
        output = output.astype(np.float32)
        output = board(output, sr)
        
        sf.write(output_path, output, sr)
        return output_path

    def separate_stems(self, input_path, output_dir):
        """
        Extracts basic stems using HPSS and frequency filtering.
        In a full pro environment, this would use Demucs/Spleeter.
        """
        os.makedirs(output_dir, exist_ok=True)
        y, sr = librosa.load(input_path, sr=self.sr)
        
        # 1. Harmonic / Percussive Separation (Librosa)
        harmonic, percussive = librosa.effects.hpss(y)
        
        # 2. Pseudo-Vocal extraction (Bandpass 300Hz - 3kHz on Harmonic)
        # This is a very rough approximation
        board = pedalboard.Pedalboard([
            pedalboard.HighpassFilter(cutoff_frequency_hz=300),
            pedalboard.LowpassFilter(cutoff_frequency_hz=3000)
        ])
        vocal_candidate = board(harmonic.astype(np.float32), sr)
        
        # Subtract vocal from harmonic to get "instruments"
        instr_harmonic = harmonic - vocal_candidate
        
        stems = {
            "drums": percussive,
            "vocals": vocal_candidate,
            "other": instr_harmonic
        }
        
        paths = {}
        for name, data in stems.items():
            path = os.path.join(output_dir, f"{name}.wav")
            sf.write(path, data, sr)
            paths[name] = path
            
        return paths

