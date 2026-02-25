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
            
            # Ensure fade_len is never larger than the segment itself
            fade_len = int(sr * 0.5) # 500ms crossfade
            if fade_len > len(loop_segment) // 2:
                fade_len = len(loop_segment) // 2
            
            # If still too small (extremely short clip), use 0 fade
            if fade_len < 10: fade_len = 0
            
            extended = y[:end_sample]
            
            if fade_len > 0:
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
            else:
                # Simple concatenation fallback for tiny clips
                while (len(extended)/sr) < target_duration + 2.0:
                    extended = np.concatenate([extended, loop_segment])
            
            final = extended[:int(target_duration * sr)]
        
        if output_path:
            sf.write(output_path, final, sr)
        return final

    def apply_rhythmic_gate(self, y, sr, bpm, pattern="1/8"):
        """
        Applies a rhythmic volume gate (stutter effect) to a numpy array.
        pattern: "1/4", "1/8", "1/16", or "triplet"
        """
        if len(y.shape) > 1:
            # Handle stereo by applying to both channels
            channels = []
            for i in range(y.shape[0]):
                channels.append(self.apply_rhythmic_gate(y[i], sr, bpm, pattern))
            return np.stack(channels)

        # Calculate durations in samples
        beat_dur = (60.0 / bpm) * sr
        if pattern == "1/4": division = 1.0
        elif pattern == "1/8": division = 0.5
        elif pattern == "1/16": division = 0.25
        elif pattern == "triplet": division = 1.0/3.0
        else: division = 0.5
        
        gate_dur = int(beat_dur * division)
        # Create a square wave for the gate
        num_gates = int(len(y) / gate_dur) + 1
        # Gate: 70% on, 30% off for a crisp but not too harsh feel
        on_samples = int(gate_dur * 0.7)
        off_samples = gate_dur - on_samples
        
        gate_cycle = np.concatenate([np.ones(on_samples), np.zeros(off_samples)])
        full_gate = np.tile(gate_cycle, num_gates)[:len(y)]
        
        # Smooth the gate edges to prevent clicks
        fade_samples = int(sr * 0.01) # 10ms
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        # Very simple smoothing by applying short fades to the gate signal itself
        # This is a bit rough but works for a quick stutter
        return y * full_gate.astype(np.float32)

    def get_waveform_envelope(self, input_path, num_points=500):
        """Returns a low-res amplitude envelope for waveform display."""
        try:
            y, sr = librosa.load(input_path, sr=22050) # Use lower sr for speed
            # Compute RMSE or just absolute mean in blocks
            hop_length = max(1, len(y) // num_points)
            envelope = []
            for i in range(0, len(y), hop_length):
                chunk = y[i : i + hop_length]
                if chunk.size > 0:
                    envelope.append(float(np.max(np.abs(chunk))))
                else:
                    envelope.append(0.0)
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
        Extracts stems using high-quality remote Demucs OR local HPSS fallback.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        from src.core.config import AppConfig
        remote_url = AppConfig.REMOTE_SEP_URL
        
        # 1. Try Remote Separation (High Quality)
        try:
            import requests
            import zipfile
            import io
            
            print(f"Requesting Remote Stem Separation for: {os.path.basename(input_path)}...")
            with open(input_path, 'rb') as f:
                response = requests.post(
                    remote_url, 
                    files={'file': f},
                    timeout=120 # 2 minute timeout for separation
                )
            
            if response.status_code == 200:
                print("Remote separation success. Extracting stems...")
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    zf.extractall(output_dir)
                
                # Verify we got the standard 4 stems
                paths = {}
                for s in ["vocals", "drums", "bass", "other"]:
                    p = os.path.join(output_dir, f"{s}.wav")
                    if os.path.exists(p): paths[s] = p
                
                if paths: return paths
            else:
                print(f"Remote separation failed (HTTP {response.status_code}). Using local fallback.")
        except Exception as e:
            print(f"Remote separation error: {e}. Using local fallback.")

        # 2. Local Fallback (Basic HPSS)
        print("Performing local basic stem separation (HPSS)...")
        y, sr = librosa.load(input_path, sr=self.sr)
        
        harmonic, percussive = librosa.effects.hpss(y)
        
        # Pseudo-Vocal extraction (Bandpass 300Hz - 3kHz on Harmonic)
        import pedalboard
        board = pedalboard.Pedalboard([
            pedalboard.HighpassFilter(cutoff_frequency_hz=300),
            pedalboard.LowpassFilter(cutoff_frequency_hz=3000)
        ])
        vocal_candidate = board(harmonic.astype(np.float32), sr)
        instr_harmonic = harmonic - vocal_candidate
        
        stems = {
            "drums": percussive,
            "vocals": vocal_candidate,
            "other": instr_harmonic,
            "bass": np.zeros_like(percussive) # HPSS doesn't isolate bass well
        }
        
        paths = {}
        for name, data in stems.items():
            path = os.path.join(output_dir, f"{name}.wav")
            sf.write(path, data, sr)
            paths[name] = path
            
        return paths

    def calculate_sidechain_keyframes(self, source_path, duration_ms, depth=0.8, sensitivity=0.1):
        """
        Analyzes audio energy and returns a list of (ms, volume) keyframes.
        Perfect for visible, editable sidechaining.
        """
        try:
            y, sr = librosa.load(source_path, sr=22050, duration=duration_ms/1000.0)
            
            # Use RMS energy to find peaks
            hop_length = 512
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
            
            # Normalize RMS
            if rms.size > 0:
                max_val = np.max(rms)
                if max_val > 0:
                    rms = rms / max_val
            
            keyframes = []
            # We add points where energy crosses the sensitivity threshold
            # To keep keyframe count reasonable, we sample every ~50ms
            for i in range(0, len(rms), 4): # Every 4 frames at 22k sr / 512 hop is ~100ms
                t_ms = times[i] * 1000.0
                energy = rms[i]
                
                # Invert energy for ducking (High energy = Low volume)
                # target_vol = 1.0 - (energy * depth)
                val = 1.0 - (min(1.0, energy / sensitivity) * depth)
                keyframes.append((t_ms, float(val)))
                
            return keyframes
        except Exception as e:
            print(f"Error calculating sidechain: {e}")
            return []

