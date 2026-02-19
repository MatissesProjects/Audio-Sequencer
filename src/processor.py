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
        
        # Calculate stretch factor
        stretch_factor = target_bpm / current_bpm
        
        # Pedalboard expects (channels, samples) and float32
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
        y = y.astype(np.float32)
            
        # Corrected signature: time_stretch(input, samplerate, stretch_factor=...)
        y_stretched = pedalboard.time_stretch(y, float(sr), stretch_factor=float(stretch_factor))
        
        # Flatten back for soundfile if mono
        y_out = y_stretched.flatten()
        
        if output_path:
            sf.write(output_path, y_out, sr)
            return output_path
        return y_out

    def shift_pitch(self, input_path, steps, output_path=None):
        """Shifts pitch using librosa."""
        y, sr = librosa.load(input_path, sr=self.sr)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        
        if output_path:
            sf.write(output_path, y_shifted, sr)
            return output_path
        return y_shifted

    def loop_track(self, input_path, target_duration, onsets, output_path=None):
        """
        Extends a track to a target duration by looping it rhythmically.
        onsets: List of beat timestamps.
        """
        y, sr = librosa.load(input_path, sr=self.sr)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration >= target_duration:
            return y[:int(target_duration * sr)]

        # Use the first and last beat onsets as loop points if available
        # to avoid silence/tail issues at the very start/end
        start_sample = int(onsets[0] * sr) if onsets else 0
        end_sample = int(onsets[-1] * sr) if onsets else len(y)
        
        loop_segment = y[start_sample:end_sample]
        num_repeats = int(np.ceil((target_duration - duration) / (len(loop_segment)/sr))) + 1
        
        # Concatenate with a tiny crossfade to prevent clicks
        fade_len = int(sr * 0.05) # 50ms fade
        extended = y[:end_sample]
        
        for _ in range(num_repeats):
            # Simple append for now, can add crossfade logic here
            extended = np.concatenate([extended, loop_segment])
            
        # Trim to exact target
        final = extended[:int(target_duration * sr)]
        
        if output_path:
            sf.write(output_path, final, sr)
            return output_path
        return final
