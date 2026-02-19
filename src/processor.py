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
