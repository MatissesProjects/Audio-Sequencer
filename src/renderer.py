from pydub import AudioSegment, effects
import os
import numpy as np
import pedalboard
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Limiter, Compressor
from tqdm import tqdm

class FlowRenderer:
    """Handles mixing, layering, and crossfading multiple tracks with creative modulation."""
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def segment_to_numpy(self, seg):
        """Helper to convert pydub segment to numpy float32."""
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        samples /= (1 << (8 * seg.sample_width - 1))
        if seg.channels == 2:
            samples = samples.reshape((-1, 2)).T
        else:
            samples = samples.reshape((1, -1))
        return samples

    def numpy_to_segment(self, samples, sr):
        """Helper to convert numpy float32 back to pydub segment."""
        peak = np.max(np.abs(samples))
        if peak > 1.0:
            samples /= peak
            
        samples = (samples * 32767).astype(np.int16)
        if samples.shape[0] == 2:
            samples = samples.T.flatten()
            return AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=2)
        else:
            return AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    def apply_dynamic_ducking(self, samples, curve, filter_type='highpass'):
        """
        Applies a changing filter/volume curve over time.
        curve: numpy array of values from 0.0 to 1.0.
        """
        num_samples = samples.shape[1]
        chunk_size = 4410 # 100ms
        processed = np.zeros_like(samples)
        
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            intensity = curve[start]
            chunk = samples[:, start:end]
            
            if filter_type == 'highpass':
                cutoff = 20 + (intensity * 400)
                board = Pedalboard([HighpassFilter(cutoff_frequency_hz=cutoff)])
            else:
                cutoff = 20000 - (intensity * 15000)
                board = Pedalboard([LowpassFilter(cutoff_frequency_hz=cutoff)])
                
            processed[:, start:end] = board(chunk, self.sr)
        return processed

    def dj_stitch(self, track_paths, output_path, overlay_ms=12000):
        """
        Creates a 'Professional DJ Mix' with dynamic ducking automation.
        """
        if not track_paths:
            return None
            
        combined = AudioSegment.from_file(track_paths[0])
        combined = combined.set_frame_rate(self.sr).set_channels(2)
        combined = effects.normalize(combined, headroom=0.5)
        
        for next_track_path in tqdm(track_paths[1:], desc="Professional Stitching"):
            next_seg = AudioSegment.from_file(next_track_path)
            next_seg = next_seg.set_frame_rate(self.sr).set_channels(2)
            next_seg = effects.normalize(next_seg, headroom=0.5)
            
            # 1. Segments for transition
            out_seg = combined[-overlay_ms:]
            in_seg = next_seg[:overlay_ms]
            
            out_np = self.segment_to_numpy(out_seg)
            in_np = self.segment_to_numpy(in_seg)
            
            # 2. Dynamic Curves (Automation)
            out_curve = np.linspace(0, 1, out_np.shape[1])
            in_curve = np.linspace(1, 0, in_np.shape[1])
            
            # Outgoing: High-pass (remove bass)
            out_filtered = self.apply_dynamic_ducking(out_np, out_curve, 'highpass')
            # Incoming: Low-pass (keep only bass then fade in highs)
            in_filtered = self.apply_dynamic_ducking(in_np, in_curve, 'lowpass')
            
            # 3. Sum and Master Bus
            summed_np = (out_filtered * (1.0 - out_curve)) + (in_filtered * (1.0 - in_curve))
            master_bus = Pedalboard([
                Compressor(threshold_db=-10, ratio=2.0),
                Limiter(threshold_db=-0.1)
            ])
            transition_np = master_bus(summed_np, self.sr)
            
            transition_seg = self.numpy_to_segment(transition_np, self.sr)
            combined = combined[:-overlay_ms] + transition_seg + next_seg[overlay_ms:]
            
        combined = effects.normalize(combined, headroom=0.1)
        combined.export(output_path, format="mp3", bitrate="320k")
        return output_path

if __name__ == "__main__":
    # Test mixing the original and the stretched version
    renderer = FlowRenderer()
    # We will use this in preview_mix.py
