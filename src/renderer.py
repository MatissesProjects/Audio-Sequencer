from pydub import AudioSegment, effects
import os
import numpy as np
import pedalboard
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Limiter
from tqdm import tqdm

class FlowRenderer:
    """Handles mixing, layering, and crossfading multiple tracks with pro gain staging."""
    
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
        # Ensure no clipping before conversion
        peak = np.max(np.abs(samples))
        if peak > 1.0:
            samples /= peak
            
        samples = (samples * 32767).astype(np.int16)
        if samples.shape[0] == 2:
            samples = samples.T.flatten()
            return AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=2)
        else:
            return AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    def dj_stitch(self, track_paths, output_path, overlay_ms=8000):
        """
        Creates a 'DJ Mix' style sequence with 'Bass Swap' and Limiter protection.
        """
        if not track_paths:
            return None
            
        # 1. Load and Normalize first track
        combined = AudioSegment.from_file(track_paths[0])
        combined = combined.set_frame_rate(self.sr).set_channels(2)
        combined = effects.normalize(combined, headroom=1.0)
        
        for next_track_path in tqdm(track_paths[1:], desc="Stitching tracks"):
            next_seg = AudioSegment.from_file(next_track_path)
            next_seg = next_seg.set_frame_rate(self.sr).set_channels(2)
            # Normalize incoming track to match
            next_seg = effects.normalize(next_seg, headroom=1.0)
            
            # Prepare segments for transition
            out_seg = combined[-overlay_ms:]
            in_seg = next_seg[:overlay_ms]
            
            # Convert to numpy for filtering and limiting
            out_np = self.segment_to_numpy(out_seg)
            in_np = self.segment_to_numpy(in_seg)
            
            # 2. Apply Filters (The Bass Swap)
            hp_board = Pedalboard([HighpassFilter(cutoff_frequency_hz=300)])
            out_filtered = hp_board(out_np, self.sr)
            
            lp_board = Pedalboard([LowpassFilter(cutoff_frequency_hz=300)])
            in_filtered = lp_board(in_np, self.sr)
            
            # 3. Layer and Apply Limiter to transition
            # This prevents the sum of two tracks from exceeding 0dB
            summed_np = out_filtered + in_filtered
            limiter = Pedalboard([Limiter(threshold_db=-0.5, release_ms=100)])
            transition_np = limiter(summed_np, self.sr)
            
            # Convert back
            transition_seg = self.numpy_to_segment(transition_np, self.sr)
            
            # Reconstruct with fades for smooth transition
            combined = combined[:-overlay_ms] + transition_seg + next_seg[overlay_ms:]
            
        # Final pass normalization to ensure perfect levels
        combined = effects.normalize(combined, headroom=0.1)
        combined.export(output_path, format="mp3", bitrate="320k")
        return output_path

if __name__ == "__main__":
    # Test mixing the original and the stretched version
    renderer = FlowRenderer()
    # We will use this in preview_mix.py
