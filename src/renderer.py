from pydub import AudioSegment, effects
import os
import numpy as np
import pedalboard
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Limiter, Compressor
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
        # Safety normalization
        peak = np.max(np.abs(samples))
        if peak > 1.0:
            samples /= peak
            
        samples = (samples * 32767).astype(np.int16)
        if samples.shape[0] == 2:
            samples = samples.T.flatten()
            return AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=2)
        else:
            return AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    def dj_stitch(self, track_paths, output_path, overlay_ms=12000):
        """
        Creates a 'Professional DJ Mix' with Parallel Filter Blending and S-curve easing.
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
            
            # Prepare segments
            out_seg = combined[-overlay_ms:]
            in_seg = next_seg[:overlay_ms]
            
            out_np = self.segment_to_numpy(out_seg)
            in_np = self.segment_to_numpy(in_seg)
            
            # S-Curve for blending
            num_samples = out_np.shape[1]
            t = np.linspace(0, 1, num_samples)
            s_curve = 0.5 * (1 - np.cos(np.pi * t))
            
            # 1. OUTGOING: Blend from Unfiltered to High-passed (remove bass)
            hp_board = Pedalboard([HighpassFilter(cutoff_frequency_hz=400)])
            out_hp = hp_board(out_np, self.sr)
            # Starts unfiltered (1.0), ends high-passed (0.0)
            # Actually, we want to gradually apply the filter
            out_final_np = (out_np * (1.0 - s_curve)) + (out_hp * s_curve)
            # Then apply the overall volume fade
            out_final_np *= (1.0 - s_curve)

            # 2. INCOMING: Blend from Low-passed to Unfiltered (add highs)
            lp_board = Pedalboard([LowpassFilter(cutoff_frequency_hz=400)])
            in_lp = lp_board(in_np, self.sr)
            # Starts low-passed (1.0), ends unfiltered (0.0) -> s_curve goes 0 to 1
            in_final_np = (in_lp * (1.0 - s_curve)) + (in_np * s_curve)
            # Then apply the overall volume fade-in
            in_final_np *= s_curve
            
            # 3. Sum and Glue
            summed_np = out_final_np + in_final_np
            master_bus = Pedalboard([
                Compressor(threshold_db=-12, ratio=2.5, attack_ms=10, release_ms=200),
                Limiter(threshold_db=-0.1)
            ])
            transition_np = master_bus(summed_np, self.sr)
            
            transition_seg = self.numpy_to_segment(transition_np, self.sr)
            combined = combined[:-overlay_ms] + transition_seg + next_seg[overlay_ms:]
            
        combined = effects.normalize(combined, headroom=0.1)
        combined.export(output_path, format="mp3", bitrate="320k")
        return output_path
