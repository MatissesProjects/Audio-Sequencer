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
            out_final_np = (out_np * (1.0 - s_curve)) + (out_hp * s_curve)
            out_final_np *= (1.0 - s_curve)

            # 2. INCOMING: Blend from Low-passed to Unfiltered (add highs)
            lp_board = Pedalboard([LowpassFilter(cutoff_frequency_hz=400)])
            in_lp = lp_board(in_np, self.sr)
            in_final_np = (in_lp * (1.0 - s_curve)) + (in_np * s_curve)
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

    def layered_mix(self, foundation_path, layer_configs, output_path):
        """
        Layers multiple tracks over a foundation with smooth S-curve frequency ducking.
        """
        foundation = AudioSegment.from_file(foundation_path)
        foundation = foundation.set_frame_rate(self.sr).set_channels(2)
        foundation = effects.normalize(foundation, headroom=0.5)
        
        final = foundation
        
        # We process the foundation in chunks to allow for smooth automation
        for config in tqdm(layer_configs, desc="Layering tracks"):
            layer = AudioSegment.from_file(config['path'])
            layer = layer.set_frame_rate(self.sr).set_channels(2)
            layer = effects.normalize(layer, headroom=1.0)
            
            duration_ms = len(layer)
            start_ms = config['start_ms']
            fade_ms = 4000 # 4-second smooth transition for ducking
            
            if start_ms + duration_ms > len(final):
                continue

            # 1. Extract the foundation part that needs ducking
            # We take extra room for the fades
            duck_seg = final[start_ms : start_ms + duration_ms]
            duck_np = self.segment_to_numpy(duck_seg)
            num_samples = duck_np.shape[1]
            
            # 2. Create the Ducking Envelope (S-Curve)
            # 0.0 = no ducking, 1.0 = full ducking
            t = np.linspace(0, 1, int(self.sr * fade_ms / 1000))
            s_fade_in = 0.5 * (1 - np.cos(np.pi * t))
            s_fade_out = 1.0 - s_fade_in
            
            envelope = np.ones(num_samples)
            # Apply fade in at start
            envelope[:len(s_fade_in)] = s_fade_in
            # Apply fade out at end
            envelope[-len(s_fade_out):] = s_fade_out
            
            # 3. Apply Progressive Filtering to Foundation
            # Blend between clean and high-passed based on envelope
            hp_board = Pedalboard([HighpassFilter(cutoff_frequency_hz=600)])
            duck_hp = hp_board(duck_np, self.sr)
            
            # Foundation becomes filtered and quieter as layer swells in
            ducked_np = (duck_np * (1.0 - envelope * 0.5)) + (duck_hp * (envelope * 0.5))
            ducked_np *= (1.0 - envelope * 0.4) # Overall gain reduction
            
            ducked_seg = self.numpy_to_segment(ducked_np, self.sr)
            
            # 4. Prepare layer with matching crossfades
            layer_processed = layer + config.get('gain', -2.0)
            layer_processed = layer_processed.fade_in(fade_ms).fade_out(fade_ms)
            
            overlaid = ducked_seg.overlay(layer_processed)
            final = final[:start_ms] + overlaid + final[start_ms + duration_ms:]
            
        final = effects.normalize(final, headroom=0.1)
        final.export(output_path, format="mp3", bitrate="320k")
        return output_path
