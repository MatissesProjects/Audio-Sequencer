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
        peak = np.max(np.abs(samples))
        if peak > 1.0:
            samples /= peak
            
        samples = (samples * 32767).astype(np.int16)
        if samples.shape[0] == 2:
            samples = samples.T.flatten()
            return AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=2)
        else:
            return AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    def generate_curve(self, type, num_samples, intro_samples):
        """Generates precisely sized automation curves using S-curve easing."""
        swap_samples = num_samples - intro_samples
        if swap_samples < 0:
            intro_samples = num_samples
            swap_samples = 0
            
        # S-Curve for the swap phase
        if swap_samples > 0:
            t = np.linspace(0, 1, swap_samples)
            s_curve = 0.5 * (1 - np.cos(np.pi * t))
        else:
            s_curve = np.array([])

        if type == 'out_filter': # 0 -> 1
            res = np.concatenate([np.zeros(intro_samples), s_curve])
        elif type == 'in_filter': # 1 -> 0
            res = np.concatenate([np.ones(intro_samples), 1.0 - s_curve])
        elif type == 'out_vol': # 1 -> 0
            vol_curve = np.cos(0.5 * np.pi * t) if swap_samples > 0 else np.array([])
            res = np.concatenate([np.ones(intro_samples), vol_curve])
        elif type == 'in_vol': # 0 -> 0.7 -> 1
            vol_intro = np.linspace(0, 0.7, intro_samples)
            vol_swap = 0.7 + (0.3 * np.sin(0.5 * np.pi * t)) if swap_samples > 0 else np.array([])
            res = np.concatenate([vol_intro, vol_swap])
        else:
            res = np.ones(num_samples)
            
        return res[:num_samples]

    def dj_stitch(self, track_paths, output_path, overlay_ms=20000):
        """
        Creates a 'Professional DJ Mix' with precise shape matching.
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
            
            safe_overlay = min(overlay_ms, int(len(combined) * 0.4), int(len(next_seg) * 0.4))
            
            out_seg = combined[-safe_overlay:]
            in_seg = next_seg[:safe_overlay]
            
            out_np = self.segment_to_numpy(out_seg)
            in_np = self.segment_to_numpy(in_seg)
            
            # THE MASTER SIZE
            num_samples = out_np.shape[1]
            in_np = in_np[:, :num_samples]
            
            intro_samples = int(self.sr * (safe_overlay * 0.4) / 1000)
            
            # Generate curves directly at target size - reshape to (1, N) for broadcasting
            out_f = self.generate_curve('out_filter', num_samples, intro_samples).reshape(1, -1)
            in_f = self.generate_curve('in_filter', num_samples, intro_samples).reshape(1, -1)
            out_v = self.generate_curve('out_vol', num_samples, intro_samples).reshape(1, -1)
            in_v = self.generate_curve('in_vol', num_samples, intro_samples).reshape(1, -1)

            # Processing
            hp_board = Pedalboard([HighpassFilter(cutoff_frequency_hz=400)])
            out_hp = hp_board(out_np, self.sr)[:, :num_samples]
            in_hp = hp_board(in_np, self.sr)[:, :num_samples]
            
            out_final = ((out_np * (1.0 - out_f)) + (out_hp * out_f)) * out_v
            in_final = ((in_hp * in_f) + (in_np * (1.0 - in_f))) * in_v
            
            summed_np = out_final + in_final
            master_bus = Pedalboard([
                Compressor(threshold_db=-12, ratio=2.5),
                Limiter(threshold_db=-0.1)
            ])
            transition_np = master_bus(summed_np, self.sr)
            
            transition_seg = self.numpy_to_segment(transition_np, self.sr)
            combined = combined[:-safe_overlay] + transition_seg + next_seg[safe_overlay:]
            
        combined = effects.normalize(combined, headroom=0.1)
        combined.export(output_path, format="mp3", bitrate="320k")
        return output_path

    def layered_mix(self, foundation_path, layer_configs, output_path):
        foundation = AudioSegment.from_file(foundation_path)
        foundation = foundation.set_frame_rate(self.sr).set_channels(2)
        foundation = effects.normalize(foundation, headroom=0.5)
        final = foundation
        
        for config in tqdm(layer_configs, desc="Layering tracks"):
            layer = AudioSegment.from_file(config['path'])
            layer = layer.set_frame_rate(self.sr).set_channels(2)
            layer = effects.normalize(layer, headroom=1.0)
            
            duration_ms = len(layer)
            start_ms = config['start_ms']
            fade_ms = 4000
            
            if start_ms + duration_ms > len(final): continue

            duck_seg = final[start_ms : start_ms + duration_ms]
            duck_np = self.segment_to_numpy(duck_seg)
            num_samples = duck_np.shape[1]
            
            t = np.linspace(0, 1, int(self.sr * fade_ms / 1000))
            s_fade = (0.5 * (1 - np.cos(np.pi * t))).astype(np.float32)
            
            envelope = np.ones(num_samples, dtype=np.float32)
            f_s = len(s_fade)
            if num_samples >= f_s * 2:
                envelope[:f_s] = s_fade
                envelope[-f_s:] = 1.0 - s_fade
            
            env_b = envelope.reshape(1, -1)
            
            hp_board = Pedalboard([HighpassFilter(cutoff_frequency_hz=600)])
            duck_hp = hp_board(duck_np, self.sr)[:, :num_samples]
            
            ducked_np = (duck_np * (1.0 - env_b * 0.5)) + (duck_hp * (env_b * 0.5))
            ducked_np *= (1.0 - env_b * 0.4)
            
            ducked_seg = self.numpy_to_segment(ducked_np, self.sr)
            layer_processed = layer + config.get('gain', -2.0)
            layer_processed = layer_processed.fade_in(fade_ms).fade_out(fade_ms)
            
            overlaid = ducked_seg.overlay(layer_processed)
            final = final[:start_ms] + overlaid + final[start_ms + duration_ms:]
            
        final = effects.normalize(final, headroom=0.1)
        final.export(output_path, format="mp3", bitrate="320k")
        return output_path
