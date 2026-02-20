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

    def render_timeline(self, segments, output_path, target_bpm=124):
        """
        Renders a full mix from a list of timeline segments.
        Each segment should have: file_path, start_ms, duration_ms, bpm, offset_ms, is_primary
        """
        if not segments:
            return None
            
        total_duration = max(s['start_ms'] + s['duration_ms'] for s in segments) + 2000 
        master = AudioSegment.silent(duration=total_duration, frame_rate=self.sr)
        master = master.set_channels(2)

        print(f"Rendering timeline: {len(segments)} segments, {total_duration/1000:.1f}s total.")

        processed_audios = []
        
        # Phase 1: Process each segment individually
        for i, s in enumerate(tqdm(segments, desc="Processing Segments")):
            from src.processor import AudioProcessor
            proc = AudioProcessor()
            
            # 1. Loop/Trim to duration + offset
            required_raw_dur = (s['duration_ms'] + s['offset_ms']) / 1000.0
            onsets = [] 
            tmp_loop = f"temp_render_{i}_loop.wav"
            proc.loop_track(s['file_path'], required_raw_dur + 2.0, onsets, tmp_loop)
            
            # 2. Stretch to target BPM
            y_sync = proc.stretch_to_bpm(tmp_loop, s['bpm'], target_bpm)
            
            # NEW: 2.5 Pitch Shift if needed
            ps = s.get('pitch_shift', 0)
            if ps != 0:
                import librosa
                y_sync = librosa.effects.pitch_shift(y_sync, sr=self.sr, n_steps=ps)

            seg_audio = self.numpy_to_segment(y_sync, self.sr)
            if os.path.exists(tmp_loop): os.remove(tmp_loop)

            # 3. Apply Offset Trim
            # Start at offset_ms and take duration_ms
            seg_audio = seg_audio[s['offset_ms'] : s['offset_ms'] + s['duration_ms']]
            
            # 4. Initial Gain & Fades
            vol_db = 20 * np.log10(s.get('volume', 1.0) + 1e-9)
            seg_audio = seg_audio + vol_db
            
            fi = s.get('fade_in_ms', 2000)
            fo = s.get('fade_out_ms', 2000)
            seg_audio = seg_audio.fade_in(fi).fade_out(fo)
            
            processed_audios.append({
                'audio': seg_audio,
                'start_ms': s['start_ms'],
                'is_primary': s.get('is_primary', False)
            })

        # Phase 2: Overlay with Lead Focus Ducking
        for i, current in enumerate(processed_audios):
            audio_to_overlay = current['audio']
            
            # Check for ducking: if I am NOT primary, but someone else overlapping IS primary
            if not current['is_primary']:
                curr_end = current['start_ms'] + len(audio_to_overlay)
                for other in processed_audios:
                    if other == current: continue
                    if other['is_primary']:
                        other_end = other['start_ms'] + len(other['audio'])
                        # Check overlap
                        overlap_start = max(current['start_ms'], other['start_ms'])
                        overlap_end = min(curr_end, other_end)
                        
                        if overlap_start < overlap_end:
                            # Duck this segment during the overlap
                            # For simplicity, we duck the WHOLE segment if it overlaps a primary
                            # In a pro version, we'd use automation curves
                            audio_to_overlay = audio_to_overlay - 4.0 # Duck 4dB
                            break

            master = master.overlay(audio_to_overlay, position=current['start_ms'])

        # Final Polish
        master = effects.normalize(master, headroom=0.1)
        master.export(output_path, format="mp3", bitrate="320k")
        return output_path

