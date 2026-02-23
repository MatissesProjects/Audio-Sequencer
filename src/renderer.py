from pydub import AudioSegment, effects
import os
import numpy as np
import pedalboard
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Limiter, Compressor, Reverb, Phaser, Chorus, Distortion
from tqdm import tqdm

class FlowRenderer:
    """Handles mixing, layering, and crossfading multiple tracks with pro gain staging."""
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def segment_to_numpy(self, seg):
        """Helper to convert pydub segment to numpy float32 (stereo)."""
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        samples /= (1 << (8 * seg.sample_width - 1))
        if seg.channels == 2:
            samples = samples.reshape((-1, 2)).T
        else:
            # Duplicate mono to stereo
            samples = np.stack([samples, samples])
        return samples

    def numpy_to_segment(self, samples, sr):
        """Helper to convert numpy float32 back to pydub segment."""
        peak = np.max(np.abs(samples))
        if peak > 1.0:
            samples /= (peak + 1e-6)
            
        samples = (samples * 32767).astype(np.int16)
        if samples.shape[0] == 2:
            samples = samples.T.flatten()
            return AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=2)
        else:
            return AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    def dj_stitch(self, track_paths, output_path, overlay_ms=20000):
        """Simplified sequential stitch for quick previews."""
        if not track_paths: return None
        combined = AudioSegment.from_file(track_paths[0]).set_frame_rate(self.sr).set_channels(2)
        for next_p in track_paths[1:]:
            next_s = AudioSegment.from_file(next_p).set_frame_rate(self.sr).set_channels(2)
            combined = combined.append(next_s, crossfade=min(len(combined)//3, len(next_s)//3, overlay_ms))
        combined.export(output_path, format="mp3", bitrate="320k")
        return output_path

    def render_timeline(self, segments, output_path, target_bpm=124, mutes=None, solos=None, progress_cb=None, time_range=None):
        return self._render_internal(segments, output_path, target_bpm, mutes, solos, progress_cb, time_range)

    def render_stems(self, segments, output_folder, target_bpm=124, progress_cb=None, time_range=None):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        lanes = {}
        for s in segments:
            l = s.get('lane', 0)
            if l not in lanes: lanes[l] = []
            lanes[l].append(s)
            
        stem_paths = []
        global_processed = 0
        for lane_id, lane_segs in lanes.items():
            path = os.path.join(output_folder, f"lane_{lane_id+1}.mp3")
            print(f"Exporting Stem: Lane {lane_id+1}...")
            
            # Closure to capture current total and pass to internal
            def stem_cb(count, current_lane_total=global_processed):
                if progress_cb: progress_cb(current_lane_total + count)
                
            self._render_internal(lane_segs, path, target_bpm, progress_cb=stem_cb, time_range=time_range)
            global_processed += len(lane_segs)
            stem_paths.append(path)
            
        return stem_paths

    def _apply_sidechain(self, target_samples, source_samples, amount=0.8):
        """Applies volume ducking to target based on source envelope (Fake Sidechain)."""
        # Calculate RMS envelope of source
        frame_len = 1024
        hop_len = 512
        rms = np.array([np.sqrt(np.mean(source_samples[:, i:i+frame_len]**2)) 
                        for i in range(0, source_samples.shape[1], hop_len)])
        
        # Upsample envelope to match sample rate
        envelope = np.repeat(rms, hop_len)[:source_samples.shape[1]]
        if len(envelope) < source_samples.shape[1]:
            envelope = np.pad(envelope, (0, source_samples.shape[1] - len(envelope)))
            
        # Invert envelope for ducking
        # Normalize envelope 0-1
        if np.max(envelope) > 0:
            envelope /= np.max(envelope)
            
        # Apply ducking curve: 1.0 - (env * amount)
        ducking_curve = 1.0 - (envelope * amount)
        ducking_curve = np.clip(ducking_curve, 0.2, 1.0) # Floor at -14dB
        
        # Match lengths if target is different
        min_len = min(target_samples.shape[1], len(ducking_curve))
        target_samples[:, :min_len] *= ducking_curve[:min_len]
        return target_samples

    def _apply_panning(self, samples, pan):
        """
        Applies linear stereo panning.
        pan: -1.0 (Full Left) to 1.0 (Full Right)
        """
        if pan == 0: return samples
        l_gain = max(0.0, min(1.0, 1.0 - pan))
        r_gain = max(0.0, min(1.0, 1.0 + pan))
        samples[0, :] *= l_gain
        samples[1, :] *= r_gain
        return samples

    def _apply_spectral_ducking(self, target_samples, sr, low_cut=300, high_cut=12000):
        """Applies filters to clear mud or soften air."""
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=low_cut),
            LowpassFilter(cutoff_frequency_hz=high_cut)
        ])
        return board(target_samples, sr)

    def _render_internal(self, segments, output_path, target_bpm=124, mutes=None, solos=None, progress_cb=None, time_range=None):
        """
        Internal rendering logic shared by full mix and stems.
        """
        if not segments:
            return None
            
        range_start, range_end = (0, 0)
        if time_range:
            range_start, range_end = time_range

        # Filter segments based on Mute/Solo and Time Range
        active_segments = []
        any_solo = any(solos) if solos else False
        for s in segments:
            if time_range:
                s_end = s['start_ms'] + s['duration_ms']
                if s_end <= range_start or s['start_ms'] >= range_end:
                    continue

            l = s.get('lane', 0)
            is_muted = mutes[l] if mutes and l < len(mutes) else False
            is_soloed = solos[l] if solos and l < len(solos) else False
            
            if any_solo:
                if is_soloed: active_segments.append(s)
            elif not is_muted:
                active_segments.append(s)
        
        if not active_segments:
            dur = (range_end - range_start) if time_range else 1000
            silence = np.zeros((2, int(self.sr * max(1000, dur) / 1000.0)), dtype=np.float32)
            self.numpy_to_segment(silence, self.sr).export(output_path, format="mp3")
            return output_path

        # Calculate master buffer size
        if time_range:
            total_duration_ms = range_end - range_start
        else:
            total_duration_ms = max(s['start_ms'] + s['duration_ms'] for s in active_segments) + 2000 
            
        master_samples = np.zeros((2, int(self.sr * total_duration_ms / 1000.0)), dtype=np.float32)

        print(f"Sonic Rendering ({'Region' if time_range else 'Full'}): {len(active_segments)} segments...")
        if progress_cb: progress_cb(0)

        processed_data = []
        for i, s in enumerate(active_segments):
            s_start = s['start_ms']
            s_dur = s['duration_ms']
            s_off = s['offset_ms']
            
            render_start_ms = s_start
            render_end_ms = s_start + s_dur
            render_offset_ms = s_off
            
            if time_range:
                if render_start_ms < range_start:
                    diff = range_start - render_start_ms
                    render_offset_ms += diff
                    render_start_ms = range_start
                if render_end_ms > range_end:
                    render_end_ms = range_end
            
            effective_dur = render_end_ms - render_start_ms
            if effective_dur <= 0: continue

            from src.processor import AudioProcessor
            proc = AudioProcessor()
            
            required_raw_dur = (effective_dur + render_offset_ms) / 1000.0
            tmp_loop = f"temp_render_{i}.wav"
            proc.loop_track(s['file_path'], required_raw_dur + 1.0, [], tmp_loop)
            
            y_sync = proc.stretch_to_bpm(tmp_loop, s['bpm'], target_bpm)
            ps = s.get('pitch_shift', 0)
            if ps != 0:
                import librosa
                y_sync = librosa.effects.pitch_shift(y_sync, sr=self.sr, n_steps=ps)
            
            if os.path.exists(tmp_loop): os.remove(tmp_loop)

            seg_audio = self.numpy_to_segment(y_sync, self.sr)
            seg_audio = seg_audio[int(render_offset_ms) : int(render_offset_ms + effective_dur)]
            seg_np = self.segment_to_numpy(seg_audio)
            num_samples = seg_np.shape[1]

            clip_rms = np.sqrt(np.mean(seg_np**2)) + 1e-9
            target_rms = 0.15 
            balancing_gain = target_rms / clip_rms
            vol_mult = s.get('volume', 1.0)
            
            fi_s = int(s.get('fade_in_ms', 2000) * self.sr / 1000.0)
            fo_s = int(s.get('fade_out_ms', 2000) * self.sr / 1000.0)
            if time_range and s_start < range_start: fi_s = 0
            if time_range and (s_start + s_dur) > range_end: fo_s = 0

            envelope = np.ones(num_samples, dtype=np.float32)
            if fi_s > 0:
                t_in = np.linspace(0, 1, min(fi_s, num_samples))
                envelope[:len(t_in)] = 0.5 * (1 - np.cos(np.pi * t_in))
            if fo_s > 0:
                t_out = np.linspace(0, 1, min(fo_s, num_samples))
                envelope[-len(t_out):] *= 0.5 * (1 + np.cos(np.pi * t_out))
            
            seg_np *= (balancing_gain * vol_mult * envelope)
            
            is_amb = s.get('is_ambient', False)
            lc = s.get('low_cut', 400 if is_amb else 20)
            hc = s.get('high_cut', 20000)
            if lc > 25 or hc < 19000:
                seg_np = self._apply_spectral_ducking(seg_np, self.sr, low_cut=lc, high_cut=hc)
            
            seg_np = self._apply_panning(seg_np, s.get('pan', 0.0))
            
            # --- APPLY FX (Reverb & Harmonics) ---
            rev_amt = s.get('reverb', 0.0)
            harm_amt = s.get('harmonics', 0.0)
            
            if rev_amt > 0 or harm_amt > 0:
                fx_chain = []
                if harm_amt > 0:
                    # Harmonics: Use Distortion for saturation
                    # We use moderate drive (up to 20dB) and mix it
                    fx_chain.append(Distortion(drive_db=harm_amt * 15))
                if rev_amt > 0:
                    # Reverb: Large room for "cool" cinematic feel
                    fx_chain.append(Reverb(room_size=0.8, wet_level=rev_amt * 0.6, dry_level=1.0 - (rev_amt * 0.2)))
                
                if fx_chain:
                    seg_np = Pedalboard(fx_chain)(seg_np, self.sr)

            processed_data.append({
                'samples': seg_np,
                'start_idx': int((render_start_ms - range_start) * self.sr / 1000.0) if time_range else int(render_start_ms * self.sr / 1000.0),
                'is_primary': s.get('is_primary', False),
                'is_ambient': is_amb
            })
            
            if progress_cb: progress_cb(i + 1)

        for i, current in enumerate(processed_data):
            samples = current['samples']; start = current['start_idx']; end = start + samples.shape[1]
            if not current['is_primary']:
                for other in processed_data:
                    if (other is current) or not other['is_primary']: continue
                    o_start = other['start_idx']; o_end = o_start + other['samples'].shape[1]
                    overlap_start = max(start, o_start); overlap_end = min(end, o_end)
                    if overlap_start < overlap_end:
                        samples = self._apply_spectral_ducking(samples, self.sr)
                        rel_start_o = overlap_start - o_start; rel_end_o = overlap_end - o_start
                        source_segment = other['samples'][:, rel_start_o:rel_end_o]
                        rel_start_c = overlap_start - start; rel_end_c = overlap_end - start
                        target_segment = samples[:, rel_start_c:rel_end_c]
                        duck_amt = 0.85 if current['is_ambient'] else 0.7
                        ducked_segment = self._apply_sidechain(target_segment, source_segment, amount=duck_amt)
                        samples[:, rel_start_c:rel_end_c] = ducked_segment
                        break
            render_end = min(master_samples.shape[1], end); render_len = render_end - start
            if render_len > 0: master_samples[:, start:render_end] += samples[:, :render_len]

        print("Finalizing Master Bus...")
        master_bus = Pedalboard([Compressor(threshold_db=-14, ratio=2.5), Limiter(threshold_db=-0.1)])
        final_y = master_bus(master_samples, self.sr)
        final_seg = self.numpy_to_segment(final_y, self.sr)
        final_seg.export(output_path, format="mp3", bitrate="320k")
        return output_path
