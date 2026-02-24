from pydub import AudioSegment, effects
import os
import numpy as np
import pedalboard
import hashlib
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Limiter, Compressor, Reverb, Phaser, Chorus, Distortion
from tqdm import tqdm
from src.processor import AudioProcessor
from src.core.config import AppConfig
from src.core.effects import FXChain

def _process_single_segment(s, i, target_bpm, sr, time_range):
    """Standalone function for parallel processing of a single segment with caching."""
    s_start = s['start_ms']
    s_dur = s['duration_ms']
    s_off = s['offset_ms']
    stems_dir = s.get('stems_path')
    
    range_start = time_range[0] if time_range else 0
    range_end = time_range[1] if time_range else (s_start + s_dur)
    
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
    if effective_dur <= 0: return None

    # --- CACHE CHECK ---
    cache_dir = AppConfig.CACHE_DIR
    AppConfig.ensure_dirs()
    # Create unique hash for this segment's heavy processing
    # (file + bpm + pitch + duration + offset + stem mix + harmony)
    key_str = f"{s['file_path']}_{s['bpm']}_{target_bpm}_{s.get('pitch_shift',0)}_{s_dur}_{s_off}_{stems_dir}_{s.get('vocal_shift',0)}_{s.get('harmony_level',0)}_{s.get('vocal_vol',1.0)}_{s.get('drum_vol',1.0)}_{s.get('instr_vol',1.0)}"
    cache_hash = hashlib.md5(key_str.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_hash}.npy")
    
    if os.path.exists(cache_file):
        try:
            seg_np = np.load(cache_file)
            return {
                'samples': seg_np,
                'start_idx': int((render_start_ms - range_start) * sr / 1000.0) if time_range else int(render_start_ms * sr / 1000.0),
                'is_primary': s.get('is_primary', False),
                'is_ambient': s.get('is_ambient', False),
                'vocal_energy': s.get('vocal_energy', 0.0),
                'ducking_depth': s.get('ducking_depth', 0.7)
            }
        except: pass

    proc = AudioProcessor(sample_rate=sr)
    fx_chain = FXChain()
    required_raw_dur = (effective_dur + render_offset_ms) / 1000.0
    
    # --- STEM PROCESSING ---
    if stems_dir and os.path.exists(stems_dir):
        combined_seg_np = None
        stem_types = ["vocals", "drums", "other"]
        
        for stype in stem_types:
            stem_file = os.path.join(stems_dir, f"{stype}.wav")
            if not os.path.exists(stem_file): continue
            
            # 1. Load & Loop
            y, _ = librosa.load(stem_file, sr=sr)
            onsets = [float(x)*1000 for x in s.get('onsets_json', "").split(',') if x]
            y_looped = proc.loop_numpy(y, sr, required_raw_dur + 1.0, onsets)
            
            # 2. Stretch
            y_sync = proc.stretch_numpy(y_looped, sr, s['bpm'], target_bpm)
            
            # 3. Pitch Shift
            if stype == "vocals":
                v_ps = s.get('pitch_shift', 0) + s.get('vocal_shift', 0)
                if v_ps != 0:
                    y_sync = proc.shift_pitch_numpy(y_sync, sr, v_ps)
            else:
                ps = s.get('pitch_shift', 0)
                if ps != 0:
                    y_sync = proc.shift_pitch_numpy(y_sync, sr, ps)
            
            # 4. Trim to exactly what we need
            start_sample = int(render_offset_ms * sr / 1000.0)
            end_sample = int((render_offset_ms + effective_dur) * sr / 1000.0)
            y_sync = y_sync[start_sample : end_sample]
            
            # 5. Convert to Stereo
            if len(y_sync.shape) == 1:
                stem_np = np.stack([y_sync, y_sync])
            else:
                stem_np = y_sync

            # Apply Vocal Specific Harmonics & Harmonic Rhythms
            if stype == "vocals":
                vocal_harm = 0.4 + (s.get('harmonics', 0.0) * 0.6)
                stem_np = Pedalboard([Distortion(drive_db=vocal_harm * 12)])(stem_np, sr)
                stem_np *= s.get('vocal_vol', 1.0)

                h_level = s.get('harmony_level', 0.0)
                if h_level > 0:
                    # 1. Perfect 5th Layer (+7st)
                    h_layer1 = proc.shift_pitch_numpy(y_sync, sr, 7)
                    if len(h_layer1.shape) == 1: h_layer1 = np.stack([h_layer1, h_layer1])
                    h_layer1 = proc.apply_rhythmic_gate(h_layer1, sr, target_bpm, pattern="1/8")
                    
                    # 2. Octave Layer (+12st)
                    h_layer2 = proc.shift_pitch_numpy(y_sync, sr, 12)
                    if len(h_layer2.shape) == 1: h_layer2 = np.stack([h_layer2, h_layer2])
                    h_layer2 = proc.apply_rhythmic_gate(h_layer2, sr, target_bpm, pattern="1/4")
                    
                    # 3. Mix layers
                    min_l = min(stem_np.shape[1], h_layer1.shape[1], h_layer2.shape[1])
                    stem_np[:, :min_l] += (h_layer1[:, :min_l] * h_level * 0.5)
                    stem_np[:, :min_l] += (h_layer2[:, :min_l] * h_level * 0.3)
            elif stype == "drums":
                stem_np *= s.get('drum_vol', 1.0)
            elif stype == "other":
                stem_np *= s.get('instr_vol', 1.0)
            
            if combined_seg_np is None:
                combined_seg_np = stem_np
            else:
                if stype == "other":
                    rms = np.sqrt(np.mean(combined_seg_np**2, axis=0))
                    envelope = np.repeat(rms[::512], 512)[:combined_seg_np.shape[1]]
                    if len(envelope) < combined_seg_np.shape[1]:
                        envelope = np.pad(envelope, (0, combined_seg_np.shape[1]-len(envelope)))
                    if np.max(envelope) > 0: envelope /= np.max(envelope)
                    ducking = 1.0 - (envelope * 0.5)
                    stem_np *= ducking
                
                min_l = min(combined_seg_np.shape[1], stem_np.shape[1])
                combined_seg_np[:, :min_l] += stem_np[:, :min_l]
        seg_np = combined_seg_np
    else:
        # Single file fallback
        y, _ = librosa.load(s['file_path'], sr=sr)
        onsets = [float(x)*1000 for x in s.get('onsets_json', "").split(',') if x]
        y_looped = proc.loop_numpy(y, sr, required_raw_dur + 1.0, onsets)
        y_sync = proc.stretch_numpy(y_looped, sr, s['bpm'], target_bpm)
        ps = s.get('pitch_shift', 0)
        if ps != 0:
            y_sync = proc.shift_pitch_numpy(y_sync, sr, ps)
        
        start_sample = int(render_offset_ms * sr / 1000.0)
        end_sample = int((render_offset_ms + effective_dur) * sr / 1000.0)
        y_sync = y_sync[start_sample : end_sample]
        if len(y_sync.shape) == 1:
            seg_np = np.stack([y_sync, y_sync])
        else:
            seg_np = y_sync

    # --- BALANCING & FADES ---
    clip_rms = np.sqrt(np.mean(seg_np**2)) + 1e-9
    balancing_gain = 0.15 / clip_rms
    vol_mult = s.get('volume', 1.0)
    
    num_samples = seg_np.shape[1]
    fi_s = int(s.get('fade_in_ms', 2000) * sr / 1000.0)
    fo_s = int(s.get('fade_out_ms', 2000) * sr / 1000.0)
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
    
    # --- MODULAR FX CHAIN ---
    seg_np = fx_chain.process(seg_np, sr, s)
    
    pan = s.get('pan', 0.0)
    if pan != 0:
        l_gain = max(0.0, min(1.0, 1.0 - pan))
        r_gain = max(0.0, min(1.0, 1.0 + pan))
        seg_np[0, :] *= l_gain
        seg_np[1, :] *= r_gain

    # --- SAVE TO CACHE ---
    try: np.save(cache_file, seg_np)
    except: pass

    return {
        'samples': seg_np,
        'start_idx': int((render_start_ms - range_start) * sr / 1000.0) if time_range else int(render_start_ms * sr / 1000.0),
        'is_primary': s.get('is_primary', False),
        'is_ambient': s.get('is_ambient', False),
        'vocal_energy': s.get('vocal_energy', 0.0),
        'ducking_depth': s.get('ducking_depth', 0.7)
    }

class FlowRenderer:
    """Handles mixing, layering, and crossfading multiple tracks with pro gain staging."""
    
    def __init__(self, sample_rate=None):
        self.sr = sample_rate or AppConfig.SAMPLE_RATE

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

    def render_timeline(self, segments, output_path, target_bpm=None, mutes=None, solos=None, progress_cb=None, time_range=None):
        target_bpm = target_bpm or AppConfig.DEFAULT_BPM
        return self._render_internal(segments, output_path, target_bpm, mutes, solos, progress_cb, time_range)

    def render_stems(self, segments, output_folder, target_bpm=None, progress_cb=None, time_range=None):
        target_bpm = target_bpm or AppConfig.DEFAULT_BPM
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
        Optimized parallel rendering logic.
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

        print(f"Parallel Sonic Rendering: {len(active_segments)} segments...")
        
        processed_data = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_process_single_segment, s, i, target_bpm, self.sr, time_range) 
                       for i, s in enumerate(active_segments)]
            
            completed = 0
            for future in as_completed(futures):
                res = future.result()
                if res:
                    processed_data.append(res)
                completed += 1
                if progress_cb: progress_cb(completed)

        print("Mixing & Applying Dynamic Ducking...")
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
                        
                        # Vocal-aware ducking base amount
                        is_vocal = other.get('vocal_energy', 0) > 0.2
                        base_duck = 0.9 if is_vocal else (0.85 if current['is_ambient'] else 0.7)
                        
                        # Apply user-defined ducking depth preference
                        depth = current.get('ducking_depth', 0.7)
                        final_duck_amt = base_duck * (depth / 0.7) # Scale based on 0.7 being "normal"
                        
                        ducked_segment = self._apply_sidechain(target_segment, source_segment, amount=min(0.95, final_duck_amt))
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
