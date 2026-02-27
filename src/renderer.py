from pydub import AudioSegment, effects
import os
import numpy as np
import pedalboard
import hashlib
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Limiter, Compressor, Reverb, Phaser, Chorus, Distortion
from typing import List, Dict, Optional, Any, Union, Tuple, Callable
from src.processor import AudioProcessor
from src.core.config import AppConfig
from src.core.effects import FXChain

def _interpolate_value(points: List[Tuple[float, float]], current_ms: float, default_val: float) -> float:
    if not points: return default_val
    if current_ms <= points[0][0]: return points[0][1]
    if current_ms >= points[-1][0]: return points[-1][1]
    for i in range(len(points) - 1):
        t1, v1 = points[i]; t2, v2 = points[i+1]
        if t1 <= current_ms <= t2:
            return v1 + (v2 - v1) * (current_ms - t1) / (t2 - t1)
    return default_val

def _get_modulation_envelope(points: List[Tuple[float, float]], num_samples: int, sr: int, default_val: float = 1.0) -> np.ndarray:
    env = np.full(num_samples, default_val, dtype=np.float32)
    if not points: return env
    pts = sorted(points, key=lambda x: x[0])
    first_ms, first_val = pts[0]
    first_idx = int(first_ms * sr / 1000.0)
    env[:min(first_idx, num_samples)] = first_val
    for i in range(len(pts) - 1):
        t1, v1 = pts[i]; t2, v2 = pts[i+1]
        idx1 = int(t1 * sr / 1000.0); idx2 = int(t2 * sr / 1000.0)
        if idx2 <= 0 or idx1 >= num_samples: continue
        idx1 = max(0, idx1); idx2 = min(num_samples, idx2)
        if idx2 > idx1:
            env[idx1:idx2] = np.linspace(v1, v2, idx2 - idx1)
    last_ms, last_val = pts[-1]
    last_idx = int(last_ms * sr / 1000.0)
    if last_idx < num_samples:
        env[max(0, last_idx):] = last_val
    return env

def _process_single_segment(s: Dict[str, Any], i: int, target_bpm: float, sr: int, time_range: Optional[Tuple[int, int]]) -> Optional[Dict[str, Any]]:
    """Standalone function for parallel processing of a single segment with caching."""
    s_start = int(s['start_ms']); s_dur = int(s['duration_ms']); s_off = float(s['offset_ms'])
    stems_dir = s.get('stems_path')
    range_start = time_range[0] if time_range else 0
    range_end = time_range[1] if time_range else (s_start + s_dur)
    render_start_ms = s_start; render_end_ms = s_start + s_dur; render_offset_ms = s_off
    if time_range:
        if render_start_ms < range_start:
            diff = range_start - render_start_ms
            render_offset_ms += diff; render_start_ms = range_start
        if render_end_ms > range_end: render_end_ms = range_end
    effective_dur = render_end_ms - render_start_ms
    if effective_dur <= 0: return None
    cache_dir = AppConfig.CACHE_DIR; AppConfig.ensure_dirs()
    if not stems_dir:
        try:
            from src.database import DataManager
            dm = DataManager(); conn = dm.get_conn(); cursor = conn.cursor()
            cursor.execute("SELECT stems_path FROM tracks WHERE file_path = ?", (os.path.abspath(s['file_path']),))
            row = cursor.fetchone()
            if row and row[0]: stems_dir = row[0]
            conn.close()
        except: pass
    key_str = f"{s['file_path']}_{s['bpm']}_{target_bpm}_{s.get('pitch_shift',0)}_{s_dur}_{s_off}_{stems_dir}_{s.get('vocal_shift',0)}_{s.get('gender_swap','none')}_{s.get('harmony_level',0)}_{s.get('vocal_vol',1.0)}_{s.get('drum_vol',1.0)}_{s.get('bass_vol',1.0)}_{s.get('instr_vol',1.0)}_{s.get('duck_low',1.0)}_{s.get('duck_mid',1.0)}_{s.get('duck_high',1.0)}_{str(s.get('keyframes', {}))}"
    cache_hash = hashlib.md5(key_str.encode()).hexdigest(); cache_file = os.path.join(cache_dir, f"{cache_hash}.npy")
    if os.path.exists(cache_file):
        try:
            seg_np = np.load(cache_file)
            return {
                'samples': seg_np, 'start_idx': int((render_start_ms - range_start) * sr / 1000.0) if time_range else int(render_start_ms * sr / 1000.0),
                'is_primary': s.get('is_primary', False), 'is_ambient': s.get('is_ambient', False), 'vocal_energy': s.get('vocal_energy') or 0.0,
                'ducking_depth': s.get('ducking_depth') or 0.7, 'duck_low': s.get('duck_low') or 1.0, 'duck_mid': s.get('duck_mid') or 1.0, 'duck_high': s.get('duck_high') or 1.0
            }
        except: pass
    proc = AudioProcessor(sample_rate=sr); required_raw_dur = (effective_dur + render_offset_ms) / 1000.0; keyframes = s.get('keyframes', {})
    if stems_dir and os.path.exists(stems_dir):
        combined_seg_np = None; stem_types = ["vocals", "drums", "bass", "other"]
        for stype in stem_types:
            stem_file = os.path.join(stems_dir, f"{stype}.wav")
            if not os.path.exists(stem_file): continue
            v_shift_applied = False
            if stype == "vocals":
                g_swap = s.get('gender_swap', 'none')
                if g_swap != "none":
                    v_shift = s.get('vocal_shift', 0); gs_hash = hashlib.md5(f"{stem_file}_{g_swap}_{v_shift}".encode()).hexdigest(); gs_cache = os.path.join(AppConfig.CACHE_DIR, f"gs_{gs_hash}.wav")
                    if not os.path.exists(gs_cache): proc.generate_gender_swap_remote(stem_file, gs_cache, target=g_swap, steps=float(v_shift))
                    if os.path.exists(gs_cache): stem_file = gs_cache; v_shift_applied = True
            y, _ = librosa.load(stem_file, sr=sr)
            onsets = [float(x)*1000 for x in s.get('onsets_json', "").split(',') if x]
            y_looped = proc.loop_numpy(y, sr, required_raw_dur + 1.0, onsets)
            y_sync = proc.stretch_numpy(y_looped, sr, float(s['bpm']), target_bpm)
            if stype == "vocals":
                v_ps = float(s.get('pitch_shift', 0))
                if not v_shift_applied: v_ps += float(s.get('vocal_shift', 0))
                if v_ps != 0: y_sync = proc.shift_pitch_numpy(y_sync, sr, v_ps)
            elif stype == "bass":
                b_ps = float(s.get('pitch_shift', 0)) + float(s.get('bass_shift', 0))
                if b_ps != 0: y_sync = proc.shift_pitch_numpy(y_sync, sr, b_ps)
            elif stype == "drums":
                d_ps = float(s.get('pitch_shift', 0)) + float(s.get('drum_shift', 0))
                if d_ps != 0: y_sync = proc.shift_pitch_numpy(y_sync, sr, d_ps)
            elif stype == "other":
                i_ps = float(s.get('pitch_shift', 0)) + float(s.get('instr_shift', 0))
                if i_ps != 0: y_sync = proc.shift_pitch_numpy(y_sync, sr, i_ps)
            else:
                ps = float(s.get('pitch_shift', 0))
                if ps != 0: y_sync = proc.shift_pitch_numpy(y_sync, sr, ps)
            s_smpl = int(render_offset_ms * sr / 1000.0); e_smpl = int((render_offset_ms + effective_dur) * sr / 1000.0); y_sync = y_sync[s_smpl : e_smpl]
            stem_np = np.stack([y_sync, y_sync]) if len(y_sync.shape) == 1 else y_sync
            if stype == "vocals":
                v_harm = 0.4 + (s.get('harmonics', 0.0) * 0.6); stem_np = Pedalboard([Distortion(drive_db=v_harm * 12)])(stem_np, sr)
                if 'vocal_vol' in keyframes: stem_np *= _get_modulation_envelope(keyframes['vocal_vol'], stem_np.shape[1], sr, default_val=s.get('vocal_vol', 1.0))
                else: stem_np *= s.get('vocal_vol', 1.0)
                h_level = float(s.get('harmony_level', 0.0))
                if h_level > 0:
                    h_layer1 = proc.shift_pitch_numpy(y_sync, sr, 7); if len(h_layer1.shape) == 1: h_layer1 = np.stack([h_layer1, h_layer1])
                    h_layer1 = proc.apply_rhythmic_gate(h_layer1, sr, target_bpm, pattern="1/8")
                    h_layer2 = proc.shift_pitch_numpy(y_sync, sr, 12); if len(h_layer2.shape) == 1: h_layer2 = np.stack([h_layer2, h_layer2])
                    h_layer2 = proc.apply_rhythmic_gate(h_layer2, sr, target_bpm, pattern="1/4")
                    min_l = min(stem_np.shape[1], h_layer1.shape[1], h_layer2.shape[1]); stem_np[:, :min_l] += (h_layer1[:, :min_l] * h_level * 0.5) + (h_layer2[:, :min_l] * h_level * 0.3)
            elif stype == "drums":
                if 'drum_vol' in keyframes: stem_np *= _get_modulation_envelope(keyframes['drum_vol'], stem_np.shape[1], sr, default_val=s.get('drum_vol', 1.0))
                else: stem_np *= s.get('drum_vol', 1.0)
            elif stype == "bass":
                if 'bass_vol' in keyframes: stem_np *= _get_modulation_envelope(keyframes['bass_vol'], stem_np.shape[1], sr, default_val=s.get('bass_vol', 1.0))
                else: stem_np *= s.get('bass_vol', 1.0)
            elif stype == "other":
                if 'instr_vol' in keyframes: stem_np *= _get_modulation_envelope(keyframes['instr_vol'], stem_np.shape[1], sr, default_val=s.get('instr_vol', 1.0))
                else: stem_np *= s.get('instr_vol', 1.0)
            if combined_seg_np is None: combined_seg_np = stem_np
            else:
                if stype == "other":
                    rms = np.sqrt(np.mean(combined_seg_np**2, axis=0)); envelope = np.repeat(rms[::512], 512)[:combined_seg_np.shape[1]]
                    if len(envelope) < combined_seg_np.shape[1]: envelope = np.pad(envelope, (0, combined_seg_np.shape[1]-len(envelope)))
                    if len(envelope) > 0:
                        mv = np.max(envelope); if mv > 0: envelope /= mv
                        stem_np *= (1.0 - (envelope * 0.5))
                min_l = min(combined_seg_np.shape[1], stem_np.shape[1]); combined_seg_np[:, :min_l] += stem_np[:, :min_l]
        seg_np = combined_seg_np
    else:
        y, _ = librosa.load(s['file_path'], sr=sr); onsets = [float(x)*1000 for x in s.get('onsets_json', "").split(',') if x]
        y_looped = proc.loop_numpy(y, sr, required_raw_dur + 1.0, onsets); y_sync = proc.stretch_numpy(y_looped, sr, float(s['bpm']), target_bpm)
        ps = float(s.get('pitch_shift', 0)); if ps != 0: y_sync = proc.shift_pitch_numpy(y_sync, sr, ps)
        s_smpl = int(render_offset_ms * sr / 1000.0); e_smpl = int((render_offset_ms + effective_dur) * sr / 1000.0); y_sync = y_sync[s_smpl : e_smpl]
        seg_np = np.stack([y_sync, y_sync]) if len(y_sync.shape) == 1 else y_sync
    c_rms = np.sqrt(np.mean(seg_np**2)) + 1e-9; seg_np *= (0.15 / c_rms) * s.get('volume', 1.0)
    fi_s = int(s.get('fade_in_ms', 2000) * sr / 1000.0); fo_s = int(s.get('fade_out_ms', 2000) * sr / 1000.0)
    if time_range and s_start < range_start: fi_s = 0; if time_range and (s_start + s_dur) > range_end: fo_s = 0
    env = np.ones(seg_np.shape[1], dtype=np.float32)
    if fi_s > 0: t_in = np.linspace(0, 1, min(fi_s, seg_np.shape[1])); env[:len(t_in)] = 0.5 * (1 - np.cos(np.pi * t_in))
    if fo_s > 0: t_out = np.linspace(0, 1, min(fo_s, seg_np.shape[1])); env[-len(t_out):] *= 0.5 * (1 + np.cos(np.pi * t_out))
    seg_np *= env
    if 'volume' in keyframes: seg_np *= _get_modulation_envelope(keyframes['volume'], seg_np.shape[1], sr)
    if 'pan' in keyframes:
        pan_env = _get_modulation_envelope(keyframes['pan'], seg_np.shape[1], sr, default_val=s.get('pan', 0.0))
        seg_np[0, :] *= np.clip(1.0 - pan_env, 0.0, 1.0); seg_np[1, :] *= np.clip(1.0 + pan_env, 0.0, 1.0); s['pan_applied'] = True
    for p_name in ['low_cut', 'high_cut']:
        if p_name in keyframes and len(keyframes[p_name]) >= 2:
            chunk_size = int(sr * 0.5); pts = sorted(keyframes[p_name], key=lambda x: x[0])
            for i in range(0, seg_np.shape[1], chunk_size):
                end = min(i + chunk_size, seg_np.shape[1]); rel_ms = (i + (end-i)/2) * 1000.0 / sr
                freq = _interpolate_value(pts, rel_ms, s.get(p_name, 20 if p_name == 'low_cut' else 20000))
                if p_name == 'low_cut' and freq > 30: seg_np[:, i:end] = HighpassFilter(cutoff_frequency_hz=freq)(seg_np[:, i:end], sr)
                elif p_name == 'high_cut' and freq < 19000: seg_np[:, i:end] = LowpassFilter(cutoff_frequency_hz=freq)(seg_np[:, i:end], sr)
    seg_np = FXChain().process(seg_np, sr, s)
    pan = float(s.get('pan', 0.0)); if pan != 0 and not s.get('pan_applied'): seg_np[0, :] *= max(0.0, min(1.0, 1.0 - pan)); seg_np[1, :] *= max(0.0, min(1.0, 1.0 + pan))
    try: np.save(cache_file, seg_np)
    except: pass
    return {
        'samples': seg_np, 'start_idx': int((render_start_ms - range_start) * sr / 1000.0) if time_range else int(render_start_ms * sr / 1000.0),
        'is_primary': s.get('is_primary', False), 'is_ambient': s.get('is_ambient', False), 'vocal_energy': s.get('vocal_energy') or 0.0,
        'ducking_depth': s.get('ducking_depth') or 0.7, 'duck_low': s.get('duck_low') or 1.0, 'duck_mid': s.get('duck_mid') or 1.0, 'duck_high': s.get('duck_high') or 1.0
    }

class FlowRenderer:
    """Handles mixing, layering, and crossfading multiple tracks with pro gain staging."""
    
    def __init__(self, sample_rate: Optional[int] = None):
        self.sr: int = sample_rate or AppConfig.SAMPLE_RATE

    def segment_to_numpy(self, seg: AudioSegment) -> np.ndarray:
        """Helper to convert pydub segment to numpy float32 (stereo)."""
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        samples /= (1 << (8 * seg.sample_width - 1))
        if seg.channels == 2: return samples.reshape((-1, 2)).T
        return np.stack([samples, samples])

    def numpy_to_segment(self, samples: np.ndarray, sr: int) -> AudioSegment:
        """Helper to convert numpy float32 back to pydub segment."""
        if samples.size == 0: return AudioSegment.empty()
        peak = np.max(np.abs(samples)); if peak > 1.0: samples /= (peak + 1e-6)
        samples_int = (samples * 32767).astype(np.int16)
        if samples_int.shape[0] == 2:
            return AudioSegment(samples_int.T.flatten().tobytes(), frame_rate=sr, sample_width=2, channels=2)
        return AudioSegment(samples_int.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    def dj_stitch(self, track_paths: List[str], output_path: str, overlay_ms: int = 20000) -> Optional[str]:
        """Simplified sequential stitch for quick previews."""
        if not track_paths: return None
        combined = AudioSegment.from_file(track_paths[0]).set_frame_rate(self.sr).set_channels(2)
        for next_p in track_paths[1:]:
            next_s = AudioSegment.from_file(next_p).set_frame_rate(self.sr).set_channels(2)
            combined = combined.append(next_s, crossfade=min(len(combined)//3, len(next_s)//3, overlay_ms))
        combined.export(output_path, format="mp3", bitrate="320k")
        return output_path

    def render_timeline(self, segments: List[Dict[str, Any]], output_path: str, target_bpm: Optional[float] = None, mutes: Optional[List[bool]] = None, solos: Optional[List[bool]] = None, progress_cb: Optional[Callable[[int], None]] = None, time_range: Optional[Tuple[int, int]] = None) -> Optional[str]:
        t_bpm = target_bpm or AppConfig.DEFAULT_BPM
        return self._render_internal(segments, output_path, t_bpm, mutes, solos, progress_cb, time_range)

    def render_single_segment(self, segment_dict: Dict[str, Any], output_path: str, target_bpm: Optional[float] = None) -> Optional[str]:
        """High-speed single segment render for real-time auditioning."""
        t_bpm = target_bpm or AppConfig.DEFAULT_BPM
        res = _process_single_segment(segment_dict, 0, t_bpm, self.sr, None)
        if res:
            self.numpy_to_segment(res['samples'], self.sr).export(output_path, format="mp3", bitrate="192k")
            return output_path
        return None

    def render_stems(self, segments: List[Dict[str, Any]], output_folder: str, target_bpm: Optional[float] = None, progress_cb: Optional[Callable[[int], None]] = None, time_range: Optional[Tuple[int, int]] = None) -> List[str]:
        t_bpm = target_bpm or AppConfig.DEFAULT_BPM
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        lanes: Dict[int, List[Dict[str, Any]]] = {}
        for s in segments:
            l = int(s.get('lane', 0)); if l not in lanes: lanes[l] = []
            lanes[l].append(s)
        stem_paths = []; global_processed = 0
        for lane_id, lane_segs in lanes.items():
            path = os.path.join(output_folder, f"lane_{lane_id+1}.mp3")
            def stem_cb(count: int, cur_total: int = global_processed):
                if progress_cb: progress_cb(cur_total + count)
            self._render_internal(lane_segs, path, t_bpm, progress_cb=stem_cb, time_range=time_range)
            global_processed += len(lane_segs); stem_paths.append(path)
        return stem_paths

    def _apply_sidechain(self, target_samples: np.ndarray, source_samples: np.ndarray, amount: float = 0.8) -> np.ndarray:
        """Fake Sidechain."""
        f_len, h_len = 1024, 512
        rms = np.array([np.sqrt(np.mean(source_samples[:, i:i+f_len]**2)) for i in range(0, source_samples.shape[1], h_len)])
        env = np.repeat(rms, h_len)[:source_samples.shape[1]]
        if len(env) < source_samples.shape[1]: env = np.pad(env, (0, source_samples.shape[1] - len(env)))
        if len(env) > 0:
            mv = np.max(env); if mv > 0: env /= mv
        duck = np.clip(1.0 - (env * amount), 0.2, 1.0); min_l = min(target_samples.shape[1], len(duck))
        target_samples[:, :min_l] *= duck[:min_l]
        return target_samples

    def _apply_spectral_ducking(self, target_samples: np.ndarray, sr: int, low_cut: float = 300, high_cut: float = 12000) -> np.ndarray:
        return Pedalboard([HighpassFilter(cutoff_frequency_hz=low_cut), LowpassFilter(cutoff_frequency_hz=high_cut)])(target_samples, sr)

    def _render_internal(self, segments: List[Dict[str, Any]], output_path: str, target_bpm: float = 124.0, mutes: Optional[List[bool]] = None, solos: Optional[List[bool]] = None, progress_cb: Optional[Callable[[int], None]] = None, time_range: Optional[Tuple[int, int]] = None) -> Optional[str]:
        if not segments: return None
        range_start = time_range[0] if time_range else 0; range_end = time_range[1] if time_range else 0
        active_segments = []; any_solo = any(solos) if solos else False
        for s in segments:
            if time_range and (s['start_ms'] + s['duration_ms'] <= range_start or s['start_ms'] >= range_end): continue
            l = int(s.get('lane', 0)); is_muted = mutes[l] if mutes and l < len(mutes) else False; is_soloed = solos[l] if solos and l < len(solos) else False
            if any_solo:
                if is_soloed: active_segments.append(s)
            elif not is_muted: active_segments.append(s)
        if not active_segments:
            dur = (range_end - range_start) if time_range else 1000; silence = np.zeros((2, int(self.sr * max(1000, dur) / 1000.0)), dtype=np.float32)
            self.numpy_to_segment(silence, self.sr).export(output_path, format="mp3"); return output_path
        total_dur_ms = (range_end - range_start) if time_range else (max(s['start_ms'] + s['duration_ms'] for s in active_segments) + 2000)
        master_samples = np.zeros((2, int(self.sr * total_dur_ms / 1000.0)), dtype=np.float32)
        processed_data = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_process_single_segment, s, i, target_bpm, self.sr, time_range) for i, s in enumerate(active_segments)]
            completed = 0
            for f in as_completed(futures):
                res = f.result(); if res: processed_data.append(res)
                completed += 1; if progress_cb: progress_cb(completed)
        for current in processed_data:
            samples = current['samples']; start = current['start_idx']; end = start + samples.shape[1]
            if not current['is_primary']:
                for other in processed_data:
                    if (other is current) or not other['is_primary']: continue
                    o_start, o_end = other['start_idx'], other['start_idx'] + other['samples'].shape[1]
                    ov_start, ov_end = max(start, o_start), min(end, o_end)
                    if ov_start < ov_end:
                        samples = self._apply_spectral_ducking(samples, self.sr)
                        src_seg = other['samples'][:, ov_start - o_start : ov_end - o_start]; tgt_seg = samples[:, ov_start - start : ov_end - start]
                        is_vocal = (other.get('vocal_energy') or 0.0) > 0.2; base_duck = 0.9 if is_vocal else (0.85 if current['is_ambient'] else 0.7)
                        depth = current.get('ducking_depth') or 0.7; final_duck = base_duck * (depth / 0.7)
                        dl, dm, dh = current.get('duck_low', 1.0), current.get('duck_mid', 1.0), current.get('duck_high', 1.0)
                        if dl < 0.95: tgt_seg = Pedalboard([HighpassFilter(300 * (1.0 - dl))])(tgt_seg, self.sr)
                        if dh < 0.95: tgt_seg = Pedalboard([LowpassFilter(20000 - (15000 * (1.0 - dh)))])(tgt_seg, self.sr)
                        if dm < 0.95: final_duck *= (dm * 1.2)
                        samples[:, ov_start - start : ov_end - start] = self._apply_sidechain(tgt_seg, src_seg, amount=min(0.95, final_duck))
                        break
            r_end = min(master_samples.shape[1], end); r_len = r_end - start
            if r_len > 0: master_samples[:, start:r_end] += samples[:, :r_len]
        final_y = Pedalboard([Compressor(threshold_db=-14, ratio=2.5), Limiter(threshold_db=-0.1)])(master_samples, self.sr)
        self.numpy_to_segment(final_y, self.sr).export(output_path, format="mp3", bitrate="320k"); return output_path
