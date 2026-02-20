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

    def render_timeline(self, segments, output_path, target_bpm=124):
        return self._render_internal(segments, output_path, target_bpm)

    def render_stems(self, segments, output_folder, target_bpm=124):
        """
        Exports each lane to a separate audio file.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        lanes = {}
        for s in segments:
            l = s.get('lane', 0)
            if l not in lanes: lanes[l] = []
            lanes[l].append(s)
            
        stem_paths = []
        for lane_id, lane_segs in lanes.items():
            path = os.path.join(output_folder, f"lane_{lane_id+1}.mp3")
            print(f"Exporting Stem: Lane {lane_id+1}...")
            self._render_internal(lane_segs, path, target_bpm)
            stem_paths.append(path)
            
        return stem_paths

    def _render_internal(self, segments, output_path, target_bpm=124):
        """
        Internal rendering logic shared by full mix and stems.
        """
        if not segments:
            return None
            
        # Calculate master buffer size
        total_duration_ms = max(s['start_ms'] + s['duration_ms'] for s in segments) + 2000 
        master_samples = np.zeros((2, int(self.sr * total_duration_ms / 1000.0)), dtype=np.float32)

        print(f"Sonic Rendering: {len(segments)} segments with Pro DSP...")

        processed_data = []

        # Phase 1: Process each segment individually
        for i, s in enumerate(tqdm(segments, desc="Processing Clips")):
            from src.processor import AudioProcessor
            proc = AudioProcessor()
            
            # 1. Loop/Trim to raw requirement
            required_raw_dur = (s['duration_ms'] + s['offset_ms']) / 1000.0
            tmp_loop = f"temp_render_{i}.wav"
            proc.loop_track(s['file_path'], required_raw_dur + 1.0, [], tmp_loop)
            
            # 2. Stretch & Pitch
            y_sync = proc.stretch_to_bpm(tmp_loop, s['bpm'], target_bpm)
            ps = s.get('pitch_shift', 0)
            if ps != 0:
                import librosa
                y_sync = librosa.effects.pitch_shift(y_sync, sr=self.sr, n_steps=ps)
            
            if os.path.exists(tmp_loop): os.remove(tmp_loop)

            # 3. Trim to Offset and Duration
            # Convert to AudioSegment temporarily for easy trimming
            seg_audio = self.numpy_to_segment(y_sync, self.sr)
            seg_audio = seg_audio[s['offset_ms'] : s['offset_ms'] + s['duration_ms']]
            
            # Convert to stereo numpy for final blending
            seg_np = self.segment_to_numpy(seg_audio)
            num_samples = seg_np.shape[1]

            # 4. Gain & S-Curve Envelopes
            vol_mult = s.get('volume', 1.0)
            fi_s = int(s.get('fade_in_ms', 2000) * self.sr / 1000.0)
            fo_s = int(s.get('fade_out_ms', 2000) * self.sr / 1000.0)
            
            envelope = np.ones(num_samples, dtype=np.float32)
            if fi_s > 0:
                t_in = np.linspace(0, 1, min(fi_s, num_samples))
                envelope[:len(t_in)] = 0.5 * (1 - np.cos(np.pi * t_in))
            if fo_s > 0:
                t_out = np.linspace(0, 1, min(fo_s, num_samples))
                envelope[-len(t_out):] *= 0.5 * (1 + np.cos(np.pi * t_out))
            
            seg_np *= (vol_mult * envelope)
            
            processed_data.append({
                'samples': seg_np,
                'start_idx': int(s['start_ms'] * self.sr / 1000.0),
                'is_primary': s.get('is_primary', False)
            })

        # Phase 2: Advanced Blending (Lead Focus Ducking)
        for i, current in enumerate(processed_data):
            samples = current['samples']
            start = current['start_idx']
            end = start + samples.shape[1]
            
            # Ducking check: if NOT primary, check if we overlap a primary
            if not current['is_primary']:
                for other in processed_data:
                    if other == current or not other['is_primary']: continue
                    o_start = other['start_idx']
                    o_end = o_start + other['samples'].shape[1]
                    
                    # Detect overlap
                    if max(start, o_start) < min(end, o_end):
                        # Apply subtle -4dB ducking for the whole overlapping clip
                        samples *= 0.63 # approx -4dB
                        break
            
            # Add to master buffer
            master_samples[:, start:end] += samples

        # Phase 3: Master Bus (Compression & Limiting)
        print("Finalizing Master Bus...")
        master_bus = Pedalboard([
            Compressor(threshold_db=-14, ratio=2.5),
            Limiter(threshold_db=-0.1)
        ])
        
        final_y = master_bus(master_samples, self.sr)
        
        # Export
        final_seg = self.numpy_to_segment(final_y, self.sr)
        fmt = "wav" if output_path.lower().endswith(".wav") else "mp3"
        if fmt == "wav":
            final_seg.export(output_path, format="wav")
        else:
            final_seg.export(output_path, format="mp3", bitrate="320k")
        return output_path
