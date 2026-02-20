import sqlite3
import os
import soundfile as sf
import numpy as np
import librosa
from src.database import DataManager
from src.scoring import CompatibilityScorer
from src.processor import AudioProcessor
from src.renderer import FlowRenderer
from tqdm import tqdm

class FullMixOrchestrator:
    """Sequences and layers curated selections for maximum musical flow."""
    
    def __init__(self):
        self.dm = DataManager()
        self.scorer = CompatibilityScorer()
        self.processor = AudioProcessor()
        self.renderer = FlowRenderer()
        self.min_score_threshold = 55.0

    def find_curated_sequence(self, max_tracks=6):
        """Finds a high-compatibility path, aiming for a specific length."""
        conn = self.dm.get_conn()
        conn.row_factory = lambda cursor, row: {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks")
        all_tracks = cursor.fetchall()
        conn.close()

        if not all_tracks:
            return []

        unvisited = all_tracks.copy()
        current = unvisited.pop(0)
        sequence = [current]

        print(f"Finding the best flow from {len(all_tracks)} available clips...")
        
        while unvisited and len(sequence) < max_tracks:
            best_next = None
            best_score = -1
            best_idx = -1
            
            curr_emb = self.dm.get_embedding(current['clp_embedding_id']) if current['clp_embedding_id'] else None

            for i, candidate in enumerate(unvisited):
                cand_emb = self.dm.get_embedding(candidate['clp_embedding_id']) if candidate['clp_embedding_id'] else None
                score = self.scorer.get_total_score(current, candidate, curr_emb, cand_emb)['total']
                
                if score > best_score:
                    best_score = score
                    best_next = candidate
                    best_idx = i
            
            if best_score < self.min_score_threshold:
                break
                
            current = unvisited.pop(best_idx)
            sequence.append(current)
            
        return sequence

    def generate_full_mix(self, output_path="full_continuous_mix.mp3", target_bpm=124):
        """Processes a curated selection with dynamic durations."""
        sequence = self.find_curated_sequence()
        if not sequence:
            print("No tracks to mix.")
            return

        processed_paths = []
        print("\nProcessing curated clips...")
        
        tmp_dir = "temp_segments"
        os.makedirs(tmp_dir, exist_ok=True)

        prev_key = None
        for i, track in enumerate(tqdm(sequence)):
            segment_path = os.path.join(tmp_dir, f"seg_{i}_{track['filename']}.wav")
            
            pitch_steps = 0
            if prev_key and track['harmonic_key'] in self.scorer.CIRCLE_OF_FIFTHS:
                curr_pos = self.scorer.CIRCLE_OF_FIFTHS[track['harmonic_key']]
                prev_pos = self.scorer.CIRCLE_OF_FIFTHS[prev_key]
                pitch_steps = prev_pos - curr_pos
                if pitch_steps > 6: pitch_steps -= 12
                if pitch_steps < -6: pitch_steps += 12
                pitch_steps = max(-2, min(2, pitch_steps))

            duration = 16.0 if i % 2 == 0 else 24.0
            onsets = [float(x) for x in track['onsets_json'].split(',')] if track['onsets_json'] else []
            loop_path = os.path.join(tmp_dir, f"loop_{i}.wav")
            self.processor.loop_track(track['file_path'], duration, onsets, loop_path)

            temp_y = self.processor.stretch_to_bpm(loop_path, track['bpm'], target_bpm)
            
            if pitch_steps != 0:
                y_shifted = librosa.effects.pitch_shift(temp_y, sr=self.processor.sr, n_steps=pitch_steps)
                sf.write(segment_path, y_shifted, self.processor.sr)
            else:
                sf.write(segment_path, temp_y, self.processor.sr)
                
            processed_paths.append(segment_path)
            if os.path.exists(loop_path): os.remove(loop_path)
            prev_key = track['harmonic_key']

        print(f"\nStitching {len(processed_paths)} curated tracks into final journey...")
        self.renderer.dj_stitch(processed_paths, output_path, overlay_ms=8000)

        for p in processed_paths:
            if os.path.exists(p): os.remove(p)
        if os.path.exists(tmp_dir):
            try: os.rmdir(tmp_dir)
            except: pass

        print(f"SUCCESS: Curated journey created at {os.path.abspath(output_path)}")
        return output_path

    def generate_layered_journey(self, output_path="layered_journey_mix.mp3", target_bpm=124):
        """Creates a long foundation track with other clips layered as interludes."""
        conn = self.dm.get_conn()
        conn.row_factory = lambda cursor, row: {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks")
        all_tracks = cursor.fetchall()
        conn.close()

        if len(all_tracks) < 5:
            print("Need at least 5 tracks for a layered journey.")
            return

        foundation_track = all_tracks[0]
        found_emb = self.dm.get_embedding(foundation_track['clp_embedding_id'])
        
        candidates = all_tracks[1:]
        scored_candidates = []
        for c in candidates:
            c_emb = self.dm.get_embedding(c['clp_embedding_id'])
            score = self.scorer.get_total_score(foundation_track, c, found_emb, c_emb)['total']
            scored_candidates.append((score, c))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        interludes = [x[1] for x in scored_candidates[:4]]

        print(f"Creating a 120s layered journey based on '{foundation_track['filename']}'...")
        
        tmp_dir = "temp_layered"
        os.makedirs(tmp_dir, exist_ok=True)

        # Process Foundation
        foundation_loop = os.path.join(tmp_dir, "foundation_extended.wav")
        onsets = [float(x) for x in foundation_track['onsets_json'].split(',')] if foundation_track['onsets_json'] else []
        self.processor.loop_track(foundation_track['file_path'], 120.0, onsets, foundation_loop)
        foundation_sync = os.path.join(tmp_dir, "foundation_sync.wav")
        self.processor.stretch_to_bpm(foundation_loop, foundation_track['bpm'], target_bpm, foundation_sync)

        # Process Interludes
        layer_configs = []
        start_offsets = [20000, 45000, 70000, 95000] #ms
        
        temp_files = [foundation_loop, foundation_sync]
        for i, track in enumerate(interludes):
            interlude_sync = os.path.join(tmp_dir, f"interlude_{i}.wav")
            # Loop interludes to 20s
            i_onsets = [float(x) for x in track['onsets_json'].split(',')] if track['onsets_json'] else []
            i_loop = self.processor.loop_track(track['file_path'], 20.0, i_onsets)
            # Sync BPM
            i_sync = self.processor.stretch_to_bpm(track['file_path'], track['bpm'], target_bpm) # Use original for stretch to avoid double process
            # Actually, loop THEN stretch
            i_loop_path = os.path.join(tmp_dir, f"i_loop_{i}.wav")
            self.processor.loop_track(track['file_path'], 20.0, i_onsets, i_loop_path)
            self.processor.stretch_to_bpm(i_loop_path, track['bpm'], target_bpm, interlude_sync)
            
            layer_configs.append({'path': interlude_sync, 'start_ms': start_offsets[i], 'gain': -2.0})
            temp_files.extend([i_loop_path, interlude_sync])

        print("Rendering layered architecture...")
        self.renderer.layered_mix(foundation_sync, layer_configs, output_path)

        for f in temp_files:
            if os.path.exists(f): os.remove(f)
        if os.path.exists(tmp_dir):
            try: os.rmdir(tmp_dir)
            except: pass

        print(f"SUCCESS: Layered journey created at {os.path.abspath(output_path)}")
        return output_path
