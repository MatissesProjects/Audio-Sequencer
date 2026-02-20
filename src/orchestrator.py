import sqlite3
import os
import soundfile as sf
from src.database import DataManager
from src.scoring import CompatibilityScorer
from src.processor import AudioProcessor
from src.renderer import FlowRenderer
from tqdm import tqdm

class FullMixOrchestrator:
    """Sequences and renders all tracks into a single continuous mix."""
    
    def __init__(self):
        self.dm = DataManager()
        self.scorer = CompatibilityScorer()
        self.processor = AudioProcessor()
        self.renderer = FlowRenderer()

    def sequence_all_tracks(self):
        """Finds the best path through all tracks using a greedy nearest-neighbor approach."""
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

        print(f"Sequencing {len(all_tracks)} tracks...")
        while unvisited:
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
            
            current = unvisited.pop(best_idx)
            sequence.append(current)
            
        return sequence

    def generate_full_mix(self, output_path="full_continuous_mix.mp3", target_bpm=120):
        """Processes all tracks to match a target BPM and stitches them."""
        sequence = self.sequence_all_tracks()
        if not sequence:
            print("No tracks to mix.")
            return

        processed_paths = []
        print("\nProcessing, Synchronizing, and Harmonic-Syncing tracks...")
        
        tmp_dir = "temp_segments"
        os.makedirs(tmp_dir, exist_ok=True)

        prev_key = None
        for i, track in enumerate(tqdm(sequence)):
            segment_path = os.path.join(tmp_dir, f"seg_{i}_{track['filename']}.wav")
            
            # 1. Harmonic Sync (Shift to match previous track if possible)
            pitch_steps = 0
            if prev_key and track['harmonic_key'] in self.scorer.CIRCLE_OF_FIFTHS:
                curr_pos = self.scorer.CIRCLE_OF_FIFTHS[track['harmonic_key']]
                prev_pos = self.scorer.CIRCLE_OF_FIFTHS[prev_key]
                pitch_steps = prev_pos - curr_pos
                if pitch_steps > 6: pitch_steps -= 12
                if pitch_steps < -6: pitch_steps += 12
                pitch_steps = max(-2, min(2, pitch_steps))

            # 2. Rhythmic Looping (Extend each track to 45s for long transitions)
            onsets = [float(x) for x in track['onsets_json'].split(',')] if track['onsets_json'] else []
            loop_path = os.path.join(tmp_dir, f"loop_{i}.wav")
            self.processor.loop_track(track['file_path'], 45.0, onsets, loop_path)

            # 3. Process: Time Stretch + Pitch Shift
            temp_y = self.processor.stretch_to_bpm(loop_path, track['bpm'], target_bpm)
            
            if pitch_steps != 0:
                import librosa
                y_shifted = librosa.effects.pitch_shift(temp_y, sr=self.processor.sr, n_steps=pitch_steps)
                sf.write(segment_path, y_shifted, self.processor.sr)
            else:
                sf.write(segment_path, temp_y, self.processor.sr)
                
            processed_paths.append(segment_path)
            # Cleanup intermediate loop
            if os.path.exists(loop_path): os.remove(loop_path)
            prev_key = track['harmonic_key']

        print(f"\nStitching {len(processed_paths)} tracks using DJ-style long overlays...")
        self.renderer.dj_stitch(processed_paths, output_path, overlay_ms=10000)

        # Cleanup
        for p in processed_paths:
            if os.path.exists(p): os.remove(p)
        if os.path.exists(tmp_dir):
            try: os.rmdir(tmp_dir)
            except: pass

        print(f"SUCCESS: Professional mix created at {os.path.abspath(output_path)}")
        return output_path
