import sqlite3
import os
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

    def generate_full_mix(self, output_path="full_continuous_mix.wav", target_bpm=120):
        """Processes all tracks to match a target BPM and stitches them."""
        sequence = self.sequence_all_tracks()
        if not sequence:
            print("No tracks to mix.")
            return

        processed_paths = []
        print("\nProcessing and Synchronizing tracks...")
        
        tmp_dir = "temp_segments"
        os.makedirs(tmp_dir, exist_ok=True)

        for i, track in enumerate(tqdm(sequence)):
            segment_path = os.path.join(tmp_dir, f"seg_{i}_{track['filename']}.wav")
            # Convert to wav if needed and stretch
            self.processor.stretch_to_bpm(track['file_path'], track['bpm'], target_bpm, segment_path)
            processed_paths.append(segment_path)

        print(f"\nStitching {len(processed_paths)} tracks into final mix...")
        self.renderer.stitch_tracks(processed_paths, output_path, crossfade_ms=3000)

        # Cleanup
        for p in processed_paths:
            if os.path.exists(p): os.remove(p)
        if os.path.exists(tmp_dir):
            try: os.rmdir(tmp_dir)
            except: pass

        print(f"SUCCESS: Final mix created at {os.path.abspath(output_path)}")
        return output_path
