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

    def find_curated_sequence(self, max_tracks=6, seed_track=None):
        """Finds a high-compatibility path, starting from a seed if provided."""
        conn = self.dm.get_conn()
        conn.row_factory = lambda cursor, row: {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks")
        all_tracks = cursor.fetchall()
        conn.close()

        if not all_tracks:
            return []

        unvisited = all_tracks.copy()
        
        # 1. Selection logic for starting track
        if seed_track:
            # Find the seed in unvisited and move it to start
            current = next((t for t in unvisited if t['id'] == seed_track['id']), unvisited[0])
            unvisited.remove(current)
        else:
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

    def generate_full_mix(self, output_path="full_continuous_mix.mp3", target_bpm=124, seed_track=None):
        """Processes a curated selection with dynamic durations and advanced mixing."""
        sequence = self.find_curated_sequence(seed_track=seed_track)
        if not sequence:
            print("No tracks to mix.")
            return

        print("\nOrchestrating high-fidelity curated sequence...")
        
        segments = []
        current_ms = 0
        for i, track in enumerate(sequence):
            # Overlap logic: 8s crossfade
            duration_ms = 30000 if i % 2 == 0 else 20000
            start_ms = current_ms
            if i > 0: start_ms -= 8000
            
            loop_start = (track.get('loop_start') or 0) * 1000.0
            
            segments.append({
                'file_path': track['file_path'],
                'bpm': track['bpm'],
                'start_ms': start_ms,
                'duration_ms': duration_ms,
                'offset_ms': loop_start,
                'volume': 1.0,
                'pan': -0.2 if i % 2 == 0 else 0.2, # Subtle wide field
                'is_primary': True, # All sequential tracks are primary for focus
                'lane': i % 2,
                'fade_in_ms': 4000,
                'fade_out_ms': 4000
            })
            current_ms = start_ms + duration_ms

        print(f"Rendering {len(segments)} segments with professional signal chain...")
        self.renderer.render_timeline(segments, output_path, target_bpm=target_bpm)
        print(f"SUCCESS: Curated journey created at {os.path.abspath(output_path)}")
        return output_path

    def generate_layered_journey(self, output_path="layered_journey_mix.mp3", target_bpm=124, seed_track=None):
        """Creates a long foundation track with other clips layered as interludes with auto-panning."""
        conn = self.dm.get_conn()
        conn.row_factory = lambda cursor, row: {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks")
        all_tracks = cursor.fetchall()
        conn.close()

        if len(all_tracks) < 5:
            print("Need at least 5 tracks for a layered journey.")
            return

        # 1. Foundation Selection
        if seed_track:
            foundation_track = next((t for t in all_tracks if t['id'] == seed_track['id']), all_tracks[0])
        else:
            foundation_track = all_tracks[0]
        
        found_emb = self.dm.get_embedding(foundation_track['clp_embedding_id']) if foundation_track['clp_embedding_id'] else None
        
        candidates = [t for t in all_tracks if t['id'] != foundation_track['id']]
        scored_candidates = []
        for c in candidates:
            c_emb = self.dm.get_embedding(c['clp_embedding_id']) if c['clp_embedding_id'] else None
            score = self.scorer.get_total_score(foundation_track, c, found_emb, c_emb)['total']
            scored_candidates.append((score, c))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        interludes = [x[1] for x in scored_candidates[:6]] # Use more for variety

        print(f"Creating a professional 120s layered journey based on '{foundation_track['filename']}'...")
        
        # We will now use the new Timeline Renderer logic for the orchestrator too
        segments = []
        
        # Add Foundation (The "Lead")
        # Use Smart Loop if available
        f_start_offset = (foundation_track.get('loop_start') or 0) * 1000.0
        segments.append({
            'file_path': foundation_track['file_path'],
            'bpm': foundation_track['bpm'],
            'start_ms': 0,
            'duration_ms': 120000,
            'offset_ms': f_start_offset,
            'volume': 1.0,
            'pan': 0.0,
            'is_primary': True,
            'lane': 0,
            'fade_in_ms': 5000,
            'fade_out_ms': 5000
        })

        # Add Interludes with Auto-Panning
        # Strategy: Alternate L/R and varying start times
        start_times = [10000, 25000, 45000, 65000, 85000, 100000]
        pans = [-0.6, 0.6, -0.4, 0.4, -0.7, 0.7]
        
        for i, track in enumerate(interludes):
            if i >= len(start_times): break
            
            i_loop_start = (track.get('loop_start') or 0) * 1000.0
            segments.append({
                'file_path': track['file_path'],
                'bpm': track['bpm'],
                'start_ms': start_times[i],
                'duration_ms': 20000,
                'offset_ms': i_loop_start,
                'volume': 0.8,
                'pan': pans[i],
                'is_primary': False,
                'lane': (i % 3) + 1,
                'fade_in_ms': 4000,
                'fade_out_ms': 4000
            })

        print("Automated Mixing with Sidechain & Spectral Ducking...")
        self.renderer.render_timeline(segments, output_path, target_bpm=target_bpm)
        print(f"SUCCESS: Automated journey created at {os.path.abspath(output_path)}")
        return output_path
