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

    def generate_hyper_mix(self, output_path="hyper_automated_mix.mp3", target_bpm=124, seed_track=None):
        """
        Creates a high-complexity 5-lane arrangement using micro-chopping, 
        sectional pitch shifting, and automated filter carving.
        """
        conn = self.dm.get_conn()
        conn.row_factory = lambda cursor, row: {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks")
        all_tracks = cursor.fetchall()
        conn.close()

        if len(all_tracks) < 8:
            print("Need at least 8 tracks for a Hyper Mix.")
            return

        # 1. Component Selection (Categorize by Vibe/Energy)
        # Sort by energy to find 'Drums/Foundation' vs 'Melodic/Atmosphere'
        all_tracks.sort(key=lambda x: x.get('energy', 0), reverse=True)
        drums = all_tracks[:3]
        others = all_tracks[3:]
        
        # 2. Arrange into Structural Blocks (Total ~160s)
        # Block structure: Intro (16s), Verse 1 (32s), Build (16s), Drop (32s), Verse 2 (32s), Outro (32s)
        blocks = [
            {'name': 'Intro', 'dur': 16000},
            {'name': 'Verse 1', 'dur': 32000},
            {'name': 'Build', 'dur': 16000},
            {'name': 'Drop', 'dur': 32000},
            {'name': 'Verse 2', 'dur': 32000},
            {'name': 'Outro', 'dur': 32000}
        ]
        
        segments = []
        current_ms = 0
        
        # Pick main components for the whole journey
        main_drum = drums[0]
        bass_track = next((t for t in others if t['harmonic_key'] == main_drum['harmonic_key']), others[0])
        melodic_leads = others[1:5]
        fx_tracks = others[5:8]

        print(f"Synthesizing Hyper-Mix structural journey...")

        for block in blocks:
            b_name = block['name']
            b_dur = block['dur']
            
            # --- LANE 0: Rhythmic Foundation (THE HEART - Never stops) ---
            # Extend foundation slightly into next block for seamless crossfade
            segments.append({
                'id': main_drum['id'], 'filename': main_drum['filename'], 'file_path': main_drum['file_path'], 'bpm': main_drum['bpm'], 'harmonic_key': main_drum['harmonic_key'],
                'start_ms': current_ms, 'duration_ms': b_dur + 2000, 'offset_ms': (main_drum.get('loop_start') or 0)*1000,
                'volume': 1.0 if b_name == 'Drop' else 0.8, 'is_primary': True, 'lane': 0,
                'fade_in_ms': 2000 if blocks.index(block) == 0 else 4000, 
                'fade_out_ms': 4000
            })

            # --- LANE 1: Harmonic Body (Bass) ---
            if b_name in ['Verse 1', 'Drop', 'Verse 2']:
                segments.append({
                    'id': bass_track['id'], 'filename': bass_track['filename'], 'file_path': bass_track['file_path'], 'bpm': bass_track['bpm'], 'harmonic_key': bass_track['harmonic_key'],
                    'start_ms': current_ms, 'duration_ms': b_dur, 'offset_ms': (bass_track.get('loop_start') or 0)*1000,
                    'volume': 0.9, 'is_primary': False, 'lane': 1,
                    'fade_in_ms': 3000, 'fade_out_ms': 3000
                })

            # --- LANE 2/3: Atmosphere & Melodic Layers ---
            lead = melodic_leads[blocks.index(block) % len(melodic_leads)]
            if b_name == 'Build':
                # Exponential Micro-Chopping: 4s -> 2s -> 1s -> 0.5s
                # To make it feel rhythmic and good with background
                sub_durs = [4000, 4000, 2000, 2000, 1000, 1000, 1000, 1000] # Total 16s
                sub_start = 0
                for idx, sd in enumerate(sub_durs):
                    segments.append({
                        'id': lead['id'], 'filename': f"CHOP {idx}", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                        'start_ms': current_ms + sub_start, 'duration_ms': sd, 'offset_ms': (lead.get('loop_start') or 0)*1000,
                        'volume': 0.7 + (idx/len(sub_durs) * 0.3),
                        'lane': 2, 'pitch_shift': int(idx/2), # Rising Pitch
                        'low_cut': 200 + (idx * 100), # Rising HPF
                        'fade_in_ms': 50, 'fade_out_ms': 50
                    })
                    sub_start += sd
            elif b_name != 'Intro':
                ps = -2 if b_name == 'Verse 2' else 0
                segments.append({
                    'id': lead['id'], 'filename': lead['filename'], 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                    'start_ms': current_ms, 'duration_ms': b_dur, 'offset_ms': (lead.get('loop_start') or 0)*1000,
                    'volume': 0.7, 'pan': -0.5 if blocks.index(block) % 2 == 0 else 0.5,
                    'is_primary': False, 'lane': 3, 'pitch_shift': ps,
                    'low_cut': 400, 'fade_in_ms': 4000, 'fade_out_ms': 4000
                })

            # --- LANE 4: Atmosphere Glue (The Floor) ---
            # Fill gaps with low-volume FX or atmosphere
            glue = fx_tracks[blocks.index(block) % len(fx_tracks)]
            segments.append({
                'id': glue['id'], 'filename': "ATMOS GLUE", 'file_path': glue['file_path'], 'bpm': glue['bpm'], 'harmonic_key': glue['harmonic_key'],
                'start_ms': current_ms, 'duration_ms': b_dur + 1000, 'offset_ms': 0,
                'volume': 0.4, 'lane': 4, 'low_cut': 600, 'high_cut': 8000, # Mid-focused glue
                'fade_in_ms': 5000, 'fade_out_ms': 5000
            })

            current_ms += b_dur

        print(f"Rendering Hyper-Mix with {len(segments)} intelligent segments...")
        self.renderer.render_timeline(segments, output_path, target_bpm=target_bpm)
        print(f"SUCCESS: Hyper-journey created at {os.path.abspath(output_path)}")
        return output_path

    def get_hyper_segments(self, seed_track=None):
        """Returns the segment data for a hyper-mix without rendering it."""
        conn = self.dm.get_conn()
        conn.row_factory = lambda cursor, row: {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks")
        all_tracks = cursor.fetchall()
        conn.close()

        if len(all_tracks) < 8: return []

        all_tracks.sort(key=lambda x: x.get('energy', 0), reverse=True)
        drums = all_tracks[:3]
        others = all_tracks[3:]
        
        blocks = [
            {'name': 'Intro', 'dur': 16000},
            {'name': 'Verse 1', 'dur': 32000},
            {'name': 'Build', 'dur': 16000},
            {'name': 'Drop', 'dur': 32000},
            {'name': 'Verse 2', 'dur': 32000},
            {'name': 'Outro', 'dur': 32000}
        ]
        
        segments = []
        current_ms = 0
        main_drum = drums[0]
        bass_track = next((t for t in others if t['harmonic_key'] == main_drum['harmonic_key']), others[0])
        melodic_leads = others[1:5]
        fx_tracks = others[5:8]

        for block in blocks:
            b_name = block['name']; b_dur = block['dur']
            if b_name != 'Intro':
                if b_name == 'Build':
                    for sub in range(0, b_dur, 4000):
                        segments.append({
                            'id': main_drum['id'], 'filename': main_drum['filename'], 'file_path': main_drum['file_path'], 'bpm': main_drum['bpm'], 'harmonic_key': main_drum['harmonic_key'],
                            'start_ms': current_ms + sub, 'duration_ms': 4000, 'offset_ms': (main_drum.get('loop_start') or 0)*1000,
                            'volume': 0.9, 'is_primary': True, 'lane': 0, 'low_cut': 100 + (sub/b_dur * 800), 'fade_in_ms': 100, 'fade_out_ms': 100
                        })
                else:
                    segments.append({
                        'id': main_drum['id'], 'filename': main_drum['filename'], 'file_path': main_drum['file_path'], 'bpm': main_drum['bpm'], 'harmonic_key': main_drum['harmonic_key'],
                        'start_ms': current_ms, 'duration_ms': b_dur, 'offset_ms': (main_drum.get('loop_start') or 0)*1000,
                        'volume': 1.0 if b_name == 'Drop' else 0.8, 'is_primary': True, 'lane': 0, 'fade_in_ms': 2000, 'fade_out_ms': 2000
                    })
            if b_name in ['Verse 1', 'Drop', 'Verse 2']:
                segments.append({
                    'id': bass_track['id'], 'filename': bass_track['filename'], 'file_path': bass_track['file_path'], 'bpm': bass_track['bpm'], 'harmonic_key': bass_track['harmonic_key'],
                    'start_ms': current_ms, 'duration_ms': b_dur, 'offset_ms': (bass_track.get('loop_start') or 0)*1000,
                    'volume': 0.9, 'is_primary': False, 'lane': 1, 'fade_in_ms': 3000, 'fade_out_ms': 3000
                })
            lead = melodic_leads[blocks.index(block) % len(melodic_leads)]
            if b_name != 'Build':
                ps = -2 if b_name == 'Verse 2' else 0
                segments.append({
                    'id': lead['id'], 'filename': lead['filename'], 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                    'start_ms': current_ms, 'duration_ms': b_dur, 'offset_ms': (lead.get('loop_start') or 0)*1000,
                    'volume': 0.7, 'pan': -0.5 if blocks.index(block) % 2 == 0 else 0.5, 'is_primary': False, 'lane': 2 + (blocks.index(block) % 2), 'pitch_shift': ps, 'low_cut': 400, 'fade_in_ms': 4000, 'fade_out_ms': 4000
                })
            if b_name == 'Build':
                fx = fx_tracks[0]
                segments.append({
                    'id': fx['id'], 'filename': fx['filename'], 'file_path': fx['file_path'], 'bpm': fx['bpm'], 'harmonic_key': fx['harmonic_key'],
                    'start_ms': current_ms, 'duration_ms': b_dur, 'offset_ms': 0, 'volume': 0.6, 'lane': 4, 'fade_in_ms': b_dur - 1000, 'fade_out_ms': 500
                })
            current_ms += b_dur
        return segments
