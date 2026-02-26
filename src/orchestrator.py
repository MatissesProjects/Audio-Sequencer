import sqlite3
import os
import soundfile as sf
import numpy as np
import librosa
import random
from src.database import DataManager
from src.scoring import CompatibilityScorer
from src.processor import AudioProcessor
from src.renderer import FlowRenderer
from src.generator import TransitionGenerator
from tqdm import tqdm

class FullMixOrchestrator:
    """Sequences and layers curated selections for maximum musical flow."""

    def __init__(self):
        self.dm = DataManager()
        self.scorer = CompatibilityScorer()
        self.processor = AudioProcessor()
        self.renderer = FlowRenderer()
        self.generator = TransitionGenerator()
        self.min_score_threshold = 55.0
        self.lane_count = 20

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

    def generate_hyper_mix(self, output_path="hyper_automated_mix.mp3", target_bpm=124, seed_track=None):
        """Direct rendering of a Hyper-Mix journey."""
        segments = self.get_hyper_segments(seed_track=seed_track)
        if not segments:
            print("Failed to generate hyper-segments.")
            return None

        print(f"Rendering Hyper-Mix with {len(segments)} intelligent segments...")
        self.renderer.render_timeline(segments, output_path, target_bpm=target_bpm)
        print(f"SUCCESS: Hyper-journey created at {os.path.abspath(output_path)}")
        return output_path

    def get_hyper_segments(self, seed_track=None, start_time_ms=0, depth=0, force_ending=False):
        """Returns organized segment data for a hyper-mix."""
        conn = self.dm.get_conn()
        conn.row_factory = lambda cursor, row: {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks")
        all_tracks = cursor.fetchall()
        conn.close()

        if len(all_tracks) < 5: return []

        print(f"[AI] Orchestrating Hyper-Mix Depth {depth} (Pool: {len(all_tracks)} clips)")

        # --- INTELLIGENT CATEGORIZATION ---
        # 1. Vocal Tracks (Explicitly check for stems and energy or transcribed lyrics)
        vocal_pool = [t for t in all_tracks if ((t.get('vocal_energy') or 0) > 0.02 or t.get('vocal_lyrics')) and t.get('stems_path')]
        
        # Log top 5 vocal energy candidates for debugging
        all_tracks.sort(key=lambda x: x.get('vocal_energy') or 0, reverse=True)
        print("[AI] Top Vocal Candidates Energy:")
        for t in all_tracks[:5]:
            print(f"  - {t['filename']}: {(t.get('vocal_energy') or 0):.4f} (Lyrics: {'Yes' if t.get('vocal_lyrics') else 'No'})")

        # 2. Percussion (High onset density, but exclude vocal tracks)
        remaining = [t for t in all_tracks if t not in vocal_pool]
        remaining.sort(key=lambda x: x.get('onset_density') or 0, reverse=True)
        drums = remaining[:max(1, int(len(remaining)*0.4))]
        
        # 3. Others (Melodic/FX)
        others = [t for t in remaining if t not in drums]
        if not others: others = all_tracks # Safety fallback

        print(f"[AI] Pool Breakdown -> Vocals: {len(vocal_pool)} | Drums: {len(drums)} | Melodic: {len(others)}")

        def rnd_dur(base): return base + (random.randint(-1, 1) * 4000)

        # Get dynamic structure from generator
        if force_ending:
            blocks = [
                {'name': 'Connect', 'dur': 8000},
                {'name': 'Pre-Finale Build', 'dur': 16000},
                {'name': 'Grand Finale', 'dur': 32000},
                {'name': 'Extended Outro', 'dur': 24000}
            ]
        else:
            blocks = self.generator.get_journey_structure(depth=depth)
            # If not forcing ending and not depth 0, ensure it doesn't end with a definitive "Outro"
            if depth > 0:
                for b in blocks:
                    if 'outro' in b['name'].lower():
                        b['name'] = 'Transition'
                        b['dur'] = 8000

        segments = []
        current_ms = start_time_ms
        target_bpm = 124.0
        if seed_track:
            target_bpm = seed_track.get('bpm', 124.0)

        # Dynamic Pool Selection based on depth (DJ-style swapping)
        d_idx = int(depth)
        
        # 1. Select Melodic/Vocal Pools using Smart Scoring against the seed
        scored_others = []
        for t in others:
            # Score against seed (or random rotation if no seed)
            if seed_track:
                # We don't have embeddings here easily, so we use BPM/Harmonic/Energy/Groove
                score = self.scorer.get_total_score(seed_track, t)['total']
            else:
                score = random.random() * 100
            scored_others.append((score, t))
        scored_others.sort(key=lambda x: x[0], reverse=True)
        
        # Take top compatible tracks for this depth section
        pool_start = (d_idx * 6) % max(1, len(scored_others))
        melodic_leads = [x[1] for x in scored_others[pool_start : pool_start+10]]
        if not melodic_leads: melodic_leads = [x[1] for x in scored_others[:10]]
        
        fx_tracks = melodic_leads[6:10] if len(melodic_leads) >= 10 else melodic_leads[:4]

        # 2. Select Vocal Pool using Smart Scoring
        scored_vocals = []
        for t in vocal_pool:
            if seed_track:
                score = self.scorer.get_total_score(seed_track, t)['total']
            else:
                score = random.random() * 100
            scored_vocals.append((score, t))
        scored_vocals.sort(key=lambda x: x[0], reverse=True)
        
        v_start = (d_idx * 3) % max(1, len(scored_vocals))
        rotated_vocals = [x[1] for x in scored_vocals[v_start : v_start+10]]
        if not rotated_vocals and vocal_pool: rotated_vocals = [x[1] for x in scored_vocals[:10]]

        main_drum = drums[d_idx % len(drums)] if drums else all_tracks[0]
        if seed_track and depth == 0: # Only seed key on start
            sk = seed_track.get('harmonic_key') or seed_track.get('key')
            if sk:
                comp_drums = [t for t in drums if self.scorer.calculate_harmonic_score(sk, t['harmonic_key']) >= 80]
                if comp_drums: main_drum = random.choice(comp_drums)

        bass_track = random.choice([t for t in melodic_leads if t['harmonic_key'] == main_drum['harmonic_key']] or [melodic_leads[0]])
        
        used_vocal_ids = []

        os.makedirs("generated_assets", exist_ok=True)
        cloud_path = os.path.abspath(f"generated_assets/spectral_pad_{main_drum['id']}_d{depth}_{random.randint(0,99)}.wav")
        if not os.path.exists(cloud_path):
            try: 
                source_p = seed_track['file_path'] if (seed_track and depth==0) else random.choice(melodic_leads)['file_path']
                # Use high-fidelity remote pad if possible
                self.processor.generate_spectral_pad_remote(source_p, cloud_path, duration=20.0)
            except: cloud_path = main_drum['file_path']

        overlap = 4000

        def find_free_lane(start, dur, role="melodic", preferred=None):
            neighborhoods = {
                "percussion": [0, 1, 8, 12, 16],
                "bass": [2, 3, 9, 13, 17],
                "melodic": [4, 5, 10, 14, 18],
                "atmosphere": [6, 7, 11, 15, 19]
            }
            candidates = neighborhoods.get(role, neighborhoods["melodic"])
            busy_lanes = set()
            for s in segments:
                if max(start, s['start_ms']) < min(start + dur, s['start_ms'] + s['duration_ms']):
                    busy_lanes.add(s['lane'])
            if preferred is not None and preferred not in busy_lanes and preferred < self.lane_count:
                return preferred
            for l in candidates:
                if l < self.lane_count and l not in busy_lanes: return l
            for l in range(self.lane_count):
                if l not in busy_lanes: return l
            return preferred or 0

        def get_best_offset(track, block_type):
            """Uses MIR section data to find the most appropriate start point."""
            # Default to stored loop_start
            default_offset = (track.get('loop_start') or 0) * 1000.0
            
            s_json = track.get('sections_json')
            if not s_json: return default_offset
            
            try:
                import json
                sections = json.loads(s_json)
                if not sections: return default_offset
                
                # Try to match block type to detected section
                target = "Verse"
                bt_l = block_type.lower()
                if any(k in bt_l for k in ['drop', 'climax', 'finale']): target = "Drop"
                elif any(k in bt_l for k in ['build', 'riser']): target = "Build"
                elif any(k in bt_l for k in ['intro', 'start']): target = "Intro"
                
                # Find first matching section
                for s in sections:
                    if s['label'] == target:
                        print(f"[AI] Section Match! Track {track['filename']} using {target} at {s['start']:.1f}s")
                        return s['start'] * 1000.0
            except:
                pass
            return default_offset

        for idx, block in enumerate(blocks):
            b_name = block['name']; b_dur = block['dur']
            
            # More flexible block detection based on keywords
            is_drop = any(k in b_name.lower() for k in ['drop', 'finale', 'climax'])
            is_build = any(k in b_name.lower() for k in ['build', 'riser', 'tension'])
            is_intro = any(k in b_name.lower() for k in ['intro', 'connect', 'start'])
            is_outro = any(k in b_name.lower() for k in ['outro', 'fade', 'end'])
            is_transition = any(k in b_name.lower() for k in ['transition', 'bridge'])

            # --- PERCUSSION (Lanes 0-1) ---
            f_start = current_ms
            p_keys = {}
            if is_intro:
                # Start subtle, then bring it in
                p_keys['drum_vol'] = [(0, 0.0), (8000, 0.5), (16000, 1.0)]
            elif is_outro:
                # Fade out drums first
                p_keys['drum_vol'] = [(0, 1.0), (b_dur/2, 0.0)]

            lane = find_free_lane(f_start, b_dur + overlap, role="percussion", preferred=0)
            segments.append({
                'id': main_drum['id'], 'filename': main_drum['filename'], 'file_path': main_drum['file_path'], 'bpm': main_drum['bpm'], 'harmonic_key': main_drum['harmonic_key'],
                'start_ms': f_start, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(main_drum, b_name), 'stems_path': main_drum.get('stems_path'),
                'vocal_lyrics': main_drum.get('vocal_lyrics'), 'vocal_gender': main_drum.get('vocal_gender'),
                'volume': 1.0 if is_drop else 0.8, 'is_primary': True, 'lane': lane,
                'fade_in_ms': 1000 if not is_intro else 4000, 'fade_out_ms': 4000,
                'drum_vol': 1.3 if is_drop else 1.0, 'instr_vol': 0.3 if is_drop else 0.6, 
                'ducking_depth': 0.3, 'keyframes': p_keys
            })
            # --- BASS (Lanes 2-3) ---
            b_start = current_ms
            bass_keys = {}
            if is_intro:
                # Filter sweep on bass
                bass_keys['low_cut'] = [(0, 1000), (8000, 200), (16000, 20)]
                bass_keys['bass_vol'] = [(0, 0.5), (16000, 1.0)]
            elif is_outro:
                # Fade out bass after drums
                bass_keys['bass_vol'] = [(b_dur/2, 1.0), (b_dur, 0.0)]
                bass_keys['low_cut'] = [(b_dur/2, 20), (b_dur, 800)]

            lane = find_free_lane(b_start, b_dur + overlap, role="bass", preferred=2)
            segments.append({
                'id': bass_track['id'], 'filename': bass_track['filename'], 'file_path': bass_track['file_path'], 'bpm': bass_track['bpm'], 'harmonic_key': bass_track['harmonic_key'],
                'start_ms': b_start, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(bass_track, b_name), 'stems_path': bass_track.get('stems_path'),
                'vocal_lyrics': bass_track.get('vocal_lyrics'), 'vocal_gender': bass_track.get('vocal_gender'),
                'volume': 0.8, 'is_primary': False, 'lane': lane, 'fade_in_ms': 3000, 'fade_out_ms': 3000,
                'instr_vol': 1.1 if is_drop else 0.8, 'vocal_vol': 0.0, 'bass_vol': 1.2,
                'ducking_depth': 0.9 if is_drop else 0.7, 'duck_high': 0.4, 'low_cut': 20,
                'keyframes': bass_keys
            })

            # --- ATMOSPHERE / AMBIENT (Lanes 6-7) ---
            if is_intro or is_outro or is_transition:
                lane = find_free_lane(current_ms, b_dur + 4000, role="atmosphere", preferred=6)
                segments.append({
                    'id': -2, 'filename': "NEURAL CLOUD", 'file_path': cloud_path, 'bpm': 120, 'harmonic_key': 'N/A',
                    'start_ms': current_ms, 'duration_ms': b_dur + 4000, 'offset_ms': 0, 'stems_path': None,
                    'volume': 0.25 if is_intro else 0.35, 'lane': lane, 'fade_in_ms': 5000, 'fade_out_ms': 5000,
                    'is_ambient': True, 'ducking_depth': 0.98, 'reverb': 0.9, 'low_cut': 600, 'duck_low': 0.1, 'duck_mid': 0.4,
                    'keyframes': {
                        'volume': [(0, 0.0), (4000, 1.0), (b_dur, 1.0), (b_dur + 4000, 0.0)]
                    }
                })
            
            # --- MELODIC (Lanes 4-5) ---
            if is_intro or is_outro:
                # Filtered version for definitive edges
                lead = melodic_leads[0] if is_intro else melodic_leads[-1]
                m_keys = {}
                if is_intro:
                    m_keys['high_cut'] = [(0, 500), (b_dur*0.6, 3000), (b_dur, 18000)]
                    m_keys['instr_vol'] = [(0, 0.4), (b_dur, 0.8)]
                else:
                    m_keys['high_cut'] = [(0, 18000), (b_dur, 600)]
                    m_keys['instr_vol'] = [(0, 0.8), (b_dur, 0.0)]
                
                lane = find_free_lane(current_ms, b_dur + overlap, role="melodic")
                segments.append({
                    'id': lead['id'], 'filename': f"{lead['filename']} ({'INTRO' if is_intro else 'OUTRO'})", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                    'start_ms': current_ms, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(lead, b_name), 'stems_path': lead.get('stems_path'),
                    'vocal_lyrics': lead.get('vocal_lyrics'), 'vocal_gender': lead.get('vocal_gender'),
                    'volume': 0.6, 'lane': lane, 'fade_in_ms': 4000, 'fade_out_ms': 4000,
                    'vocal_vol': 0.0, 'instr_vol': 0.8, 'reverb': 0.6, 'keyframes': m_keys
                })
            else:
                # Regular melodic blocks (Verses, Bridges, Drops)
                
                # Vocal Selection Strategy: Try to pick a new vocal track for each block
                available_vocals = [t for t in rotated_vocals if t['id'] not in used_vocal_ids]
                if not available_vocals and rotated_vocals:
                    # Reset memory if we run out of unique vocalists
                    used_vocal_ids = []
                    available_vocals = rotated_vocals
                
                if available_vocals:
                    lead = random.choice(available_vocals)
                    used_vocal_ids.append(lead['id'])
                    print(f"[AI] Block '{b_name}' -> Selected Vocal: {lead['filename']}")
                else:
                    # Fallback to melodic pool if no vocals at all in library
                    lead = melodic_leads[idx % len(melodic_leads)]
                    print(f"[AI] Block '{b_name}' -> No vocals found, using melodic: {lead['filename']}")

                if is_build:
                    sub_durs = [4000, 4000, 2000, 2000, 1000, 1000, 1000, 1000]; sub_start = 0
                    for c_idx, sd in enumerate(sub_durs):
                        lane = find_free_lane(current_ms + sub_start, sd + 200, role="melodic")
                        segments.append({
                            'id': lead['id'], 'filename': f"CHOP {c_idx}", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                            'start_ms': current_ms + sub_start, 'duration_ms': sd + 200, 'offset_ms': (lead.get('loop_start') or 0)*1000, 'stems_path': lead.get('stems_path'),
                            'volume': 0.7 + (c_idx/len(sub_durs) * 0.3), 'lane': lane, 'pitch_shift': int(c_idx/2), 'low_cut': 200 + (c_idx * 100), 'fade_in_ms': 50, 'fade_out_ms': 50,
                            'harmony_level': 0.3 + (c_idx/len(sub_durs) * 0.5), 'vocal_vol': 1.2, 'instr_vol': 0.3, 'ducking_depth': 0.4, 'duck_low': 0.3,
                            'keyframes': {
                                'volume': [(0, 0.8), (sd, 1.0)]
                            }
                        })
                        sub_start += sd
                else:
                    ps = -2 if 'verse 2' in b_name.lower() else 0
                    v_energy = lead.get('vocal_energy') or 0.0; is_vocal_heavy = v_energy > 0.02
                    stems_path = lead.get('stems_path')
                    
                    # AI Intelligence: Randomly apply gender transform to the MAIN lead for variety
                    g_swap = "none"
                    if is_vocal_heavy and random.random() > 0.7:
                        orig_g = (lead.get('vocal_gender') or "unknown").lower()
                        g_swap = "female" if "male" in orig_g and "female" not in orig_g else "male"
                        print(f"[AI] Transforming Lead Gender: {lead['filename']} -> {g_swap.upper()}")

                    if stems_path and os.path.exists(stems_path) and (is_drop or 'verse 1' in b_name.lower()):
                        # Sidechain Keyframes for Leads
                        m_keys = {}
                        if not is_vocal_heavy:
                            try:
                                m_keys['volume'] = self.processor.calculate_sidechain_keyframes(main_drum['file_path'], b_dur + overlap)
                            except: pass

                        segments.append({
                            'id': lead['id'], 'filename': f"{lead['filename']} (BASS)", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                            'start_ms': current_ms, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(lead, b_name), 'stems_path': stems_path,
                            'volume': 0.9, 'lane': find_free_lane(current_ms, b_dur + overlap, role="bass"), 'pitch_shift': ps, 'fade_in_ms': 4000, 'fade_out_ms': 4000,
                            'vocal_vol': 0.0, 'drum_vol': 0.0, 'bass_vol': 1.2, 'instr_vol': 0.0, 'ducking_depth': 0.5, 'keyframes': {}
                        })
                        segments.append({
                            'id': lead['id'], 'filename': f"{lead['filename']} (DRUMS)", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                            'start_ms': current_ms + 4000, 'duration_ms': b_dur + overlap - 4000, 'offset_ms': get_best_offset(lead, b_name) + 4000, 'stems_path': stems_path,
                            'vocal_lyrics': lead.get('vocal_lyrics'), 'vocal_gender': lead.get('vocal_gender'),
                            'volume': 1.0, 'lane': find_free_lane(current_ms + 4000, b_dur + overlap - 4000, role="percussion"), 'pitch_shift': ps, 'fade_in_ms': 2000, 'fade_out_ms': 4000,
                            'vocal_vol': 0.0, 'drum_vol': 1.1, 'bass_vol': 0.0, 'instr_vol': 0.0, 'is_primary': True, 'keyframes': {}
                        })
                        segments.append({
                            'id': lead['id'], 'filename': f"{lead['filename']} (LEAD)", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                            'start_ms': current_ms + 8000, 'duration_ms': b_dur + overlap - 8000, 'offset_ms': get_best_offset(lead, b_name) + 8000, 'stems_path': stems_path,
                            'vocal_lyrics': lead.get('vocal_lyrics'), 'vocal_gender': lead.get('vocal_gender'),
                            'volume': 0.8, 'lane': find_free_lane(current_ms + 8000, b_dur + overlap - 8000, role="melodic"), 'pitch_shift': ps, 'fade_in_ms': 4000, 'fade_out_ms': 4000,
                            'vocal_vol': 1.3 if is_vocal_heavy else 0.0, 'drum_vol': 0.0, 'bass_vol': 0.0, 'instr_vol': 1.0, 'duck_low': 0.4,
                            'gender_swap': g_swap,
                            'ducking_depth': 0.2 if is_vocal_heavy else 0.7,
                            'keyframes': m_keys
                        })
                    else:
                        lane = find_free_lane(current_ms, b_dur + overlap, role="melodic")
                        m_keys = {}
                        if not is_vocal_heavy:
                            try:
                                m_keys['volume'] = self.processor.calculate_sidechain_keyframes(main_drum['file_path'], b_dur + overlap)
                            except: pass
                        segments.append({
                            'id': lead['id'], 'filename': lead['filename'], 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                            'start_ms': current_ms, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(lead, b_name), 'stems_path': lead.get('stems_path'),
                            'vocal_lyrics': lead.get('vocal_lyrics'), 'vocal_gender': lead.get('vocal_gender'),
                            'volume': 0.85 if is_vocal_heavy else 0.7, 'pan': 0.0, 'lane': lane, 'pitch_shift': ps, 'low_cut': 400, 'fade_in_ms': 4000, 'fade_out_ms': 4000,
                            'vocal_vol': 1.3 if is_vocal_heavy else 0.8, 'instr_vol': 0.4 if is_vocal_heavy else 0.9, 'bass_vol': 0.6 if is_vocal_heavy else 0.8,
                            'vocal_shift': 12 if is_drop and is_vocal_heavy else 0, 
                            'gender_swap': g_swap,
                            'ducking_depth': 0.15 if is_vocal_heavy else 0.75, 'harmony_level': 0.4 if is_drop else 0.1, 'duck_low': 0.1 if is_vocal_heavy else 0.5,
                            'keyframes': m_keys
                        })

                    should_stack = is_vocal_heavy or (random.random() > 0.5)
                    if should_stack and not is_build:
                        for s_shift in [7, -5]:
                            lane = find_free_lane(current_ms, b_dur + overlap, role="atmosphere")
                            segments.append({
                                'id': lead['id'], 'filename': f"{lead['filename']} (H{s_shift:+})", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                                'start_ms': current_ms, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(lead, b_name), 'stems_path': lead.get('stems_path'),
                                'volume': 0.4, 'pan': -0.7 if s_shift > 0 else 0.7, 'lane': lane, 'pitch_shift': ps, 'low_cut': 800, 'fade_in_ms': 5000, 'fade_out_ms': 5000,
                                'vocal_vol': 1.0 if is_vocal_heavy else 0.0, 'instr_vol': 0.0 if is_vocal_heavy else 0.8, 'bass_vol': 0.0, 'vocal_shift': s_shift, 'ducking_depth': 0.8, 'reverb': 0.5, 'duck_low': 0.1, 'duck_mid': 0.6,
                                'keyframes': {}
                            })
                        
                        # 2. Gender-Swapped Layer (Pro Feature)
                        if is_vocal_heavy and lead.get('vocal_gender'):
                            # Detect opposite gender
                            orig_g = lead.get('vocal_gender').lower()
                            target_g = "female" if "male" in orig_g and "female" not in orig_g else "male"
                            
                            lane = find_free_lane(current_ms, b_dur + overlap, role="atmosphere")
                            segments.append({
                                'id': lead['id'], 'filename': f"{lead['filename']} ({target_g.upper()})", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                                'start_ms': current_ms, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(lead, b_name), 'stems_path': lead.get('stems_path'),
                                'vocal_lyrics': lead.get('vocal_lyrics'), 'vocal_gender': lead.get('vocal_gender'),
                                'volume': 0.5, 'pan': 0.4, 'lane': lane, 'pitch_shift': ps, 'low_cut': 400, 'fade_in_ms': 4000, 'fade_out_ms': 4000,
                                'vocal_vol': 1.2, 'instr_vol': 0.0, 'bass_vol': 0.0, 'vocal_shift': 0, 'gender_swap': target_g, 'ducking_depth': 0.7, 'reverb': 0.4,
                                'keyframes': {}
                            })

            # --- ATMOSPHERE GLUE (Lanes 6-7) ---
            if not is_intro and not is_outro:
                glue = fx_tracks[idx % len(fx_tracks)]
                lane = find_free_lane(current_ms, b_dur + overlap, role="atmosphere")
                segments.append({
                    'id': glue['id'], 'filename': "ATMOS GLUE", 'file_path': glue['file_path'], 'bpm': glue['bpm'], 'harmonic_key': glue['harmonic_key'],
                    'start_ms': current_ms, 'duration_ms': b_dur + overlap, 'offset_ms': 0, 'stems_path': glue.get('stems_path'),
                    'volume': random.uniform(0.15, 0.25) if is_intro else random.uniform(0.2, 0.3), 
                    'lane': lane, 'low_cut': 1200 if is_intro else 800, 'high_cut': 8000, 'fade_in_ms': 8000 if is_intro else 5000, 'fade_out_ms': 5000,
                    'pan': 0.0, 'is_ambient': True, 'ducking_depth': 0.99, 'reverb': 0.8, 'duck_low': 0.1, 'duck_mid': 0.3,
                    'keyframes': {
                        'pan': [(0, random.uniform(-0.8, -0.3)), (b_dur/2, random.uniform(0.3, 0.8)), (b_dur, random.uniform(-0.8, -0.3))]
                    }
                })
            current_ms += (b_dur - overlap)
        
        # --- PHASE 2: AI Generative Asset Injection ---
        build_end_ms = 0; running_ms = start_time_ms
        for b in blocks:
            b_name_l = b['name'].lower()
            is_b = any(k in b_name_l for k in ['build', 'riser', 'tension'])
            if is_b:
                build_end_ms = running_ms + b['dur']
                break
            running_ms += (b['dur'] - overlap)

        if build_end_ms > 0:
            op_riser = os.path.abspath(f"generated_assets/hyper_riser_{random.randint(0,999)}.wav")
            try:
                p = self.generator.get_transition_params(melodic_leads[0], melodic_leads[1], type_context="Build a high-energy riser.")
                self.generator.generate_riser(duration_sec=4.0, bpm=target_bpm, output_path=op_riser, params=p)
                # Auto-ingest
                from src.ingestion import IngestionEngine
                IngestionEngine(db_path=self.dm.db_path).ingest_single_file(op_riser)
                segments.append({
                    'id': -1, 'filename': f"HYPER RISER ({p.get('description', 'Neural')})", 'file_path': op_riser,
                    'bpm': target_bpm, 'harmonic_key': 'N/A', 'start_ms': build_end_ms - 4000, 'duration_ms': 4000,
                    'offset_ms': 0, 'stems_path': None, 'lane': find_free_lane(build_end_ms - 4000, 4000, role="atmosphere", preferred=11),
                    'volume': 0.7, 'fade_in_ms': 3500, 'fade_out_ms': 500, 'keyframes': {}
                })
            except: pass
            op_drop = os.path.abspath(f"generated_assets/hyper_drop_{random.randint(0,999)}.wav")
            try:
                p = self.generator.get_transition_params(melodic_leads[1], melodic_leads[2], type_context="Create a heavy sub-bass impact.")
                self.generator.generate_riser(duration_sec=4.0, bpm=target_bpm, output_path=op_drop, params=p)
                # Auto-ingest
                from src.ingestion import IngestionEngine
                IngestionEngine(db_path=self.dm.db_path).ingest_single_file(op_drop)
                segments.append({
                    'id': -1, 'filename': f"HYPER DROP ({p.get('description', 'Sub')})", 'file_path': op_drop,
                    'bpm': target_bpm, 'harmonic_key': 'N/A', 'start_ms': build_end_ms, 'duration_ms': 4000,
                    'offset_ms': 0, 'stems_path': None, 'lane': find_free_lane(build_end_ms, 4000, role="atmosphere", preferred=11),
                    'volume': 0.9, 'fade_in_ms': 50, 'fade_out_ms': 3500, 'keyframes': {}
                })
            except: pass
        return segments

    def find_best_filler_for_gap(self, prev_track_id=None, next_track_id=None):
        """Finds the most compatible track to fill a gap."""
        conn = self.dm.get_conn()
        conn.row_factory = lambda cursor, row: {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks"); all_tracks = cursor.fetchall(); conn.close()
        if not all_tracks: return None
        prev_track = next((t for t in all_tracks if t['id'] == prev_track_id), None) if prev_track_id else None
        next_track = next((t for t in all_tracks if t['id'] == next_track_id), None) if next_track_id else None
        scored = []
        for cand in all_tracks:
            if prev_track and cand['id'] == prev_track['id']: continue
            if next_track and cand['id'] == next_track['id']: continue
            c_emb = self.dm.get_embedding(cand['clp_embedding_id']) if cand['clp_embedding_id'] else None
            if prev_track and next_track:
                score = self.scorer.calculate_bridge_score(prev_track, next_track, cand, c_emb=c_emb)
            elif prev_track:
                p_emb = self.dm.get_embedding(prev_track['clp_embedding_id']) if prev_track['clp_embedding_id'] else None
                score = self.scorer.get_total_score(prev_track, cand, p_emb, c_emb)['total']
            elif next_track:
                n_emb = self.dm.get_embedding(next_track['clp_embedding_id']) if next_track['clp_embedding_id'] else None
                score = self.scorer.get_total_score(next_track, cand, n_emb, c_emb)['total']
            else:
                score = (cand.get('energy') or 0) * 100
            scored.append((score, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None
