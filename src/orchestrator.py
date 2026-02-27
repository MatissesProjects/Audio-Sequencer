import sqlite3
import os
import soundfile as sf
import numpy as np
import librosa
import random
import json
from typing import List, Dict, Optional, Any, Union, Tuple
from src.database import DataManager
from src.scoring import CompatibilityScorer
from src.processor import AudioProcessor
from src.renderer import FlowRenderer
from src.generator import TransitionGenerator
from tqdm import tqdm

class FullMixOrchestrator:
    """Sequences and layers curated selections for maximum musical flow."""

    def __init__(self):
        self.dm: DataManager = DataManager()
        self.scorer: CompatibilityScorer = CompatibilityScorer()
        self.processor: AudioProcessor = AudioProcessor()
        self.renderer: FlowRenderer = FlowRenderer()
        self.generator: TransitionGenerator = TransitionGenerator()
        self.min_score_threshold: float = 55.0
        self.lane_count: int = 20

    def find_curated_sequence(self, max_tracks: int = 6, seed_track: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Finds a high-compatibility path, starting from a seed if provided."""
        conn = self.dm.get_conn()
        conn.row_factory = lambda cursor, row: {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks")
        all_tracks: List[Dict[str, Any]] = cursor.fetchall()
        conn.close()

        if not all_tracks:
            return []

        unvisited = all_tracks.copy()

        if seed_track:
            current = next((t for t in unvisited if t['id'] == seed_track['id']), unvisited[0])   
            unvisited.remove(current)
        else:
            current = unvisited.pop(0)

        sequence = [current]

        while unvisited and len(sequence) < max_tracks:
            best_next = None
            best_score = -1.0
            best_idx = -1

            curr_emb = self.dm.get_embedding(current['clp_embedding_id']) if current['clp_embedding_id'] else None

            for i, candidate in enumerate(unvisited):
                cand_emb = self.dm.get_embedding(candidate['clp_embedding_id']) if candidate['clp_embedding_id'] else None
                score = float(self.scorer.get_total_score(current, candidate, curr_emb, cand_emb)['total'])

                if score > best_score:
                    best_score = score
                    best_next = candidate
                    best_idx = i

            if best_score < self.min_score_threshold:
                break

            current = unvisited.pop(best_idx)
            sequence.append(current)

        return sequence

    def generate_hyper_mix(self, output_path: str = "hyper_automated_mix.mp3", target_bpm: float = 124.0, seed_track: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Direct rendering of a Hyper-Mix journey."""
        segments = self.get_hyper_segments(seed_track=seed_track)
        if not segments:
            return None

        self.renderer.render_timeline(segments, output_path, target_bpm=target_bpm)
        return output_path

    def get_hyper_segments(self, seed_track: Optional[Dict[str, Any]] = None, start_time_ms: int = 0, depth: int = 0, force_ending: bool = False) -> List[Dict[str, Any]]:
        """Returns organized segment data for a hyper-mix."""
        conn = self.dm.get_conn()
        conn.row_factory = lambda cursor, row: {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks")
        all_tracks: List[Dict[str, Any]] = cursor.fetchall()
        conn.close()

        if len(all_tracks) < 5: return []

        vocal_pool = [t for t in all_tracks if ((t.get('vocal_energy') or 0) > 0.02 or t.get('vocal_lyrics')) and t.get('stems_path')]
        
        all_tracks.sort(key=lambda x: x.get('vocal_energy') or 0, reverse=True)
        
        remaining = [t for t in all_tracks if t not in vocal_pool]
        remaining.sort(key=lambda x: x.get('onset_density') or 0, reverse=True)
        drums = remaining[:max(1, int(len(remaining)*0.4))]
        others = [t for t in remaining if t not in drums]
        if not others: others = all_tracks 

        if force_ending:
            blocks = [
                {'name': 'Connect', 'dur': 8000},
                {'name': 'Pre-Finale Build', 'dur': 16000},
                {'name': 'Grand Finale', 'dur': 32000},
                {'name': 'Extended Outro', 'dur': 24000}
            ]
        else:
            blocks = self.generator.get_journey_structure(depth=depth)
            if depth > 0:
                for b in blocks:
                    if 'outro' in b['name'].lower():
                        b['name'] = 'Transition'; b['dur'] = 8000

        segments: List[Dict[str, Any]] = []
        current_ms = start_time_ms
        target_bpm = 124.0
        if seed_track:
            target_bpm = float(seed_track.get('bpm', 124.0))

        d_idx = int(depth)
        scored_others = []
        for t in others:
            score = float(self.scorer.get_total_score(seed_track, t)['total']) if seed_track else 50.0
            jitter = ((t['id'] * (d_idx + 1)) % 100) / 10.0
            scored_others.append((score + jitter, t))
        
        scored_others.sort(key=lambda x: x[0], reverse=True)
        melodic_leads = [x[1] for x in scored_others[:15]]
        if not melodic_leads: melodic_leads = others[:10] if others else all_tracks[:10]
        fx_tracks = melodic_leads[6:12] if len(melodic_leads) >= 12 else melodic_leads[:4]

        scored_vocals = []
        for t in vocal_pool:
            score = float(self.scorer.get_total_score(seed_track, t)['total']) if seed_track else 50.0
            jitter = ((t['id'] * (d_idx + 1)) % 100) / 10.0
            scored_vocals.append((score + jitter, t))
        
        scored_vocals.sort(key=lambda x: x[0], reverse=True)
        rotated_vocals = [x[1] for x in scored_vocals[:10]]
        if not rotated_vocals and vocal_pool: rotated_vocals = vocal_pool[:10]

        main_drum = drums[d_idx % len(drums)] if drums else all_tracks[0]
        if seed_track and depth == 0:
            sk = seed_track.get('harmonic_key') or seed_track.get('key')
            if sk:
                comp_drums = [t for t in drums if self.scorer.calculate_harmonic_score(sk, t['harmonic_key']) >= 80]
                if comp_drums: main_drum = random.choice(comp_drums)

        bass_track = random.choice([t for t in melodic_leads if t['harmonic_key'] == main_drum['harmonic_key']] or [melodic_leads[0]])
        used_vocal_ids: List[int] = []

        os.makedirs("generated_assets", exist_ok=True)
        cloud_path = os.path.abspath(f"generated_assets/spectral_pad_{main_drum['id']}_d{depth}_{random.randint(0,99)}.wav")
        if not os.path.exists(cloud_path):
            try: 
                source_p = seed_track['file_path'] if (seed_track and depth==0) else random.choice(melodic_leads)['file_path']
                self.processor.generate_spectral_pad_remote(source_p, cloud_path, duration=20.0)
            except: cloud_path = main_drum['file_path']

        overlap = 4000

        def find_free_lane(start: int, dur: int, role: str = "melodic", preferred: Optional[int] = None) -> int:
            neighborhoods = {
                "percussion": [0, 1, 8, 12, 16], "bass": [2, 3, 9, 13, 17],
                "melodic": [4, 5, 10, 14, 18], "atmosphere": [6, 7, 11, 15, 19]
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

        def get_best_offset(track: Dict[str, Any], block_type: str) -> float:
            default_offset = (track.get('loop_start') or 0) * 1000.0
            s_json = track.get('sections_json')
            if not s_json: return default_offset
            try:
                sections = json.loads(s_json)
                if not sections: return default_offset
                target = "Verse"
                bt_l = block_type.lower()
                if any(k in bt_l for k in ['drop', 'climax', 'finale']): target = "Drop"
                elif any(k in bt_l for k in ['build', 'riser']): target = "Build"
                elif any(k in bt_l for k in ['intro', 'start']): target = "Intro"
                for s in sections:
                    if s['label'] == target: return s['start'] * 1000.0
            except: pass
            return default_offset

        for idx, block in enumerate(blocks):
            b_name = block['name']; b_dur = block['dur']
            is_drop = any(k in b_name.lower() for k in ['drop', 'finale', 'climax'])
            is_build = any(k in b_name.lower() for k in ['build', 'riser', 'tension'])
            is_intro = any(k in b_name.lower() for k in ['intro', 'connect', 'start'])
            is_outro = any(k in b_name.lower() for k in ['outro', 'fade', 'end'])
            is_transition = any(k in b_name.lower() for k in ['transition', 'bridge'])

            # PERCUSSION
            f_start = current_ms
            p_keys = {}
            if is_intro: p_keys['drum_vol'] = [(0, 0.0), (8000, 0.5), (16000, 1.0)]
            elif is_outro: p_keys['drum_vol'] = [(0, 1.0), (b_dur/2, 0.0)]
            segments.append({
                'id': main_drum['id'], 'filename': main_drum['filename'], 'file_path': main_drum['file_path'], 'bpm': main_drum['bpm'], 'harmonic_key': main_drum['harmonic_key'],
                'start_ms': f_start, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(main_drum, b_name), 'stems_path': main_drum.get('stems_path'),
                'vocal_lyrics': main_drum.get('vocal_lyrics'), 'vocal_gender': main_drum.get('vocal_gender'),
                'volume': 1.0 if is_drop else 0.8, 'is_primary': True, 'lane': find_free_lane(f_start, b_dur + overlap, role="percussion", preferred=0),
                'fade_in_ms': 1000 if not is_intro else 4000, 'fade_out_ms': 4000,
                'drum_vol': 1.3 if is_drop else 1.0, 'instr_vol': 0.3 if is_drop else 0.6, 
                'ducking_depth': 0.3, 'keyframes': p_keys
            })

            # BASS
            b_start = current_ms
            bass_keys = {}
            if is_intro:
                bass_keys['low_cut'] = [(0, 1000), (8000, 200), (16000, 20)]
                bass_keys['bass_vol'] = [(0, 0.5), (16000, 1.0)]
            elif is_outro:
                bass_keys['bass_vol'] = [(b_dur/2, 1.0), (b_dur, 0.0)]
                bass_keys['low_cut'] = [(b_dur/2, 20), (b_dur, 800)]
            segments.append({
                'id': bass_track['id'], 'filename': bass_track['filename'], 'file_path': bass_track['file_path'], 'bpm': bass_track['bpm'], 'harmonic_key': bass_track['harmonic_key'],
                'start_ms': b_start, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(bass_track, b_name), 'stems_path': bass_track.get('stems_path'),
                'vocal_lyrics': bass_track.get('vocal_lyrics'), 'vocal_gender': bass_track.get('vocal_gender'),
                'volume': 0.8, 'is_primary': False, 'lane': find_free_lane(b_start, b_dur + overlap, role="bass", preferred=2), 'fade_in_ms': 3000, 'fade_out_ms': 3000,
                'instr_vol': 1.1 if is_drop else 0.8, 'vocal_vol': 0.0, 'bass_vol': 1.2,
                'ducking_depth': 0.9 if is_drop else 0.7, 'duck_high': 0.4, 'low_cut': 20,
                'keyframes': bass_keys
            })

            # ATMOSPHERE
            if is_intro or is_outro or is_transition:
                segments.append({
                    'id': -2, 'filename': "NEURAL CLOUD", 'file_path': cloud_path, 'bpm': 120, 'harmonic_key': 'N/A',
                    'start_ms': current_ms, 'duration_ms': b_dur + 4000, 'offset_ms': 0, 'stems_path': None,
                    'volume': 0.25 if is_intro else 0.35, 'lane': find_free_lane(current_ms, b_dur + 4000, role="atmosphere", preferred=6), 'fade_in_ms': 5000, 'fade_out_ms': 5000,
                    'is_ambient': True, 'ducking_depth': 0.98, 'reverb': 0.9, 'low_cut': 600, 'duck_low': 0.1, 'duck_mid': 0.4,
                    'keyframes': {'volume': [(0, 0.0), (4000, 1.0), (b_dur, 1.0), (b_dur + 4000, 0.0)]}
                })
            
            # MELODIC
            if is_intro or is_outro:
                lead = melodic_leads[0] if is_intro else melodic_leads[-1]
                m_keys = {}
                if is_intro:
                    m_keys['high_cut'] = [(0, 500), (b_dur*0.6, 3000), (b_dur, 18000)]
                    m_keys['instr_vol'] = [(0, 0.4), (b_dur, 0.8)]
                else:
                    m_keys['high_cut'] = [(0, 18000), (b_dur, 600)]
                    m_keys['instr_vol'] = [(0, 0.8), (b_dur, 0.0)]
                segments.append({
                    'id': lead['id'], 'filename': f"{lead['filename']} ({'INTRO' if is_intro else 'OUTRO'})", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                    'start_ms': current_ms, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(lead, b_name), 'stems_path': lead.get('stems_path'),
                    'vocal_lyrics': lead.get('vocal_lyrics'), 'vocal_gender': lead.get('vocal_gender'),
                    'volume': 0.6, 'lane': find_free_lane(current_ms, b_dur + overlap, role="melodic"), 'fade_in_ms': 4000, 'fade_out_ms': 4000,
                    'vocal_vol': 0.0, 'instr_vol': 0.8, 'reverb': 0.6, 'keyframes': m_keys
                })
            else:
                available_vocals = [t for t in rotated_vocals if t['id'] not in used_vocal_ids]
                if not available_vocals and rotated_vocals:
                    used_vocal_ids = []; available_vocals = rotated_vocals
                
                if available_vocals:
                    lead = random.choice(available_vocals); used_vocal_ids.append(lead['id'])
                else:
                    lead = melodic_leads[idx % len(melodic_leads)]

                if is_build:
                    sub_durs = [4000, 4000, 2000, 2000, 1000, 1000, 1000, 1000]; sub_start = 0
                    for c_idx, sd in enumerate(sub_durs):
                        segments.append({
                            'id': lead['id'], 'filename': f"CHOP {c_idx}", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                            'start_ms': current_ms + sub_start, 'duration_ms': sd + 200, 'offset_ms': (lead.get('loop_start') or 0)*1000, 'stems_path': lead.get('stems_path'),
                            'volume': 0.7 + (c_idx/len(sub_durs) * 0.3), 'lane': find_free_lane(current_ms + sub_start, sd + 200, role="melodic"), 'pitch_shift': int(c_idx/2), 'low_cut': 200 + (c_idx * 100), 'fade_in_ms': 50, 'fade_out_ms': 50,
                            'harmony_level': 0.3 + (c_idx/len(sub_durs) * 0.5), 'vocal_vol': 1.2, 'instr_vol': 0.3, 'ducking_depth': 0.4, 'duck_low': 0.3,
                            'keyframes': {'volume': [(0, 0.8), (sd, 1.0)]}
                        })
                        sub_start += sd
                else:
                    ps = -2 if 'verse 2' in b_name.lower() else 0
                    v_energy = float(lead.get('vocal_energy') or 0.0); is_vocal_heavy = v_energy > 0.02
                    stems_path = lead.get('stems_path')
                    g_swap = "none"
                    if is_vocal_heavy and random.random() > 0.7:
                        orig_g = (lead.get('vocal_gender') or "unknown").lower()
                        g_swap = "female" if "male" in orig_g and "female" not in orig_g else "male"

                    if stems_path and os.path.exists(stems_path) and (is_drop or 'verse 1' in b_name.lower()):
                        m_keys = {}
                        if not is_vocal_heavy:
                            try: m_keys['volume'] = self.processor.calculate_sidechain_keyframes(main_drum['file_path'], b_dur + overlap)
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
                            'gender_swap': g_swap, 'ducking_depth': 0.2 if is_vocal_heavy else 0.7, 'keyframes': m_keys
                        })
                    else:
                        m_keys = {}
                        if not is_vocal_heavy:
                            try: m_keys['volume'] = self.processor.calculate_sidechain_keyframes(main_drum['file_path'], b_dur + overlap)
                            except: pass
                        segments.append({
                            'id': lead['id'], 'filename': lead['filename'], 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                            'start_ms': current_ms, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(lead, b_name), 'stems_path': lead.get('stems_path'),
                            'vocal_lyrics': lead.get('vocal_lyrics'), 'vocal_gender': lead.get('vocal_gender'),
                            'volume': 0.85 if is_vocal_heavy else 0.7, 'pan': 0.0, 'lane': find_free_lane(current_ms, b_dur + overlap, role="melodic"), 'pitch_shift': ps, 'low_cut': 400, 'fade_in_ms': 4000, 'fade_out_ms': 4000,
                            'vocal_vol': 1.3 if is_vocal_heavy else 0.8, 'instr_vol': 0.4 if is_vocal_heavy else 0.9, 'bass_vol': 0.6 if is_vocal_heavy else 0.8,
                            'vocal_shift': 12 if is_drop and is_vocal_heavy else 0, 
                            'gender_swap': g_swap, 'ducking_depth': 0.15 if is_vocal_heavy else 0.75, 'harmony_level': 0.4 if is_drop else 0.1, 'duck_low': 0.1 if is_vocal_heavy else 0.5,
                            'keyframes': m_keys
                        })

                    if (is_vocal_heavy or random.random() > 0.5) and not is_build:
                        for s_shift in [7, -5]:
                            h_gswap = "none"
                            if s_shift == 7 and lead.get('vocal_gender'):
                                orig_g = lead.get('vocal_gender').lower()
                                h_gswap = "female" if "male" in orig_g and "female" not in orig_g else "male"
                            segments.append({
                                'id': lead['id'], 'filename': f"{lead['filename']} (H{s_shift:+})", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                                'start_ms': current_ms, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(lead, b_name), 'stems_path': lead.get('stems_path'),
                                'vocal_lyrics': lead.get('vocal_lyrics'), 'vocal_gender': lead.get('vocal_gender'),
                                'volume': 0.4, 'pan': -0.7 if s_shift > 0 else 0.7, 'lane': find_free_lane(current_ms, b_dur + overlap, role="atmosphere"), 'pitch_shift': ps, 'low_cut': 800, 'fade_in_ms': 5000, 'fade_out_ms': 5000,
                                'vocal_vol': 1.0 if is_vocal_heavy else 0.0, 'instr_vol': 0.0 if is_vocal_heavy else 0.8, 'bass_vol': 0.0, 'vocal_shift': s_shift, 
                                'gender_swap': h_gswap, 'ducking_depth': 0.8, 'reverb': 0.5, 'duck_low': 0.1, 'duck_mid': 0.6, 'keyframes': {}
                            })
                        
                        if is_vocal_heavy and lead.get('vocal_gender'):
                            orig_g = lead.get('vocal_gender').lower()
                            target_g = "female" if "male" in orig_g and "female" not in orig_g else "male"
                            segments.append({
                                'id': lead['id'], 'filename': f"{lead['filename']} ({target_g.upper()})", 'file_path': lead['file_path'], 'bpm': lead['bpm'], 'harmonic_key': lead['harmonic_key'],
                                'start_ms': current_ms, 'duration_ms': b_dur + overlap, 'offset_ms': get_best_offset(lead, b_name), 'stems_path': lead.get('stems_path'),
                                'vocal_lyrics': lead.get('vocal_lyrics'), 'vocal_gender': lead.get('vocal_gender'),
                                'volume': 0.5, 'pan': 0.4, 'lane': find_free_lane(current_ms, b_dur + overlap, role="atmosphere"), 'pitch_shift': ps, 'low_cut': 400, 'fade_in_ms': 4000, 'fade_out_ms': 4000,
                                'vocal_vol': 1.2, 'instr_vol': 0.0, 'bass_vol': 0.0, 'vocal_shift': 0, 'gender_swap': target_g, 'ducking_depth': 0.7, 'reverb': 0.4, 'keyframes': {}
                            })

            if not is_intro and not is_outro:
                glue = fx_tracks[idx % len(fx_tracks)]
                b_dur_val = float(b_dur)
                segments.append({
                    'id': glue['id'], 'filename': "ATMOS GLUE", 'file_path': glue['file_path'], 'bpm': glue['bpm'], 'harmonic_key': glue['harmonic_key'],
                    'start_ms': current_ms, 'duration_ms': b_dur + overlap, 'offset_ms': 0, 'stems_path': glue.get('stems_path'),
                    'volume': random.uniform(0.15, 0.25) if is_intro else random.uniform(0.2, 0.3), 
                    'lane': find_free_lane(current_ms, b_dur + overlap, role="atmosphere"), 'low_cut': 1200 if is_intro else 800, 'high_cut': 8000, 'fade_in_ms': 8000 if is_intro else 5000, 'fade_out_ms': 5000,
                    'pan': 0.0, 'is_ambient': True, 'ducking_depth': 0.99, 'reverb': 0.8, 'duck_low': 0.1, 'duck_mid': 0.3,
                    'keyframes': {'pan': [(0, random.uniform(-0.8, -0.3)), (b_dur_val/2, random.uniform(0.3, 0.8)), (b_dur_val, random.uniform(-0.8, -0.3))]}
                })
            current_ms += (b_dur - overlap)
        
        # Generative Asset Injection
        build_end_ms = 0; running_ms = start_time_ms
        for b in blocks:
            if any(k in b['name'].lower() for k in ['build', 'riser', 'tension']):
                build_end_ms = running_ms + b['dur']; break
            running_ms += (b['dur'] - overlap)

        if build_end_ms > 0:
            try:
                op_riser = os.path.abspath(f"generated_assets/hyper_riser_{random.randint(0,999)}.wav")
                p = self.generator.get_transition_params(melodic_leads[0], melodic_leads[1], type_context="Build a high-energy riser.")
                self.generator.generate_riser(duration_sec=4.0, bpm=target_bpm, output_path=op_riser, params=p)
                segments.append({
                    'id': -1, 'filename': f"HYPER RISER", 'file_path': op_riser, 'bpm': target_bpm, 'harmonic_key': 'N/A', 
                    'start_ms': build_end_ms - 4000, 'duration_ms': 4000, 'offset_ms': 0, 'stems_path': None, 
                    'lane': find_free_lane(build_end_ms - 4000, 4000, role="atmosphere", preferred=11),
                    'volume': 0.7, 'fade_in_ms': 3500, 'fade_out_ms': 500, 'keyframes': {}
                })
            except: pass
        return segments

    def find_best_filler_for_gap(self, prev_track_id: Optional[int] = None, next_track_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Finds the most compatible track to fill a gap."""
        conn = self.dm.get_conn()
        conn.row_factory = lambda cursor, row: {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks"); all_tracks: List[Dict[str, Any]] = cursor.fetchall(); conn.close()
        if not all_tracks: return None
        prev_track = next((t for t in all_tracks if t['id'] == prev_track_id), None) if prev_track_id else None
        next_track = next((t for t in all_tracks if t['id'] == next_track_id), None) if next_track_id else None
        scored = []
        for cand in all_tracks:
            if prev_track and cand['id'] == prev_track['id']: continue
            if next_track and cand['id'] == next_track['id']: continue
            c_emb = self.dm.get_embedding(cand['clp_embedding_id']) if cand['clp_embedding_id'] else None
            if prev_track and next_track:
                score = float(self.scorer.calculate_bridge_score(prev_track, next_track, cand, c_emb=c_emb))
            elif prev_track:
                p_emb = self.dm.get_embedding(prev_track['clp_embedding_id']) if prev_track['clp_embedding_id'] else None
                score = float(self.scorer.get_total_score(prev_track, cand, p_emb, c_emb)['total'])
            elif next_track:
                n_emb = self.dm.get_embedding(next_track['clp_embedding_id']) if next_track['clp_embedding_id'] else None
                score = float(self.scorer.get_total_score(next_track, cand, n_emb, c_emb)['total'])
            else:
                score = (float(cand.get('energy') or 0)) * 100
            scored.append((score, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None
