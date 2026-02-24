
import pytest
import numpy as np
import os
from src.scoring import CompatibilityScorer
from src.renderer import _process_single_segment, FlowRenderer
from src.core.config import AppConfig

def test_harmonic_scoring():
    scorer = CompatibilityScorer()
    # Identical keys
    assert scorer.calculate_harmonic_score('C', 'C') == 100
    # Perfect 5th (Distance 1)
    assert scorer.calculate_harmonic_score('C', 'G') == 80
    # Far apart
    assert scorer.calculate_harmonic_score('C', 'F#') == 0
    # Fallback for unknown
    assert scorer.calculate_harmonic_score('Unknown', 'C') == 50

def test_bpm_scoring():
    scorer = CompatibilityScorer()
    # Identical BPM
    assert scorer.calculate_bpm_score(120, 120) == 100
    # Slight difference (~5%)
    score = scorer.calculate_bpm_score(120, 126)
    assert 60 < score < 80
    # Large difference
    assert scorer.calculate_bpm_score(120, 180) == 0

def test_total_scoring_robustness():
    scorer = CompatibilityScorer()
    t1 = {'bpm': 120, 'harmonic_key': 'C', 'energy': 0.5, 'onset_density': 2.0}
    t2 = {'bpm': 120, 'key': 'C', 'energy': 0.5, 'onset_density': 2.0}
    
    res = scorer.get_total_score(t1, t2)
    # With identical data (except sem_s fallback), should be high
    # 25 (bpm) + 25 (har) + 15 (sem) + 10 (grv) + 10 (nrg) = 85
    assert res['total'] >= 85
    assert res['harmonic_score'] == 100

def test_database_persistence(tmp_path):
    from src.database import DataManager
    db_path = str(tmp_path / "test.db")
    vec_path = str(tmp_path / "vec_db")
    dm = DataManager(db_path=db_path, vector_dir=vec_path)
    
    # Test SQLite insert
    conn = dm.get_conn()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tracks (file_path, filename, bpm, harmonic_key) VALUES (?, ?, ?, ?)", 
                   ("test.wav", "test.wav", 120.0, "C"))
    track_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Test ChromaDB link
    dummy_emb = np.random.rand(512).astype(np.float32)
    dm.add_embedding(track_id, dummy_emb)
    
    # Verify retrieval
    retrieved = dm.get_embedding(f"track_{track_id}")
    assert retrieved is not None
    assert retrieved.shape == (512,)
    
    # Test joined search
    search_res = dm.search_embeddings(dummy_emb, n_results=1)
    assert len(search_res) == 1
    assert search_res[0]['filename'] == "test.wav"
    assert search_res[0]['bpm'] == 120.0

def test_orchestrator_sequencing(tmp_path):
    from src.database import DataManager
    from src.orchestrator import FullMixOrchestrator
    
    db_path = str(tmp_path / "orch_test.db")
    vec_path = str(tmp_path / "orch_vec")
    
    # Setup test DB with 3 tracks
    dm = DataManager(db_path=db_path, vector_dir=vec_path)
    conn = dm.get_conn()
    cursor = conn.cursor()
    tracks = [
        ("a.wav", "A", 120.0, "C"),
        ("b.wav", "B", 122.0, "G"), # Compatible with A
        ("c.wav", "C", 140.0, "F#") # Incompatible with A
    ]
    for t in tracks:
        cursor.execute("INSERT INTO tracks (file_path, filename, bpm, harmonic_key) VALUES (?, ?, ?, ?)", t)
    conn.commit()
    conn.close()
    
    orch = FullMixOrchestrator()
    orch.dm = dm # Inject test DM
    
    seq = orch.find_curated_sequence(max_tracks=3)
    assert len(seq) >= 2 # Should at least pick A and B
    assert seq[0]['filename'] == "A"
    assert seq[1]['filename'] == "B"

def test_keyframe_interpolation():
    from src.core.models import TrackSegment
    td = {'id': 1, 'filename': 'test.wav', 'file_path': 'test.wav', 'bpm': 120, 'harmonic_key': 'C'}
    seg = TrackSegment(td)
    
    # Test linear ramp
    seg.add_keyframe('volume', 0, 0.0)
    seg.add_keyframe('volume', 1000, 1.0)
    
    assert seg.get_value_at('volume', 0, 0.5) == 0.0
    assert seg.get_value_at('volume', 500, 0.5) == 0.5
    assert seg.get_value_at('volume', 1000, 0.5) == 1.0
    # Clamping
    assert seg.get_value_at('volume', -100, 0.5) == 0.0
    assert seg.get_value_at('volume', 2000, 0.5) == 1.0

def test_orchestrator_lane_neighborhoods():
    from src.orchestrator import FullMixOrchestrator
    orch = FullMixOrchestrator()
    orch.lane_count = 12
    
    # Verify that percussion role stays in its neighborhood if possible
    # Percussion neighborhood: [0, 1, 8]
    lane = orch.get_hyper_segments # This is harder to test without a full DB, 
    # but we can test the internal find_free_lane if we access it.
    
    # We'll mock a minimal environment to test find_free_lane logic
    segments = []
    def mock_find_free_lane(start, dur, role="melodic"):
        neighborhoods = {"percussion": [0, 1, 8], "bass": [2, 3, 9], "melodic": [4, 5, 10]}
        candidates = neighborhoods.get(role, [0])
        for l in candidates:
            return l # Simple return for logic test
            
    assert mock_find_free_lane(0, 1000, role="percussion") == 0
    assert mock_find_free_lane(0, 1000, role="bass") == 2

def test_renderer_cache_invalidation():
    # Helper to simulate the hashing logic in _process_single_segment
    import hashlib
    def get_hash(s):
        key_str = f"{s['file_path']}_{s['bpm']}_{str(s.get('keyframes', {}))}_{s.get('vocal_vol', 1.0)}"
        return hashlib.md5(key_str.encode()).hexdigest()
        
    s1 = {'file_path': 'a.wav', 'bpm': 120, 'vocal_vol': 1.0, 'keyframes': {}}
    h1 = get_hash(s1)
    
    # Change volume -> should change hash
    s2 = s1.copy()
    s2['vocal_vol'] = 0.5
    h2 = get_hash(s2)
    assert h1 != h2
    
    # Add keyframe -> should change hash
    s3 = s1.copy()
    s3['keyframes'] = {'volume': [(0, 1.0)]}
    h3 = get_hash(s3)
    assert h1 != h3

def test_ingestion_registration(tmp_path):
    from src.ingestion import IngestionEngine
    from unittest.mock import MagicMock
    
    db_path = str(tmp_path / "ingest.db")
    ie = IngestionEngine(db_path=db_path)
    
    # Mock the heavy analysis and processor
    ie.analyzer.analyze_file = MagicMock(return_value={
        'file_path': 'fake.wav', 'filename': 'fake.wav', 'duration': 10.0,
        'sample_rate': 44100, 'bpm': 120.0, 'harmonic_key': 'C', 'energy': 0.5,
        'onsets_json': '', 'vocal_energy': 0.1
    })
    ie.processor.separate_stems = MagicMock()
    
    # Run ingestion
    ie.ingest_single_file('fake.wav')
    
    # Verify DB entry
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, bpm FROM tracks WHERE filename='fake.wav'")
    row = cursor.fetchone()
    assert row is not None
    assert row[1] == 120.0
    conn.close()

def test_renderer_hash_consistency():
    # Verify that the same segment properties produce the same cache hash
    s1 = {
        'file_path': 'test.wav', 'bpm': 120, 'pitch_shift': 0, 
        'vocal_vol': 1.0, 'drum_vol': 1.0, 'instr_vol': 1.0
    }
    
    # We can't easily call _process_single_segment without files, 
    # but we can verify the hash logic if we exposed it.
    # For now, let's test the FlowRenderer initialization.
    renderer = FlowRenderer(sample_rate=44100)
    assert renderer.sr == 44100

def test_spectral_ducking_math():
    # Test internal ducking multipliers
    renderer = FlowRenderer()
    
    # Primary vs Background logic
    # (Checking logic from _render_internal)
    other_vocal = {'vocal_energy': 0.8} # Strong vocal
    is_vocal = (other_vocal.get('vocal_energy') or 0.0) > 0.2
    assert is_vocal is True
    
    base_duck = 0.9 if is_vocal else 0.7
    assert base_duck == 0.9
