import sqlite3
import os
from src.scoring import CompatibilityScorer
from src.processor import AudioProcessor
from src.renderer import FlowRenderer
from src.database import DataManager

def run_test():
    dm = DataManager()
    conn = dm.get_conn()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, bpm, harmonic_key, file_path, clp_embedding_id FROM tracks LIMIT 2")
    rows = cursor.fetchall()
    
    if len(rows) < 2:
        print("Need at least 2 tracks in DB.")
        return

    t1, t2 = rows[0], rows[1]
    
    # Fetch embeddings
    emb1 = dm.get_embedding(t1['clp_embedding_id']) if t1['clp_embedding_id'] else None
    emb2 = dm.get_embedding(t2['clp_embedding_id']) if t2['clp_embedding_id'] else None
    
    scorer = CompatibilityScorer()
    proc = AudioProcessor()
    rend = FlowRenderer()
    
    scores = scorer.get_total_score(t1, t2, emb1, emb2)
    
    print("\n=== Multi-Dimensional Compatibility Test ===")
    print(f"Track A: {t1['filename']} ({t1['bpm']} BPM)")
    print(f"Track B: {t2['filename']} ({t2['bpm']} BPM)")
    print("-" * 45)
    print(f"BPM Match:      {scores['bpm_score']}%")
    print(f"Harmonic Match: {scores['harmonic_score']}%")
    print(f"Semantic Vibe:  {scores['semantic_score']}%")
    print(f"OVERALL SCORE:  {scores['total']}%")
    print("-" * 45)
    
    if scores['total'] > 75:
        print(f"Action: Stretching {t2['filename']} to {t1['bpm']} BPM...")
        stretched_path = "temp_stretched.wav"
        proc.stretch_to_bpm(t2['file_path'], t2['bpm'], t1['bpm'], stretched_path)
        
        print(f"Action: Layering Track A + Stretched Track B...")
        final_mix = "final_layered_mix.wav"
        rend.mix_tracks(t1['file_path'], stretched_path, final_mix)
        
        # Cleanup temp
        if os.path.exists(stretched_path):
            os.remove(stretched_path)
            
        print(f"SUCCESS: Listen to {os.path.abspath(final_mix)}")
    else:
        print("Result: Compatibility too low for a good mix.")
    print("-" * 35 + "\n")

if __name__ == "__main__":
    run_test()
