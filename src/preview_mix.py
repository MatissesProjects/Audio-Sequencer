import sqlite3
import os
from src.scoring import CompatibilityScorer
from src.processor import AudioProcessor
from src.renderer import FlowRenderer

def run_test():
    conn = sqlite3.connect('audio_library.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT filename, bpm, harmonic_key, file_path FROM tracks LIMIT 2")
    rows = cursor.fetchall()
    
    if len(rows) < 2:
        print("Need at least 2 tracks in DB.")
        return

    t1, t2 = rows[0], rows[1]
    scorer = CompatibilityScorer()
    proc = AudioProcessor()
    rend = FlowRenderer()
    
    scores = scorer.get_total_score(t1, t2)
    
    print("\n=== Advanced Mixing & Layering Test ===")
    print(f"Track A: {t1['filename']} ({t1['bpm']} BPM)")
    print(f"Track B: {t2['filename']} ({t2['bpm']} BPM)")
    print("-" * 35)
    
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
