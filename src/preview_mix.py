import sqlite3
import os
from src.scoring import CompatibilityScorer
from src.processor import AudioProcessor

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
    
    scores = scorer.get_total_score(t1, t2)
    
    print("\n=== Advanced Compatibility Test ===")
    print(f"Track A: {t1['filename']} ({t1['bpm']} BPM, Key: {t1['harmonic_key']})")
    print(f"Track B: {t2['filename']} ({t2['bpm']} BPM, Key: {t2['harmonic_key']})")
    print("-" * 35)
    print(f"BPM Match:      {scores['bpm_score']}%")
    print(f"Harmonic Match: {scores['harmonic_score']}%")
    print(f"OVERALL SCORE:  {scores['total']}%")
    print("-" * 35)
    
    if scores['total'] > 75:
        print("Result: EXCELLENT MATCH")
        print(f"Action: Stretching {t2['filename']} to {t1['bpm']} BPM...")
        out_path = "preview_sync.wav"
        proc.stretch_to_bpm(t2['file_path'], t2['bpm'], t1['bpm'], out_path)
        print(f"Success: Preview file created at {os.path.abspath(out_path)}")
    else:
        print("Result: MEDIOCRE MATCH")
    print("-" * 35 + "\n")

if __name__ == "__main__":
    run_test()
