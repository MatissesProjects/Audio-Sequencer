import os
import sys
from typing import List, Dict, Optional, Any, Union, Tuple

# Ensure the project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.database import DataManager
from src.ingestion import IngestionEngine
from tqdm import tqdm

def main() -> None:
    parser = argparse.ArgumentParser(description="AudioSequencer AI - The Flow")
    parser.add_argument("--scan", type=str, help="Directory to scan for audio files")
    parser.add_argument("--stats", action="store_true", help="Show library statistics")
    parser.add_argument("--embed", action="store_true", help="Generate AI embeddings for all tracks")
    parser.add_argument("--gui", action="store_true", help="Launch the Desktop GUI")
    parser.add_argument("--full-mix", action="store_true", help="Sequence and mix ALL tracks")
    parser.add_argument("--layered", action="store_true", help="Create a 2-minute layered journey")
    parser.add_argument("--hyper", action="store_true", help="Create a professional generative arrangement")
    parser.add_argument("--separate-all", action="store_true", help="Batch process stem separation")
    
    args = parser.parse_args()
    dm = DataManager()
    
    if args.scan:
        engine = IngestionEngine(db_path=dm.db_path)
        engine.scan_directory(args.scan)

    if args.separate_all:
        print("Starting bulk stem separation (Remote AI)...")
        from src.processor import AudioProcessor
        from src.core.config import AppConfig
        proc = AudioProcessor()
        conn = dm.get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path, filename, stems_path FROM tracks")
        tracks: List[Tuple[Any, ...]] = cursor.fetchall()
        
        for tid, f_path, fname, existing_stems in tqdm(tracks):
            if existing_stems and os.path.exists(existing_stems): continue
            try:
                stems_dir = AppConfig.get_stems_path(fname)
                proc.separate_stems(f_path, stems_dir)
                cursor.execute("UPDATE tracks SET stems_path = ? WHERE id = ?", (os.path.abspath(stems_dir), tid))
                conn.commit()
            except Exception as e:
                print(f"Error separating {fname}: {e}")
        conn.close(); print("Bulk separation complete.")
        
    if args.embed:
        print("Initializing AI Embedding Engine...")
        from src.embeddings import EmbeddingEngine
        embed_engine = EmbeddingEngine()
        conn = dm.get_conn(); cursor = conn.cursor()
        cursor.execute("SELECT id, file_path, clp_embedding_id FROM tracks")
        tracks = cursor.fetchall()
        for track_id, file_path, existing_embed in tqdm(tracks):
            if existing_embed: continue
            try:
                embedding = embed_engine.get_embedding(file_path)
                dm.add_embedding(track_id, embedding, metadata={"file_path": file_path})
            except Exception as e:
                print(f"Error embedding {file_path}: {e}")
        conn.close()

    if args.stats:
        stats = dm.get_library_stats()
        print("\n=== Audio Library Statistics ===")
        print(f"Total Tracks: {stats.get('total_tracks', 0)}")
        if stats.get('total_tracks', 0) > 0:
            print(f"BPM Range:    {stats.get('min_bpm')} - {stats.get('max_bpm')} (Avg: {stats.get('avg_bpm')})")
            print("Key Distribution:")
            for key, count in stats.get('key_distribution', {}).items():
                print(f"  {key}: {count}")
        print("================================\n")

    if args.gui:
        from PyQt6.QtWidgets import QApplication
        from src.ui.main_window import AudioSequencerApp
        app = QApplication(sys.argv)
        window = AudioSequencerApp()
        window.show()
        sys.exit(app.exec())

    if args.full_mix:
        from src.orchestrator import FullMixOrchestrator
        FullMixOrchestrator().generate_hyper_mix(target_bpm=124.0)

if __name__ == "__main__":
    main()
