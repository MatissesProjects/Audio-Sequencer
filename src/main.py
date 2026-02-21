import os
import sys

# Ensure the project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.database import DataManager
from src.ingestion import IngestionEngine
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="AudioSequencer AI - The Flow")
    parser.add_argument("--scan", type=str, help="Directory to scan for audio files")
    parser.add_argument("--stats", action="store_true", help="Show library statistics")
    parser.add_argument("--embed", action="store_true", help="Generate AI embeddings for all tracks")
    parser.add_argument("--gui", action="store_true", help="Launch the Desktop GUI")
    parser.add_argument("--full-mix", action="store_true", help="Sequence and mix ALL tracks into a continuous journey")
    parser.add_argument("--layered", action="store_true", help="Create a 2-minute layered journey with a foundation track")
    parser.add_argument("--hyper", action="store_true", help="Create a professional 5-lane generative arrangement")
    
    args = parser.parse_args()
    dm = DataManager()
    
    if args.scan:
        engine = IngestionEngine(db_path=dm.db_path)
        engine.scan_directory(args.scan)
        
    if args.embed:
        print("Initializing AI Embedding Engine...")
        from src.embeddings import EmbeddingEngine
        embed_engine = EmbeddingEngine()
        conn = dm.get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path, clp_embedding_id FROM tracks")
        tracks = cursor.fetchall()
        
        print(f"Checking {len(tracks)} tracks for missing embeddings...")
        for track_id, file_path, existing_embed in tqdm(tracks):
            if existing_embed:
                continue
            
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
            print(f"BPM Range:    {stats['min_bpm']} - {stats['max_bpm']} (Avg: {stats['avg_bpm']})")
            print("Key Distribution:")
            for key, count in stats['key_distribution'].items():
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
        orch = FullMixOrchestrator()
        orch.generate_full_mix(target_bpm=124)

    if args.layered:
        from src.orchestrator import FullMixOrchestrator
        orch = FullMixOrchestrator()
        orch.generate_layered_journey(target_bpm=124)

    if args.hyper:
        from src.orchestrator import FullMixOrchestrator
        orch = FullMixOrchestrator()
        orch.generate_hyper_mix(target_bpm=124)

if __name__ == "__main__":
    main()
