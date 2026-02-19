import os
import argparse
from src.database import DataManager
from src.ingestion import IngestionEngine

def main():
    parser = argparse.ArgumentParser(description="AudioSequencer AI - Analysis & Ingestion")
    parser.add_argument("--scan", type=str, help="Directory to scan for audio files")
    parser.add_argument("--stats", action="store_true", help="Show library statistics")
    
    args = parser.parse_args()
    dm = DataManager()
    
    if args.scan:
        engine = IngestionEngine(db_path=dm.db_path)
        engine.scan_directory(args.scan)
        
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

if __name__ == "__main__":
    main()
