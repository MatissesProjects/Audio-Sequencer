import os
import sqlite3
from src.analysis import AnalysisModule
from src.processor import AudioProcessor
from src.database import init_db
from tqdm import tqdm

class IngestionEngine:
    """Manages scanning directories and populating the database."""
    
    SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.flac', '.aiff', '.ogg')

    def __init__(self, db_path="audio_library.db"):
        self.db_path = db_path
        self.analyzer = AnalysisModule()
        self.processor = AudioProcessor()
        # Ensure DB exists
        if not os.path.exists(db_path):
            init_db(db_path)

    def scan_directory(self, root_dir):
        """Recursively scans a directory for audio files and processes them."""
        print(f"Scanning directory: {root_dir}")
        audio_files = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(self.SUPPORTED_EXTENSIONS):
                    audio_files.append(os.path.join(root, f))

        print(f"Found {len(audio_files)} audio files. Starting analysis...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for file_path in tqdm(audio_files):
            # Skip if already analyzed (simple path check)
            cursor.execute("SELECT id, stems_path FROM tracks WHERE file_path = ?", (os.path.abspath(file_path),))
            row = cursor.fetchone()
            if row and row[1]: # Already has stems
                continue

            try:
                features = self.analyzer.analyze_file(file_path)
                
                # Trigger Stem Separation
                stems_dir = os.path.join("stems_library", str(features['filename']).replace(" ", "_"))
                self.processor.separate_stems(file_path, stems_dir)
                
                if row: # Update existing record with stems_path
                    cursor.execute("UPDATE tracks SET stems_path = ? WHERE id = ?", (os.path.abspath(stems_dir), row[0]))
                else:
                    cursor.execute('''
                        INSERT INTO tracks (
                            file_path, filename, duration, sample_rate, 
                            bpm, harmonic_key, energy, onset_density,
                            loop_start, loop_duration, onsets_json, stems_path, vocal_energy
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        features['file_path'], features['filename'], features['duration'],
                        features['sample_rate'], features['bpm'], features['harmonic_key'],
                        features['energy'], features.get('onset_density', 0),
                        features.get('loop_start', 0), features.get('loop_duration', 0),
                        features['onsets_json'], os.path.abspath(stems_dir), features.get('vocal_energy', 0)
                    ))
                conn.commit()
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        conn.close()
        print("Ingestion complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        engine = IngestionEngine()
        engine.scan_directory(sys.argv[1])
    else:
        print("Usage: python src/ingestion.py <directory_to_scan>")
