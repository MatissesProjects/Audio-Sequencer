import os
import sqlite3
from src.analysis import AnalysisModule
from src.processor import AudioProcessor
from src.database import init_db
from src.core.config import AppConfig
from tqdm import tqdm

class IngestionEngine:
    """Manages scanning directories and populating the database."""
    
    SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.flac', '.aiff', '.ogg')

    def __init__(self, db_path=None):
        self.db_path = db_path or AppConfig.DB_PATH
        self.analyzer = AnalysisModule()
        self.processor = AudioProcessor()
        AppConfig.ensure_dirs()
        # Ensure DB exists
        if not os.path.exists(self.db_path):
            init_db(self.db_path)

    def scan_directory(self, root_dir):
        """Recursively scans a directory for audio files and processes them."""
        print(f"Scanning directory: {root_dir}")
        audio_files = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(self.SUPPORTED_EXTENSIONS):
                    audio_files.append(os.path.join(root, f))

        print(f"Found {len(audio_files)} audio files. Starting analysis...")
        
        for file_path in tqdm(audio_files):
            self.ingest_single_file(file_path)

        print("Ingestion complete.")

    def ingest_single_file(self, file_path):
        """Analyzes a single file, separates stems, and stores in DB."""
        abs_path = os.path.abspath(file_path)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, stems_path FROM tracks WHERE file_path = ?", (abs_path,))
        row = cursor.fetchone()
        if row and row[1]: # Already has stems
            conn.close()
            return

        try:
            features = self.analyzer.analyze_file(file_path)
            stems_dir = AppConfig.get_stems_path(features['filename'])
            self.processor.separate_stems(file_path, stems_dir)                
            
            # Vocal Analysis
            vocal_lyrics = None
            vocal_gender = None
            if features.get('vocal_energy', 0) > 0.1:
                vocal_path = os.path.join(stems_dir, "vocals.wav")
                if os.path.exists(vocal_path):
                    from src.vocal_analyzer import VocalAnalyzer
                    va = VocalAnalyzer()
                    res = va.analyze_vocals(vocal_path)
                    vocal_lyrics = res.get("lyrics")
                    vocal_gender = res.get("gender")

            import json
            sections_json = json.dumps(features.get('sections', []))

            # Update DB with stems_path
            cursor.execute("UPDATE tracks SET stems_path = ? WHERE file_path = ?", (os.path.abspath(stems_dir), abs_path))
            
            if not row:
                cursor.execute('''
                    INSERT INTO tracks (
                        file_path, filename, duration, sample_rate, 
                        bpm, harmonic_key, energy, onset_density,
                        loop_start, loop_duration, onsets_json, stems_path, vocal_energy, vocal_lyrics, vocal_gender, sections_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    features['file_path'], features['filename'], features['duration'],
                    features['sample_rate'], features['bpm'], features['harmonic_key'],
                    features['energy'], features.get('onset_density', 0),
                    features.get('loop_start', 0), features.get('loop_duration', 0),
                    features['onsets_json'], os.path.abspath(stems_dir), features.get('vocal_energy', 0),
                    vocal_lyrics, vocal_gender, sections_json
                ))
            else:
                 cursor.execute('''
                    UPDATE tracks SET vocal_lyrics = ?, vocal_gender = ?, sections_json = ? WHERE file_path = ?
                 ''', (vocal_lyrics, vocal_gender, sections_json, abs_path))
                 
            conn.commit()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        engine = IngestionEngine()
        engine.scan_directory(sys.argv[1])
    else:
        print("Usage: python src/ingestion.py <directory_to_scan>")
