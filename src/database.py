import sqlite3
import os
import chromadb
from chromadb.config import Settings

class DataManager:
    """Unified manager for SQLite (metadata) and ChromaDB (vectors)."""
    
    def __init__(self, db_path="audio_library.db", vector_dir="vector_db"):
        self.db_path = db_path
        self.vector_dir = vector_dir
        self.init_sqlite()
        self.init_chroma()

    def init_sqlite(self):
        """Initializes the SQLite database with the required schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for audio file metadata and MIR features
        # Added 'onsets_json' to store beat timings for seamless looping
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                duration REAL,
                sample_rate INTEGER,
                bpm REAL,
                harmonic_key TEXT,
                loudness_lufs REAL,
                energy REAL,
                onsets_json TEXT, -- JSON string of beat timestamps
                date_added DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_analyzed DATETIME,
                clp_embedding_id TEXT -- Link to vector database ID
            )
        ''')
        
        conn.commit()
        conn.close()

    def init_chroma(self):
        """Initializes the ChromaDB vector storage."""
        if not os.path.exists(self.vector_dir):
            os.makedirs(self.vector_dir)
            
        self.chroma_client = chromadb.PersistentClient(path=self.vector_dir)
        # Collection for CLAP embeddings
        self.collection = self.chroma_client.get_or_create_collection(name="audio_embeddings")

    def get_conn(self):
        return sqlite3.connect(self.db_path)

    def get_library_stats(self):
        """Returns high-level statistics about the audio library."""
        conn = self.get_conn()
        cursor = conn.cursor()
        
        stats = {}
        cursor.execute("SELECT COUNT(*) FROM tracks")
        stats['total_tracks'] = cursor.fetchone()[0]
        
        if stats['total_tracks'] > 0:
            cursor.execute("SELECT AVG(bpm), MIN(bpm), MAX(bpm) FROM tracks")
            bpm_stats = cursor.fetchone()
            stats['avg_bpm'] = round(bpm_stats[0], 2)
            stats['min_bpm'] = round(bpm_stats[1], 2)
            stats['max_bpm'] = round(bpm_stats[2], 2)
            
            cursor.execute("SELECT harmonic_key, COUNT(*) as count FROM tracks GROUP BY harmonic_key ORDER BY count DESC")
            stats['key_distribution'] = dict(cursor.fetchall())
        
        conn.close()
        return stats

def init_db(db_path="audio_library.db"):
    # Keeping this for backward compatibility
    DataManager(db_path=db_path)

if __name__ == "__main__":
    dm = DataManager()
    print(f"Databases initialized at {os.path.abspath(dm.db_path)} and {os.path.abspath(dm.vector_dir)}")
