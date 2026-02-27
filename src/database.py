import sqlite3
import os
import chromadb
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
from src.core.config import AppConfig

class DataManager:
    """Unified manager for SQLite (metadata) and ChromaDB (vectors)."""
    
    def __init__(self, db_path: Optional[str] = None, vector_dir: Optional[str] = None):
        self.db_path: str = db_path or AppConfig.DB_PATH
        self.vector_dir: str = vector_dir or AppConfig.VECTOR_DB_DIR
        self.init_sqlite()
        self.init_chroma()

    def init_sqlite(self) -> None:
        """Initializes the SQLite database with the required schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
                onset_density REAL,
                loop_start REAL,
                loop_duration REAL,
                onsets_json TEXT,
                date_added DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_analyzed DATETIME,
                clp_embedding_id TEXT,
                stems_path TEXT,
                vocal_energy REAL,
                vocal_lyrics TEXT,
                vocal_gender TEXT,
                sections_json TEXT
            )
        ''')
        
        # Migrations
        migrations = [
            "onset_density REAL", "loop_start REAL", "loop_duration REAL",
            "stems_path TEXT", "vocal_energy REAL", "vocal_lyrics TEXT",
            "vocal_gender TEXT", "sections_json TEXT"
        ]
        for col in migrations:
            try:
                cursor.execute(f"ALTER TABLE tracks ADD COLUMN {col}")
            except sqlite3.OperationalError: pass
        
        conn.commit()
        conn.close()

    def init_chroma(self) -> None:
        """Initializes the ChromaDB vector storage."""
        if not os.path.exists(self.vector_dir):
            os.makedirs(self.vector_dir)
            
        self.chroma_client = chromadb.PersistentClient(path=self.vector_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="audio_embeddings")

    def get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def add_embedding(self, track_id: int, embedding: Union[np.ndarray, List[float]], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Stores a vector in ChromaDB and links it to the track_id."""
        embed_id = f"track_{track_id}"
        self.collection.add(
            ids=[embed_id],
            embeddings=[embedding.tolist() if isinstance(embedding, np.ndarray) else embedding],
            metadatas=[metadata] if metadata else None
        )
        
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute("UPDATE tracks SET clp_embedding_id = ? WHERE id = ?", (embed_id, track_id))
        conn.commit()
        conn.close()
        return embed_id

    def get_embedding(self, embed_id: str) -> Optional[np.ndarray]:
        """Retrieves a vector from ChromaDB."""
        result = self.collection.get(ids=[embed_id], include=['embeddings'])
        if result and 'embeddings' in result and result['embeddings'] is not None and len(result['embeddings']) > 0:
            return np.array(result['embeddings'][0])
        return None

    def search_embeddings(self, query_vector: Union[np.ndarray, List[float]], n_results: int = 10) -> List[Dict[str, Any]]:
        """Performs a vector search in ChromaDB and joins with SQLite metadata."""
        results = self.collection.query(
            query_embeddings=[query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector],
            n_results=n_results
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []
            
        final_results: List[Dict[str, Any]] = []
        conn = self.get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        for i, embed_id in enumerate(results['ids'][0]):
            distance = results['distances'][0][i]
            cursor.execute("SELECT * FROM tracks WHERE clp_embedding_id = ?", (embed_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d['distance'] = float(distance)
                final_results.append(d)
        
        conn.close()
        return final_results

    def get_library_stats(self) -> Dict[str, Any]:
        """Returns high-level statistics about the audio library."""
        conn = self.get_conn()
        cursor = conn.cursor()
        
        stats: Dict[str, Any] = {}
        cursor.execute("SELECT COUNT(*) FROM tracks")
        stats['total_tracks'] = cursor.fetchone()[0]
        
        if stats['total_tracks'] > 0:
            cursor.execute("SELECT AVG(bpm), MIN(bpm), MAX(bpm) FROM tracks")
            bpm_stats = cursor.fetchone()
            stats['avg_bpm'] = round(bpm_stats[0], 2) if bpm_stats[0] else 0
            stats['min_bpm'] = round(bpm_stats[1], 2) if bpm_stats[1] else 0
            stats['max_bpm'] = round(bpm_stats[2], 2) if bpm_stats[2] else 0
            
            cursor.execute("SELECT harmonic_key, COUNT(*) as count FROM tracks GROUP BY harmonic_key ORDER BY count DESC")
            stats['key_distribution'] = dict(cursor.fetchall())
        
        conn.close()
        return stats

def init_db(db_path: str = "audio_library.db") -> None:
    DataManager(db_path=db_path)

if __name__ == "__main__":
    dm = DataManager()
    print(f"Databases initialized at {os.path.abspath(dm.db_path)} and {os.path.abspath(dm.vector_dir)}")
