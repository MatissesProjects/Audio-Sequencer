import sqlite3
import os

def init_db(db_path="audio_library.db"):
    """Initializes the SQLite database with the required schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Table for audio file metadata and MIR features
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
            date_added DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_analyzed DATETIME,
            clp_embedding_id TEXT -- Link to vector database ID
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {os.path.abspath(db_path)}")

if __name__ == "__main__":
    init_db()
