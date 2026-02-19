import os
import sqlite3
import pytest
from src.database import init_db

def test_database_init(tmp_path):
    db_file = tmp_path / "test_audio.db"
    init_db(str(db_file))
    
    assert os.path.exists(db_file)
    
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracks';")
    assert cursor.fetchone() is not None
    conn.close()

# More tests for AnalysisModule would typically require a mock or a small test .wav file
