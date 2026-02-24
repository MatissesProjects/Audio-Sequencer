import os

class AppConfig:
    """Centralized configuration for the AudioSequencer AI."""
    
    # Audio Settings
    SAMPLE_RATE = 44100
    DEFAULT_BPM = 124.0
    
    # Paths (relative to project root)
    DB_PATH = "audio_library.db"
    VECTOR_DB_DIR = "vector_db"
    CACHE_DIR = "render_cache"
    STEMS_DIR = "stems_library"
    GENERATED_ASSETS_DIR = "generated_assets"
    
    # Processing Settings
    DEFAULT_DUCKING_DEPTH = 0.7
    CROSSFADE_MS = 500
    
    @classmethod
    def ensure_dirs(cls):
        """Ensures all required directories exist."""
        for d in [cls.VECTOR_DB_DIR, cls.CACHE_DIR, cls.STEMS_DIR, cls.GENERATED_ASSETS_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def get_stems_path(cls, filename):
        """Returns a standardized stems directory for a given track."""
        safe_name = str(filename).replace(" ", "_").replace(".", "_")
        return os.path.join(cls.STEMS_DIR, safe_name)
