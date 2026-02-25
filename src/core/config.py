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
    
    # Remote AI Server (RTX 4090 machine)
    # Set REMOTE_AI_HOST environment variable to switch (e.g., "192.168.1.x")
    REMOTE_AI_HOST = os.getenv("REMOTE_AI_HOST", "matisse-INTEL") 
    REMOTE_AI_PORT = os.getenv("REMOTE_AI_PORT", "5001")
    REMOTE_GEN_URL = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/generate"
    REMOTE_SEP_URL = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/separate"
    REMOTE_ANALYZE_URL = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/analyze"
    REMOTE_PAD_URL = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/process/pad"
    REMOTE_SECTIONS_URL = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/analyze/sections"
    REMOTE_HARMONIZE_URL = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/process/harmonize"
    REMOTE_CONTINUE_URL = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/process/continue"
    
    # Ollama Settings (Local AI on remote 4090)
    OLLAMA_URL = f"http://{REMOTE_AI_HOST}:11434/api/generate"
    OLLAMA_MODEL = "qwen3:8b" 

    
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
