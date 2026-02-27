import os
from typing import List, Dict, Optional, Any, Union, Tuple

class AppConfig:
    """Centralized configuration for the AudioSequencer AI."""
    
    # Audio Settings
    SAMPLE_RATE: int = 44100
    DEFAULT_BPM: float = 124.0
    
    # Paths (relative to project root)
    DB_PATH: str = "audio_library.db"
    VECTOR_DB_DIR: str = "vector_db"
    CACHE_DIR: str = "render_cache"
    STEMS_DIR: str = "stems_library"
    GENERATED_ASSETS_DIR: str = "generated_assets"
    
    # Remote AI Server (RTX 4090 machine)
    REMOTE_AI_HOST: str = os.getenv("REMOTE_AI_HOST", "192.168.4.165") 
    REMOTE_AI_PORT: str = os.getenv("REMOTE_AI_PORT", "5001")
    REMOTE_GEN_URL: str = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/generate"
    REMOTE_SEP_URL: str = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/separate"
    REMOTE_ANALYZE_URL: str = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/analyze"
    REMOTE_PAD_URL: str = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/process/pad"
    REMOTE_SECTIONS_URL: str = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/analyze/sections"
    REMOTE_HARMONIZE_URL: str = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/process/harmonize"
    REMOTE_CONTINUE_URL: str = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/process/continue"
    REMOTE_GENDER_URL: str = f"http://{REMOTE_AI_HOST}:{REMOTE_AI_PORT}/process/gender_transform"
    
    # Ollama Settings
    OLLAMA_URL: str = f"http://{REMOTE_AI_HOST}:11434/api/generate"
    OLLAMA_MODEL: str = "qwen3:8b" 
    
    # Processing Settings
    DEFAULT_DUCKING_DEPTH: float = 0.7
    CROSSFADE_MS: int = 500
    
    @classmethod
    def ensure_dirs(cls) -> None:
        """Ensures all required directories exist."""
        for d in [cls.VECTOR_DB_DIR, cls.CACHE_DIR, cls.STEMS_DIR, cls.GENERATED_ASSETS_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def get_stems_path(cls, filename: str) -> str:
        """Returns a standardized stems directory for a given track."""
        safe_name = str(filename).replace(" ", "_").replace(".", "_")
        return os.path.join(cls.STEMS_DIR, safe_name)
