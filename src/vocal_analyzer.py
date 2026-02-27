import os
import json
import re
import requests
from google import genai
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, Union, Tuple
from src.core.config import AppConfig

load_dotenv()

class VocalAnalyzer:
    """Analyzes vocal stems using a remote 4090 machine (Faster-Whisper) or Gemini fallback."""
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.client: Optional[genai.Client] = genai.Client(api_key=api_key)
        else:
            self.client = None
        
        self.remote_url: str = AppConfig.REMOTE_ANALYZE_URL
        self.ollama_url: str = AppConfig.OLLAMA_URL
        self.ollama_model: str = AppConfig.OLLAMA_MODEL

    def analyze_vocals(self, stem_path: str) -> Dict[str, Optional[str]]:
        """Sends vocal stem to remote 4090 server for local analysis."""
        if not os.path.exists(stem_path):
            return {"lyrics": None, "gender": None}

        try:
            with open(stem_path, 'rb') as f:
                response = requests.post(self.remote_url, files={'file': f}, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return {"lyrics": data.get("lyrics"), "gender": data.get("gender")}
        except: pass

        if not self.client:
            return {"lyrics": None, "gender": None}

        try:
            audio_file = self.client.files.upload(file=stem_path)
            prompt = "Listen to this audio track. Provide ONLY a valid JSON object with keys: 'lyrics' and 'gender'."
            response = self.client.models.generate_content(model="gemini-2.0-flash", contents=[audio_file, prompt])
            try: self.client.files.delete(name=audio_file.name)
            except: pass
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                res = json.loads(match.group())
                return {"lyrics": res.get("lyrics"), "gender": res.get("gender")}
        except: pass
        return {"lyrics": None, "gender": None}
