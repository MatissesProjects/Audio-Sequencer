import os
import json
import re
import requests
from google import genai
from dotenv import load_dotenv
from src.core.config import AppConfig

load_dotenv()

class VocalAnalyzer:
    """Analyzes vocal stems using a remote 4090 machine (Faster-Whisper) or Gemini fallback."""
    
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
        
        self.remote_url = AppConfig.REMOTE_ANALYZE_URL

    def analyze_vocals(self, stem_path):
        """Sends vocal stem to remote 4090 server for local analysis."""
        if not os.path.exists(stem_path):
            return {"lyrics": None, "gender": None}

        # 1. Try Remote 4090 Analysis (Faster-Whisper + Librosa)
        try:
            print(f"Requesting Remote Vocal Analysis for: {os.path.basename(stem_path)}...")
            with open(stem_path, 'rb') as f:
                response = requests.post(
                    self.remote_url,
                    files={'file': f},
                    timeout=60
                )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Remote analysis success: {data.get('gender')} | {data.get('language')}")
                return {
                    "lyrics": data.get("lyrics"),
                    "gender": data.get("gender")
                }
            else:
                print(f"Remote analysis failed (HTTP {response.status_code}). Trying Gemini fallback.")
        except Exception as e:
            print(f"Remote analysis error: {e}. Trying Gemini fallback.")

        # 2. Gemini Fallback
        if not self.client:
            return {"lyrics": None, "gender": None}

        try:
            print(f"Uploading {stem_path} for Gemini vocal analysis (Fallback)...")
            audio_file = self.client.files.upload(file=stem_path)

            prompt = """
            Listen to this audio track containing vocals.
            Provide ONLY a valid JSON object with the following keys:
            - 'lyrics': The transcribed text.
            - 'gender': The perceived gender ('Male', 'Female', or 'Unknown').
            """
            
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[audio_file, prompt]
            )
            
            try: self.client.files.delete(name=audio_file.name)
            except: pass

            text = response.text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                res = json.loads(match.group())
                return {"lyrics": res.get("lyrics"), "gender": res.get("gender")}
        except Exception as e:
            print(f"Gemini fallback error: {e}")
            
        return {"lyrics": None, "gender": None}
