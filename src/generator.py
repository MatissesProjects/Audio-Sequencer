import os
import numpy as np
import soundfile as sf
from google import genai
from pedalboard import Pedalboard, Reverb, Delay, HighpassFilter, LowpassFilter
from dotenv import load_dotenv
import re
import json
from typing import List, Dict, Optional, Any, Union, Tuple

load_dotenv()

class TransitionGenerator:
    """Uses Ollama (Local) or Gemini to orchestrate transitions and a remote 4090 machine to generate them."""
    
    def __init__(self, api_key: Optional[str] = None):
        from src.core.config import AppConfig
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
        
        self.remote_url: str = AppConfig.REMOTE_GEN_URL
        self.ollama_url: str = AppConfig.OLLAMA_URL
        self.ollama_model: str = AppConfig.OLLAMA_MODEL

    def _call_ai(self, prompt: str, system_instruction: str = "") -> Optional[str]:
        """Tries Ollama first, then Gemini as fallback."""
        try:
            import requests
            full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
            print(f"[AI] Attempting Ollama connection: {self.ollama_url}")
            response = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "prompt": full_prompt, "stream": False, "format": "json"},
                timeout=15
            )
            if response.status_code == 200:
                return response.json().get("response")
        except: pass

        if self.client:
            try:
                response = self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                return response.text
            except: pass
        return None

    def get_transition_params(self, track_a: Dict[str, Any], track_b: Dict[str, Any], type_context: str = "") -> Dict[str, Any]:
        """Asks AI for transition characteristics based on track metadata."""
        bpm_a = track_a.get('bpm', 120); key_a = track_a.get('harmonic_key') or track_a.get('key') or 'Unknown'
        name_a = track_a.get('filename', 'Track A'); lyrics_a = track_a.get('vocal_lyrics')
        bpm_b = track_b.get('bpm', 120); key_b = track_b.get('harmonic_key') or track_b.get('key') or 'Unknown'
        name_b = track_b.get('filename', 'Track B'); lyrics_b = track_b.get('vocal_lyrics')

        l_ctx = ""
        if lyrics_a or lyrics_b:
            l_ctx = "Consider the lyrics/vocals to make the transition make narrative sense:\n"
            if lyrics_a: l_ctx += f"Track A Lyrics: '{lyrics_a}'\n"
            if lyrics_b: l_ctx += f"Track B Lyrics: '{lyrics_b}'\n"

        prompt = f"""
        Analyze these two tracks and describe a 4-second audio transition to bridge them.
        {type_context} {l_ctx}
        Track A: {name_a}, {bpm_a} BPM, Key: {key_a}
        Track B: {name_b}, {bpm_b} BPM, Key: {key_b}
        Provide ONLY valid JSON: 'noise_type', 'filter_type', 'reverb_amount', 'prompt', 'description'.
        """
        res_text = self._call_ai(prompt)
        if res_text:
            match = re.search(r'\{.*\}', res_text, re.DOTALL)
            if match:
                try: return json.loads(match.group())
                except: pass
        return {"noise_type": "white", "filter_type": "highpass", "reverb_amount": 0.7, "prompt": "cinematic riser"}

    def get_journey_structure(self, depth: int = 0) -> List[Dict[str, Any]]:
        """Asks AI for a musically logical sequence of blocks for the current depth."""
        prompt = f"""
        Suggest a list of musical blocks for an AI journey at depth {depth}. 5-7 blocks.
        If depth > 0, end with 'Transition' or 'Connector', NOT 'Outro'.
        Provide ONLY valid JSON list: 'name', 'dur' (16000-32000ms).
        """
        res_text = self._call_ai(prompt)
        if res_text:
            match = re.search(r'\[.*\]', res_text, re.DOTALL)
            if match:
                try: return json.loads(match.group())
                except: pass
        return self.get_journey_structure_fallback(depth=depth)

    def get_journey_structure_fallback(self, depth: int = 0) -> List[Dict[str, Any]]:
        """Fallback deterministic structures."""
        if depth == 0:
            return [{'name': 'Intro', 'dur': 16000}, {'name': 'Verse 1', 'dur': 32000}, {'name': 'Build', 'dur': 16000}, {'name': 'Drop', 'dur': 32000}, {'name': 'Verse 2', 'dur': 32000}, {'name': 'Transition', 'dur': 16000}]
        elif depth == 1:
            return [{'name': 'Connect', 'dur': 16000}, {'name': 'Verse 3', 'dur': 32000}, {'name': 'Bridge', 'dur': 24000}, {'name': 'Build', 'dur': 16000}, {'name': 'Power Drop', 'dur': 32000}, {'name': 'Transition', 'dur': 16000}]
        else:
            return [{'name': 'Connect', 'dur': 16000}, {'name': 'Atmospheric Breakdown', 'dur': 32000}, {'name': 'Build', 'dur': 16000}, {'name': 'Grand Finale', 'dur': 32000}, {'name': 'Transition', 'dur': 16000}]

    def generate_riser(self, duration_sec: float, bpm: float, output_path: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generates a procedural noise riser OR calls remote 4090 for neural riser."""
        prompt = params.get('prompt') if params else None
        if prompt:
            try:
                import requests
                response = requests.post(self.remote_url, json={'prompt': prompt, 'duration': duration_sec}, timeout=45)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f: f.write(response.content)
                    return output_path
            except: pass

        sr = 44100; num_samples = int(sr * duration_sec)
        noise_type = params.get('noise_type', 'white') if params else 'white'
        if noise_type == 'pink':
            white = np.random.uniform(-1, 1, num_samples); b = [0.0405096, -0.0601927, 0.056985, -0.0328239, 0.00959452]
            noise = np.convolve(white, b, mode='same')
        else: noise = np.random.uniform(-1, 1, num_samples)
        noise = noise * (np.linspace(0, 1, num_samples) ** 3)
        noise[:int(sr * 0.01)] = 0
        f_type = params.get('filter_type', 'highpass') if params else 'highpass'; rev_amt = params.get('reverb_amount', 0.5) if params else 0.5
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=200) if f_type == 'highpass' else LowpassFilter(cutoff_frequency_hz=5000), Reverb(room_size=rev_amt), Delay(delay_seconds=0.25, feedback=0.3)])
        processed = board(noise.reshape(1, -1).astype(np.float32), sr)
        peak = np.max(np.abs(processed))
        if peak > 0: processed /= peak
        sf.write(output_path, processed.flatten(), sr)
        return output_path
