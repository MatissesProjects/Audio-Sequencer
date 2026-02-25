import os
import numpy as np
import soundfile as sf
from google import genai
from pedalboard import Pedalboard, Reverb, Delay, HighpassFilter, LowpassFilter
from dotenv import load_dotenv
import re
import json

load_dotenv()

class TransitionGenerator:
    """Uses Gemini to orchestrate transitions and a remote 4090 machine to generate them via network."""
    
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
            print("Warning: GOOGLE_API_KEY not found. Gemini orchestration disabled.")
        
        from src.core.config import AppConfig
        self.remote_url = AppConfig.REMOTE_GEN_URL

    def get_transition_params(self, track_a, track_b, type_context=""):
        """Asks Gemini for transition characteristics based on track metadata."""
        if not self.client:
            return {"noise_type": "white", "filter_type": "highpass", "reverb_amount": 0.7, "prompt": "cinematic riser sweep"}

        bpm_a = track_a.get('bpm', 120)
        key_a = track_a.get('harmonic_key') or track_a.get('key') or 'Unknown'
        name_a = track_a.get('filename', 'Track A')
        
        bpm_b = track_b.get('bpm', 120)
        key_b = track_b.get('harmonic_key') or track_b.get('key') or 'Unknown'
        name_b = track_b.get('filename', 'Track B')

        prompt = f"""
        Analyze these two tracks and describe a 4-second audio transition to bridge them.
        {type_context}
        Track A: {name_a}, {bpm_a} BPM, Key: {key_a}
        Track B: {name_b}, {bpm_b} BPM, Key: {key_b}
        
        Provide ONLY valid JSON output with keys: 
        'noise_type' (white, pink, or brown),
        'filter_type' (lowpass or highpass),
        'reverb_amount' (float 0.0 to 1.0),
        'prompt' (A text prompt for MusicGen generation, e.g. 'heavy sub bass impact with reverb in {key_b}, {bpm_b} bpm'),
        'description' (brief vibe description).
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )
            text = response.text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"noise_type": "white", "filter_type": "highpass", "reverb_amount": 0.7, "prompt": "cinematic riser"}
        except Exception as e:
            print(f"Gemini error: {e}")
            return {"noise_type": "white", "filter_type": "highpass", "reverb_amount": 0.5, "prompt": "cinematic riser"}

    def get_journey_structure(self, depth=0):
        """Asks Gemini for a musically logical sequence of blocks for the current depth."""
        if not self.client:
            # Fallback deterministic structures
            if depth == 0:
                return [
                    {'name': 'Intro', 'dur': 16000},
                    {'name': 'Verse 1', 'dur': 32000},
                    {'name': 'Build', 'dur': 16000},
                    {'name': 'Drop', 'dur': 32000},
                    {'name': 'Verse 2', 'dur': 32000},
                    {'name': 'Outro', 'dur': 20000}
                ]
            elif depth == 1:
                return [
                    {'name': 'Connect', 'dur': 8000},
                    {'name': 'Verse 3', 'dur': 32000},
                    {'name': 'Bridge', 'dur': 24000},
                    {'name': 'Build', 'dur': 16000},
                    {'name': 'Power Drop', 'dur': 48000},
                    {'name': 'Transition', 'dur': 12000}
                ]
            else:
                return [
                    {'name': 'Connect', 'dur': 16000},
                    {'name': 'Atmospheric Breakdown', 'dur': 32000},
                    {'name': 'Build', 'dur': 16000},
                    {'name': 'Grand Finale', 'dur': 64000},
                    {'name': 'Extended Outro', 'dur': 40000}
                ]

        prompt = f"""
        Suggest a list of musical blocks for an AI-generated music journey at depth {depth}.
        Depth 0 is the start of the mix. Depth 1+ are extensions.
        
        Provide ONLY valid JSON output as a list of objects with keys:
        'name' (e.g., Intro, Verse, Build, Drop, Bridge, Connect, Outro),
        'dur' (duration in milliseconds, usually multiples of 4000 or 8000).
        
        Ensure the structure flows logically from the previous sections.
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )
            text = response.text
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"Structure Gemini error: {e}")
        
        return self.get_journey_structure(depth=-1) # Recursive call to fallback logic

    def generate_riser(self, duration_sec, bpm, output_path, params=None):
        """Generates a procedural noise riser OR calls remote 4090 for neural riser."""
        prompt = params.get('prompt') if params else None
        
        if prompt:
            try:
                import requests
                print(f"Requesting Remote Neural Riser: '{prompt}'...")
                response = requests.post(
                    self.remote_url,
                    json={'prompt': prompt, 'duration': duration_sec},
                    timeout=45 # Increased timeout for remote gen
                )
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    return output_path
                else:
                    print(f"Remote server error: {response.status_code}")
            except Exception as e:
                print(f"Remote neural generation failed: {e}. Falling back to procedural noise.")

        # Fallback: Procedural Noise
        sr = 44100
        num_samples = int(sr * duration_sec)
        # ... (remaining noise logic)
        
        noise_type = params.get('noise_type', 'white') if params else 'white'
        if noise_type == 'pink':
            white = np.random.uniform(-1, 1, num_samples)
            b = [0.0405096, -0.0601927, 0.056985, -0.0328239, 0.00959452]
            noise = np.convolve(white, b, mode='same')
        else:
            noise = np.random.uniform(-1, 1, num_samples)

        # 2. Apply smoother cubic volume ramp (The Riser)
        fade_in = np.linspace(0, 1, num_samples) ** 3
        noise = noise * fade_in
        
        # Force the first 10ms to be absolute silence to avoid "weird noise" at start
        silence_samples = int(sr * 0.01)
        noise[:silence_samples] = 0

        filter_type = params.get('filter_type', 'highpass') if params else 'highpass'
        rev_amt = params.get('reverb_amount', 0.5) if params else 0.5
        
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=200) if filter_type == 'highpass' else LowpassFilter(cutoff_frequency_hz=5000),
            Reverb(room_size=rev_amt),
            Delay(delay_seconds=0.25, feedback=0.3)
        ])
        
        noise_input = noise.reshape(1, -1).astype(np.float32)
        processed = board(noise_input, sr)
        
        # Normalize
        peak = np.max(np.abs(processed))
        if peak > 0:
            processed = processed / peak
        
        sf.write(output_path, processed.flatten(), sr)
        return output_path
