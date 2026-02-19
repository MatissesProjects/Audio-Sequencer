import os
import numpy as np
import soundfile as sf
import google.generativeai as genai
from pedalboard import Pedalboard, Reverb, Delay, HighPassFilter, LowPassFilter
from dotenv import load_dotenv

load_dotenv()

import re
import json

class TransitionGenerator:
    """Uses Gemini to orchestrate transitions and local DSP to generate them."""
    
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            print("Warning: GOOGLE_API_KEY not found. Gemini orchestration disabled.")

    def get_transition_params(self, track_a, track_b):
        """Asks Gemini for transition characteristics based on track metadata."""
        if not self.model:
            return {"noise_type": "white", "filter_type": "highpass", "reverb_amount": 0.7}

        prompt = f"""
        Analyze these two tracks and describe a 4-second audio transition (riser or sweep) to bridge them.
        Track A: {track_a['filename']}, {track_a['bpm']} BPM, Key: {track_a['harmonic_key']}
        Track B: {track_b['filename']}, {track_b['bpm']} BPM, Key: {track_b['harmonic_key']}
        
        Provide ONLY valid JSON output with keys: 
        'noise_type' (white, pink, or brown),
        'filter_type' (lowpass or highpass),
        'reverb_amount' (float 0.0 to 1.0),
        'description' (brief vibe description).
        """
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            # Use regex to find JSON in case Gemini adds conversational filler
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"noise_type": "white", "filter_type": "highpass", "reverb_amount": 0.7}
        except Exception as e:
            print(f"Gemini error: {e}")
            return {"noise_type": "white", "filter_type": "highpass", "reverb_amount": 0.5}

    def generate_riser(self, duration_sec, bpm, output_path, params=None):
        """Generates a procedural noise riser with DSP effects."""
        sr = 44100
        num_samples = int(sr * duration_sec)
        
        # 1. Generate base noise
        noise_type = params.get('noise_type', 'white') if params else 'white'
        if noise_type == 'pink':
            # Simplified pink noise
            white = np.random.uniform(-1, 1, num_samples)
            b = [0.0405096, -0.0601927, 0.056985, -0.0328239, 0.00959452]
            noise = np.convolve(white, b, mode='same')
        else:
            noise = np.random.uniform(-1, 1, num_samples)

        # 2. Apply exponential volume ramp (The Riser)
        fade_in = np.linspace(0, 1, num_samples) ** 2
        noise = noise * fade_in

        # 3. Apply Pedalboard Effects
        filter_type = params.get('filter_type', 'highpass') if params else 'highpass'
        rev_amt = params.get('reverb_amount', 0.5) if params else 0.5
        
        board = Pedalboard([
            HighPassFilter(cutoff_frequency_hz=200) if filter_type == 'highpass' else LowPassFilter(cutoff_frequency_hz=5000),
            Reverb(room_size=rev_amt),
            Delay(delay_seconds=0.25, feedback=0.3)
        ])
        
        # Reshape for pedalboard (1, samples)
        noise_input = noise.reshape(1, -1).astype(np.float32)
        processed = board(noise_input, sr)
        
        # Normalize
        processed = processed / np.max(np.abs(processed))
        
        sf.write(output_path, processed.flatten(), sr)
        return output_path

if __name__ == "__main__":
    gen = TransitionGenerator()
    # Mock metadata
    t1 = {"filename": "chill.mp3", "bpm": 120, "harmonic_key": "C"}
    t2 = {"filename": "heavy.mp3", "bpm": 128, "harmonic_key": "G"}
    
    p = gen.get_transition_params(t1, t2)
    print(f"Gemini suggested: {p}")
    out = gen.generate_riser(4.0, 120, "test_riser.wav", params=p)
    print(f"Generated transition at: {out}")
