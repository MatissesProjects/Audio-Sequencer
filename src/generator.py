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
    """Uses Gemini to orchestrate transitions and local DSP to generate them."""
    
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
            print("Warning: GOOGLE_API_KEY not found. Gemini orchestration disabled.")

    def get_transition_params(self, track_a, track_b):
        """Asks Gemini for transition characteristics based on track metadata."""
        if not self.client:
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
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            text = response.text
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
