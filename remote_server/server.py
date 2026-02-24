from flask import Flask, request, send_file
import torch
import torchaudio
from audiocraft.models import MusicGen
import io
import os

app = Flask(__name__)

# Initialize model on 4090 (GPU)
print("--- 🚀 AudioSequencer 4090 Generator Server ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading MusicGen onto {device}...")

# You can upgrade to 'facebook/musicgen-medium' if your 4090 has the VRAM
model = MusicGen.get_pretrained('facebook/musicgen-small', device=device)
model.set_generation_params(duration=4)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', 'cinematic riser')
    duration = data.get('duration', 4)
    
    print(f"Generating: '{prompt}' ({duration}s)...")
    
    try:
        # Set specific duration for this call
        model.set_generation_params(duration=duration)
        
        # Generate
        wav = model.generate([prompt], progress=True) # [1, 1, samples]
        
        # Convert to Bytes for transport
        # torchaudio expectations: (channels, samples)
        samples = wav[0].cpu()
        
        # Save to memory-buffered WAV
        byte_io = io.BytesIO()
        torchaudio.save(byte_io, samples, 32000, format="wav")
        byte_io.seek(0)
        
        return send_file(byte_io, mimetype="audio/wav")
        
    except Exception as e:
        print(f"Generation Error: {e}")
        return str(e), 500

if __name__ == '__main__':
    # Listen on all interfaces so your main machine can connect
    app.run(host='0.0.0.0', port=5000)
