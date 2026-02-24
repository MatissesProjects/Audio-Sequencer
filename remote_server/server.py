from flask import Flask, request, send_file
import torch
import torchaudio
from audiocraft.models import MusicGen
import io
import os
import shutil
import tempfile
import zipfile
import subprocess

app = Flask(__name__)

# Initialize MusicGen on 4090 (GPU)
print("--- 🚀 AudioSequencer 4090 Generator Server ---")
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.route('/separate', methods=['POST'])
def separate():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.wav")
        file.save(input_path)
        
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Separating Stems for: {file.filename}...")
        
        try:
            # Call Demucs via subprocess for reliability and easy CUDA handling
            # --two-stems=vocals can be used for faster processing if we only want vocals/other
            # but we want full 4-stem separation
            cmd = [
                "python", "-m", "demucs", 
                "--out", output_dir, 
                "--device", device,
                input_path
            ]
            subprocess.run(cmd, check=True)
            
            # Demucs creates a nested structure: output/htdemucs/input/drums.wav etc.
            model_name = "htdemucs" # default model
            stems_src = os.path.join(output_dir, model_name, "input")
            
            if not os.path.exists(stems_src):
                # Check for alternative model names if demucs changed defaults
                subdirs = os.listdir(output_dir)
                if subdirs:
                    stems_src = os.path.join(output_dir, subdirs[0], "input")

            # Create ZIP in memory
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, 'w') as zf:
                for stem in ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]:
                    stem_path = os.path.join(stems_src, stem)
                    if os.path.exists(stem_path):
                        zf.write(stem_path, stem)
            
            memory_file.seek(0)
            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f"{os.path.splitext(file.filename)[0]}_stems.zip"
            )
            
        except Exception as e:
            print(f"Separation Error: {e}")
            return str(e), 500

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
