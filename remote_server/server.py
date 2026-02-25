from flask import Flask, request, send_file
import torch
import torchaudio
import soundfile as sf
from audiocraft.models import MusicGen
import io
import os
import shutil
import tempfile
import zipfile
import subprocess
from faster_whisper import WhisperModel
import librosa
import numpy as np

app = Flask(__name__)

# Initialize Models on 4090 (GPU)
print("--- 🚀 AudioSequencer 4090 Generator Server ---")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    print("Loading MusicGen model...")
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    print("MusicGen Loaded.")
except Exception as e:
    print(f"Failed to load MusicGen: {e}")
    model = None

try:
    print("Loading Whisper model (Faster-Whisper)...")
    # Using 'base' or 'small' for speed, 'large-v3' for maximum accuracy
    whisper_model = WhisperModel("small", device=device, compute_type="float16" if device=="cuda" else "int8")
    print("Whisper Loaded.")
except Exception as e:
    print(f"Failed to load Whisper: {e}")
    whisper_model = None

@app.route('/')
def health_check():
    return {"status": "online", "device": device, "services": ["Separation", "Generation", "Analysis"]}, 200

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if whisper_model is None:
        return "Whisper model not available.", 503

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "vocal.wav")
        file.save(input_path)
        
        print(f"Analyzing Vocals for: {file.filename}...")
        
        try:
            # 1. Transcription (Whisper)
            segments, info = whisper_model.transcribe(input_path, beam_size=5)
            lyrics = " ".join([s.text for s in segments]).strip()
            
            # 2. Gender Detection (Pitch-based heuristic)
            # Male usually 85-155Hz, Female 165-255Hz
            y, sr = librosa.load(input_path, sr=16000)
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            
            gender = "Unknown"
            if f0 is not None and len(f0[voiced_flag]) > 0:
                mean_f0 = np.nanmean(f0[voiced_flag])
                if mean_f0 < 165:
                    gender = "Male"
                else:
                    gender = "Female"
                print(f"Detected Mean Pitch: {mean_f0:.1f}Hz -> {gender}")

            return {
                "lyrics": lyrics if lyrics else None,
                "gender": gender,
                "language": info.language
            }, 200
            
        except Exception as e:
            print(f"Analysis Error: {e}")
            return str(e), 500

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
            # Force torchaudio to use 'soundfile' backend to avoid torchcodec issues
            env = os.environ.copy()
            env["TORCHAUDIO_BACKEND"] = "soundfile"
            
            cmd = [
                "python", "-m", "demucs", 
                "--out", output_dir, 
                "--device", device,
                input_path
            ]
            subprocess.run(cmd, check=True, env=env)
            
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
    # High-quality settings for MusicGen
    top_k = data.get('top_k', 250)
    top_p = data.get('top_p', 0)
    temperature = data.get('temperature', 1.0)
    cfg_coef = data.get('cfg_coef', 3.0)
    
    print(f"Generating: '{prompt}' ({duration}s) | CFG: {cfg_coef} | Temp: {temperature}...")
    
    if model is None:
        return "MusicGen model failed to load on server startup.", 503
    
    try:
        model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coeff=cfg_coef
        )
        
        # Multi-prompt generation (if array provided)
        prompts = [prompt] if isinstance(prompt, str) else prompt
        wav = model.generate(prompts, progress=True)
        
        # Convert to Bytes for transport
        # torchaudio expectations: (channels, samples)
        # soundfile expectations: (samples, channels)
        samples = wav[0].cpu().numpy() # [channels, samples]
        
        # Transpose for soundfile: [samples, channels]
        samples = samples.T 

        # Save to memory-buffered WAV
        byte_io = io.BytesIO()
        sf.write(byte_io, samples, 32000, format='WAV')
        byte_io.seek(0)
        
        return send_file(byte_io, mimetype="audio/wav")
        
    except Exception as e:
        print(f"Generation Error: {e}")
        return str(e), 500

if __name__ == '__main__':
    # Listen on all interfaces so your main machine can connect
    app.run(host='0.0.0.0', port=5001)
