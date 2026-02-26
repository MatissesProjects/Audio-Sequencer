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
import pedalboard
from pedalboard import Pedalboard, Reverb, Chorus, HighpassFilter, LowpassFilter, Compressor

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
            y, sr = librosa.load(input_path, sr=16000)
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            
            gender = "Unknown"
            if f0 is not None and len(f0[voiced_flag]) > 0:
                mean_f0 = np.nanmean(f0[voiced_flag])
                gender = "Male" if mean_f0 < 165 else "Female"
                print(f"Detected Mean Pitch: {mean_f0:.1f}Hz -> {gender}")

            return {
                "lyrics": lyrics if lyrics else None,
                "gender": gender,
                "language": info.language
            }, 200
        except Exception as e:
            print(f"Analysis Error: {e}")
            return str(e), 500

@app.route('/analyze/sections', methods=['POST'])
def analyze_sections():
    print(f"[REQ] /analyze/sections from {request.remote_addr}")
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.wav")
        file.save(input_path)
        try:
            y, sr = librosa.load(input_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            features = np.vstack([mfcc, chroma])
            bound_frames = librosa.segment.agglomerative(features.T, 5) 
            bound_times = librosa.frames_to_time(bound_frames, sr=sr)
            
            sections = []
            for i in range(len(bound_times) - 1):
                start, end = bound_times[i], bound_times[i+1]
                y_seg = y[int(start*sr):int(end*sr)]
                if len(y_seg) == 0: continue
                rms = np.mean(librosa.feature.rms(y=y_seg))
                onset_env = librosa.onset.onset_strength(y=y_seg, sr=sr)
                density = np.mean(onset_env)
                
                label = "Verse" 
                if i == 0: label = "Intro"
                elif rms > 0.15 and density > 1.0: label = "Drop"
                elif density > 0.8: label = "Build"
                
                sections.append({"start": float(start), "end": float(end), "label": label, "energy": float(rms)})
            return {"sections": sections}, 200
        except Exception as e:
            print(f"Section Analysis Error: {e}")
            return str(e), 500

@app.route('/process/pad', methods=['POST'])
def process_pad():
    print(f"[REQ] /process/pad from {request.remote_addr}")
    if 'file' not in request.files:
        return "Missing 'file' parameter", 400
    
    file = request.files['file']
    duration = float(request.form.get('duration', 10.0))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.wav")
        file.save(input_path)
        try:
            y, sr = librosa.load(input_path, sr=44100)
            stft = librosa.stft(y, n_fft=4096, hop_length=512)
            mag, _ = librosa.magphase(stft)
            from scipy.ndimage import gaussian_filter1d
            mag_blurred = gaussian_filter1d(mag, sigma=15, axis=1)
            random_phase = np.exp(1j * np.random.uniform(0, 2*np.pi, size=stft.shape))
            y_pad = librosa.istft(mag_blurred * random_phase, hop_length=512)
            
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=400),
                Chorus(rate_hz=0.5, depth=0.5),
                Reverb(room_size=0.95, damping=0.5, wet_level=0.8, dry_level=0.2)
            ])
            y_pad = y_pad / (np.max(np.abs(y_pad)) + 1e-9)
            processed = board(y_pad.reshape(1, -1).astype(np.float32), sr)
            
            num_samples = int(sr * duration)
            if processed.shape[1] > num_samples:
                final_wav = processed[:, :num_samples]
            else:
                repeats = (num_samples // processed.shape[1]) + 1
                final_wav = np.tile(processed, (1, repeats))[:, :num_samples]

            byte_io = io.BytesIO()
            sf.write(byte_io, final_wav.T, sr, format='WAV')
            byte_io.seek(0)
            return send_file(byte_io, mimetype="audio/wav")
        except Exception as e:
            print(f"Pad Processing Error: {e}")
            return str(e), 500

@app.route('/process/harmonize', methods=['POST'])
def harmonize_vocals():
    """Generates a sophisticated 3-part backing harmony with formant-aware characters."""
    print(f"[REQ] /process/harmonize from {request.remote_addr}")
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "vocal.wav")
        file.save(input_path)
        try:
            y, sr = librosa.load(input_path, sr=44100)
            
            # 1. 'The Deep' (-5st, Low Formant)
            v_low = librosa.effects.pitch_shift(y, sr=sr, n_steps=-5)
            board_low = Pedalboard([
                LowpassFilter(cutoff_frequency_hz=2000), # Darker
                Chorus(rate_hz=0.3, depth=0.2),
                Compressor(threshold_db=-15, ratio=4)
            ])
            v_low_p = board_low(v_low.reshape(1, -1).astype(np.float32), sr).flatten()
            
            # 2. 'The Bright' (+7st, High Formant)
            v_high = librosa.effects.pitch_shift(y, sr=sr, n_steps=7)
            board_high = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=500), # Thinner/Airy
                Chorus(rate_hz=0.8, depth=0.3),
                Compressor(threshold_db=-20, ratio=6)
            ])
            v_high_p = board_high(v_high.reshape(1, -1).astype(np.float32), sr).flatten()
            
            # Mix: Orig (1.0) + Character Backing
            min_len = min(len(y), len(v_low_p), len(v_high_p))
            mixed = y[:min_len] + (v_low_p[:min_len] * 0.55) + (v_high_p[:min_len] * 0.45)
            
            # Master Polish
            master = Pedalboard([Reverb(room_size=0.4, wet_level=0.25)])
            final = master(mixed.reshape(1, -1).astype(np.float32), sr)
            
            byte_io = io.BytesIO()
            sf.write(byte_io, final.T, sr, format='WAV')
            byte_io.seek(0)
            return send_file(byte_io, mimetype="audio/wav")
        except Exception as e:
            print(f"Harmonization Error: {e}")
            return str(e), 500

@app.route('/process/continue', methods=['POST'])
def continue_audio():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    duration = float(request.form.get('duration', 4.0))
    prompt = request.form.get('prompt', 'continue this loop')
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.wav")
        file.save(input_path)
        try:
            y, sr = torchaudio.load(input_path)
            if model is None: return "MusicGen model not loaded.", 503
            model.set_generation_params(duration=duration)
            wav = model.generate_continuation(y.unsqueeze(0).to(device), sr, [prompt], progress=True)
            samples = wav[0].cpu().numpy().T
            byte_io = io.BytesIO()
            sf.write(byte_io, samples, 32000, format='WAV')
            byte_io.seek(0)
            return send_file(byte_io, mimetype="audio/wav")
        except Exception as e:
            print(f"Continuation Error: {e}")
            return str(e), 500

@app.route('/separate', methods=['POST'])
def separate():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.wav")
        file.save(input_path)
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        try:
            env = os.environ.copy()
            env["TORCHAUDIO_BACKEND"] = "soundfile"
            cmd = ["python", "-m", "demucs", "--out", output_dir, "--device", device, input_path]
            subprocess.run(cmd, check=True, env=env)
            stems_src = os.path.join(output_dir, "htdemucs", "input")
            if not os.path.exists(stems_src):
                subdirs = os.listdir(output_dir)
                if subdirs: stems_src = os.path.join(output_dir, subdirs[0], "input")
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, 'w') as zf:
                for stem in ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]:
                    p = os.path.join(stems_src, stem)
                    if os.path.exists(p): zf.write(p, stem)
            memory_file.seek(0)
            return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name="stems.zip")
        except Exception as e:
            print(f"Separation Error: {e}")
            return str(e), 500

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt, duration = data.get('prompt', 'cinematic riser'), data.get('duration', 4)
    top_k, top_p = data.get('top_k', 250), data.get('top_p', 0)
    temp, cfg = data.get('temperature', 1.0), data.get('cfg_coef', 3.0)
    if model is None: return "MusicGen model not loaded.", 503
    try:
        model.set_generation_params(duration=duration, top_k=top_k, top_p=top_p, temperature=temp, cfg_coef=cfg)
        wav = model.generate([prompt] if isinstance(prompt, str) else prompt, progress=True)
        samples = wav[0].cpu().numpy().T
        byte_io = io.BytesIO()
        sf.write(byte_io, samples, 32000, format='WAV')
        byte_io.seek(0)
        return send_file(byte_io, mimetype="audio/wav")
    except Exception as e:
        print(f"Generation Error: {e}")
        return str(e), 500

@app.route('/process/gender_transform', methods=['POST'])
def gender_transform():
    print(f"[REQ] /process/gender_transform from {request.remote_addr}")
    if 'file' not in request.files: return "No file part", 400
    file = request.files['file']
    target, steps = request.form.get('target', 'female'), float(request.form.get('steps', 0))
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.wav")
        file.save(input_path)
        try:
            y, sr = librosa.load(input_path, sr=44100)
            if steps == 0: steps = 12 if target == 'female' else -12
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            if target == 'female':
                board = Pedalboard([HighpassFilter(cutoff_frequency_hz=350), Chorus(rate_hz=0.2, depth=0.2), Compressor(threshold_db=-18, ratio=4)])
            else:
                board = Pedalboard([LowpassFilter(cutoff_frequency_hz=3500), Chorus(rate_hz=0.5, depth=0.1), Compressor(threshold_db=-12, ratio=3)])
            processed = board(y_shifted.reshape(1, -1).astype(np.float32), sr)
            byte_io = io.BytesIO()
            sf.write(byte_io, processed.T, sr, format='WAV')
            byte_io.seek(0)
            return send_file(byte_io, mimetype="audio/wav")
        except Exception as e:
            print(f"Gender Transform Error: {e}")
            return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
