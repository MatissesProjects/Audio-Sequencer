import requests
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.config import AppConfig

def test_pro_features():
    print("--- 🚀 4090 Pro Feature Test ---")
    
    # Use an example file if available, otherwise skip
    test_file = "musicExamples/track_1771971118.mp3"
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found. Skipping audio tests.")
        return

    # 1. Test Spectral Pad
    print("1. Testing Spectral Pad-ification...")
    try:
        with open(test_file, 'rb') as f:
            r = requests.post(AppConfig.REMOTE_PAD_URL, files={'file': f}, data={'duration': 5.0})
        if r.status_code == 200:
            with open("generated_assets/test_pad.wav", 'wb') as f:
                f.write(r.content)
            print("✅ Spectral Pad Success! (generated_assets/test_pad.wav)")
        else:
            print(f"❌ Spectral Pad Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"❌ Spectral Pad Error: {e}")

    # 2. Test Harmonization
    print("2. Testing Neural Harmonization...")
    try:
        # For test, we use the same file (assuming it has some vocal-like content)
        with open(test_file, 'rb') as f:
            r = requests.post(AppConfig.REMOTE_HARMONIZE_URL, files={'file': f})
        if r.status_code == 200:
            with open("generated_assets/test_harmony.wav", 'wb') as f:
                f.write(r.content)
            print("✅ Harmonization Success! (generated_assets/test_harmony.wav)")
        else:
            print(f"❌ Harmonization Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"❌ Harmonization Error: {e}")

    # 3. Test Structural Analysis
    print("3. Testing Structural Analysis (Deep MIR)...")
    try:
        with open(test_file, 'rb') as f:
            r = requests.post(AppConfig.REMOTE_SECTIONS_URL, files={'file': f})
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Analysis Success! Found {len(data['sections'])} sections.")
            for s in data['sections']:
                print(f"  - {s['label']}: {s['start']:.1f}s -> {s['end']:.1f}s")
        else:
            print(f"❌ Analysis Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"❌ Analysis Error: {e}")

if __name__ == "__main__":
    os.makedirs("generated_assets", exist_ok=True)
    test_pro_features()
