import requests
import json
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.config import AppConfig

def test_ollama_connectivity():
    print(f"--- 🦙 Ollama Connectivity Test ---")
    print(f"Target URL: {AppConfig.OLLAMA_URL}")
    print(f"Target Model: {AppConfig.OLLAMA_MODEL}")
    
    # 1. Test Base Connection (Tags API)
    tags_url = AppConfig.OLLAMA_URL.replace("/api/generate", "/api/tags")
    print(f"\n1. Checking Base Connectivity (fetching models)...")
    try:
        response = requests.get(tags_url, timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            print(f"✅ Connection Successful!")
            print(f"Available Models: {', '.join(model_names)}")
            
            found = False
            for m in model_names:
                if AppConfig.OLLAMA_MODEL in m:
                    found = True
                    break
            
            if found:
                print(f"✅ Target model '{AppConfig.OLLAMA_MODEL}' is available.")
            else:
                print(f"❌ Target model '{AppConfig.OLLAMA_MODEL}' NOT found in list.")
        else:
            print(f"❌ Server returned error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama at {tags_url}: {e}")
        return

    # 2. Test Generation (JSON Mode)
    print(f"\n2. Testing Generation (JSON Mode)...")
    test_prompt = "Explain what a 'riser' is in 10 words. Respond only in valid JSON with key 'explanation'."
    try:
        response = requests.post(
            AppConfig.OLLAMA_URL,
            json={
                "model": AppConfig.OLLAMA_MODEL,
                "prompt": test_prompt,
                "stream": False,
                "format": "json"
            },
            timeout=15
        )
        if response.status_code == 200:
            result = response.json().get("response")
            print(f"✅ Generation Successful!")
            print(f"Response: {result}")
            try:
                parsed = json.loads(result)
                print(f"✅ JSON is valid: {parsed}")
            except:
                print(f"⚠️ Response is NOT valid JSON.")
        else:
            print(f"❌ Generation failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Generation request failed: {e}")

if __name__ == "__main__":
    test_ollama_connectivity()
