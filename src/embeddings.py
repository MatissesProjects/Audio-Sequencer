import laion_clap
import torch
import numpy as np
import librosa
import os

class EmbeddingEngine:
    """Handles generation of semantic audio embeddings using CLAP."""
    
    def __init__(self, model_type="640", use_cuda=False):
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        
        # This will download the weights on first run
        # Using the standard 640 pretrained model
        print(f"Loading CLAP model onto {self.device}...")
        # Note: In a real environment, we'd specify a local path to weights
        # for now, we let it use the default download/cache mechanism.
        self.model.load_ckpt() 
        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, audio_path):
        """Generates a 512-d embedding for the given audio file."""
        # CLAP expects 48kHz mono
        audio_data, _ = librosa.load(audio_path, sr=48000, mono=True)
        
        # Reshape for the model (batch, samples)
        audio_data = audio_data.reshape(1, -1)
        
        with torch.no_grad():
            audio_embed = self.model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
            
        return audio_embed[0] # Return the first (only) embedding in batch

if __name__ == "__main__":
    # Test with one of the examples
    import sys
    if len(sys.argv) > 1:
        engine = EmbeddingEngine()
        emb = engine.get_embedding(sys.argv[1])
        print(f"Embedding shape: {emb.shape}")
        print(f"First 5 values: {emb[:5]}")
