import numpy as np
import librosa
import os

class EmbeddingEngine:
    """Handles generation of semantic audio embeddings using CLAP."""
    
    _model_cache = None
    _torch_cache = None
    _device_cache = None

    def __init__(self, model_type="640", use_cuda=False):
        if EmbeddingEngine._model_cache is None:
            import torch
            import laion_clap
            EmbeddingEngine._torch_cache = torch
            EmbeddingEngine._device_cache = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
            EmbeddingEngine._model_cache = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
            
            print(f"Loading CLAP model onto {EmbeddingEngine._device_cache}...")
            EmbeddingEngine._model_cache.load_ckpt() 
            EmbeddingEngine._model_cache.to(EmbeddingEngine._device_cache)
            EmbeddingEngine._model_cache.eval()
            
        self.torch = EmbeddingEngine._torch_cache
        self.device = EmbeddingEngine._device_cache
        self.model = EmbeddingEngine._model_cache

    def get_embedding(self, audio_path):
        """Generates a 512-d embedding for the given audio file."""
        audio_data, _ = librosa.load(audio_path, sr=48000, mono=True)
        audio_data = audio_data.reshape(1, -1)
        
        with self.torch.no_grad():
            audio_embed = self.model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
        return audio_embed[0]

    def get_text_embedding(self, text):
        """Generates a 512-d embedding for the given text description."""
        with self.torch.no_grad():
            text_embed = self.model.get_text_embedding([text])
        return text_embed[0]

if __name__ == "__main__":
    # Test with one of the examples
    import sys
    if len(sys.argv) > 1:
        engine = EmbeddingEngine()
        emb = engine.get_embedding(sys.argv[1])
        print(f"Embedding shape: {emb.shape}")
        print(f"First 5 values: {emb[:5]}")
