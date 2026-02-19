import numpy as np

class CompatibilityScorer:
    """Calculates weighted similarity scores between tracks."""
    
    # Camelot-like harmonic relations (Simplified)
    CIRCLE_OF_FIFTHS = {
        'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6,
        'C#': 7, 'G#': 8, 'D#': 9, 'A#': 10, 'F': 11
    }

    def __init__(self, bpm_weight=0.3, harmonic_weight=0.3, semantic_weight=0.4):
        self.bpm_weight = bpm_weight
        self.harmonic_weight = harmonic_weight
        self.semantic_weight = semantic_weight

    def calculate_bpm_score(self, bpm1, bpm2):
        diff_percent = (abs(bpm1 - bpm2) / bpm1) * 100
        score = max(0, 100 - (diff_percent * 6.66))
        return score

    def calculate_harmonic_score(self, key1, key2):
        if key1 not in self.CIRCLE_OF_FIFTHS or key2 not in self.CIRCLE_OF_FIFTHS:
            return 50
        pos1 = self.CIRCLE_OF_FIFTHS[key1]
        pos2 = self.CIRCLE_OF_FIFTHS[key2]
        distance = abs(pos1 - pos2)
        if distance > 6: distance = 12 - distance
        if distance == 0: return 100
        if distance == 1: return 80
        return max(0, 60 - (distance * 10))

    def calculate_semantic_score(self, emb1, emb2):
        """Calculates cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 50
        
        # Cosine Similarity
        dot_product = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)
        similarity = dot_product / (norm_a * norm_b)
        
        # Scale 0.0-1.0 to 0-100
        return max(0, min(100, (similarity + 1) / 2 * 100))

    def get_total_score(self, track1, track2, emb1=None, emb2=None):
        """Combines all scores into a single 0-100 value."""
        bpm_s = self.calculate_bpm_score(track1['bpm'], track2['bpm'])
        har_s = self.calculate_harmonic_score(track1['harmonic_key'], track2['harmonic_key'])
        sem_s = self.calculate_semantic_score(emb1, emb2)
        
        total = (bpm_s * self.bpm_weight) + (har_s * self.harmonic_weight) + (sem_s * self.semantic_weight)
        return {
            "total": round(total, 2),
            "bpm_score": round(bpm_s, 2),
            "harmonic_score": round(har_s, 2),
            "semantic_score": round(sem_s, 2)
        }
