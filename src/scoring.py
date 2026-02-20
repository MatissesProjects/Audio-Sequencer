import numpy as np

class CompatibilityScorer:
    """Calculates weighted similarity scores between tracks."""
    
    # Camelot-like harmonic relations (Simplified)
    CIRCLE_OF_FIFTHS = {
        'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6,
        'C#': 7, 'G#': 8, 'D#': 9, 'A#': 10, 'F': 11
    }

    def __init__(self, bpm_weight=0.25, harmonic_weight=0.25, semantic_weight=0.3, groove_weight=0.1, energy_weight=0.1):
        self.bpm_weight = bpm_weight
        self.harmonic_weight = harmonic_weight
        self.semantic_weight = semantic_weight
        self.groove_weight = groove_weight
        self.energy_weight = energy_weight

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

    def calculate_groove_score(self, d1, d2):
        """Compares rhythmic density (onsets per second)."""
        # Similarity score: 100 if identical, drops off as ratio diverges
        if d1 == 0 or d2 == 0: return 50
        ratio = min(d1, d2) / max(d1, d2)
        return ratio * 100

    def calculate_energy_score(self, e1, e2):
        """Compares RMS energy levels."""
        # Simple similarity
        diff = abs(e1 - e2)
        # Assuming energy is roughly 0.0 to 0.5 usually
        score = max(0, 100 - (diff * 200)) 
        return score

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
        
        # Safe access to new fields with defaults
        grv_s = self.calculate_groove_score(track1.get('onset_density', 0), track2.get('onset_density', 0))
        nrg_s = self.calculate_energy_score(track1.get('energy', 0), track2.get('energy', 0))
        
        total = (bpm_s * self.bpm_weight) + \
                (har_s * self.harmonic_weight) + \
                (sem_s * self.semantic_weight) + \
                (grv_s * self.groove_weight) + \
                (nrg_s * self.energy_weight)
                
        return {
            "total": round(total, 2),
            "bpm_score": round(bpm_s, 2),
            "harmonic_score": round(har_s, 2),
            "semantic_score": round(sem_s, 2),
            "groove_score": round(grv_s, 2),
            "energy_score": round(nrg_s, 2)
        }

    def calculate_bridge_score(self, prev_track, next_track, candidate, p_emb=None, n_emb=None, c_emb=None):
        """Evaluates how well a candidate track acts as a bridge between two others."""
        # Score A -> Candidate
        score_in = self.get_total_score(prev_track, candidate, p_emb, c_emb)
        # Score Candidate -> B
        score_out = self.get_total_score(candidate, next_track, c_emb, n_emb)
        
        # We want a candidate that is compatible with BOTH
        # Harmonic continuity is especially important for bridges
        avg_total = (score_in['total'] + score_out['total']) / 2
        harmonic_bonus = (score_in['harmonic_score'] + score_out['harmonic_score']) / 4 # up to 50pts bonus
        
        return round(min(100, avg_total + harmonic_bonus), 2)
