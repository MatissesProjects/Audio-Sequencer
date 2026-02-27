import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple

class CompatibilityScorer:
    """Calculates weighted similarity scores between tracks."""
    
    # Camelot-like harmonic relations (Simplified)
    CIRCLE_OF_FIFTHS: Dict[str, int] = {
        'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6,
        'C#': 7, 'G#': 8, 'D#': 9, 'A#': 10, 'F': 11
    }

    def __init__(self, bpm_weight: float = 0.25, harmonic_weight: float = 0.25, semantic_weight: float = 0.3, groove_weight: float = 0.1, energy_weight: float = 0.1):
        self.bpm_weight: float = bpm_weight
        self.harmonic_weight: float = harmonic_weight
        self.semantic_weight: float = semantic_weight
        self.groove_weight: float = groove_weight
        self.energy_weight: float = energy_weight

    def calculate_bpm_score(self, bpm1: float, bpm2: float) -> float:
        if bpm1 <= 0: return 0.0
        diff_percent = (abs(bpm1 - bpm2) / bpm1) * 100
        return max(0.0, 100.0 - (diff_percent * 6.66))

    def calculate_harmonic_score(self, key1: str, key2: str) -> float:
        if key1 not in self.CIRCLE_OF_FIFTHS or key2 not in self.CIRCLE_OF_FIFTHS:
            return 50.0
        pos1 = self.CIRCLE_OF_FIFTHS[key1]; pos2 = self.CIRCLE_OF_FIFTHS[key2]
        distance = abs(pos1 - pos2)
        if distance > 6: distance = 12 - distance
        if distance == 0: return 100.0
        if distance == 1: return 80.0
        return max(0.0, 60.0 - (distance * 10.0))

    def calculate_groove_score(self, d1: float, d2: float) -> float:
        """Compares rhythmic density (onsets per second)."""
        if d1 <= 0 or d2 <= 0: return 50.0
        return (min(d1, d2) / max(d1, d2)) * 100.0

    def calculate_energy_score(self, e1: float, e2: float) -> float:
        """Compares RMS energy levels."""
        diff = abs(e1 - e2)
        return max(0.0, 100.0 - (diff * 200.0)) 

    def calculate_semantic_score(self, emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> float:
        """Calculates cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None: return 50.0
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return max(0.0, min(100.0, (similarity + 1) / 2 * 100.0))

    def get_total_score(self, track1: Dict[str, Any], track2: Dict[str, Any], emb1: Optional[np.ndarray] = None, emb2: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Combines all scores into a single 0-100 value."""
        bpm1 = float(track1.get('bpm') or 120.0); bpm2 = float(track2.get('bpm') or 120.0)
        key1 = str(track1.get('harmonic_key') or track1.get('key') or 'N/A')
        key2 = str(track2.get('harmonic_key') or track2.get('key') or 'N/A')
        bpm_s = self.calculate_bpm_score(bpm1, bpm2); har_s = self.calculate_harmonic_score(key1, key2); sem_s = self.calculate_semantic_score(emb1, emb2)
        grv_s = self.calculate_groove_score(float(track1.get('onset_density') or 0), float(track2.get('onset_density') or 0))
        nrg_s = self.calculate_energy_score(float(track1.get('energy') or 0), float(track2.get('energy') or 0))
        total = (bpm_s * self.bpm_weight) + (har_s * self.harmonic_weight) + (sem_s * self.semantic_weight) + (grv_s * self.groove_weight) + (nrg_s * self.energy_weight)
        return {
            "total": round(total, 2), "bpm_score": round(bpm_s, 2), "harmonic_score": round(har_s, 2),
            "semantic_score": round(sem_s, 2), "groove_score": round(grv_s, 2), "energy_score": round(nrg_s, 2)
        }

    def calculate_bridge_score(self, prev_track: Dict[str, Any], next_track: Dict[str, Any], candidate: Dict[str, Any], p_emb: Optional[np.ndarray] = None, n_emb: Optional[np.ndarray] = None, c_emb: Optional[np.ndarray] = None) -> float:
        """Evaluates how well a candidate track acts as a bridge between two others."""
        s_in = self.get_total_score(prev_track, candidate, p_emb, c_emb)
        s_out = self.get_total_score(candidate, next_track, c_emb, n_emb)
        avg_total = (s_in['total'] + s_out['total']) / 2
        harmonic_bonus = (s_in['harmonic_score'] + s_out['harmonic_score']) / 4 
        return round(min(100.0, avg_total + harmonic_bonus), 2)
