class CompatibilityScorer:
    """Calculates weighted similarity scores between tracks."""
    
    # Camelot-like harmonic relations (Simplified)
    # Mapping keys to positions on circle of fifths for distance calculation
    CIRCLE_OF_FIFTHS = {
        'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6,
        'C#': 7, 'G#': 8, 'D#': 9, 'A#': 10, 'F': 11
    }

    def __init__(self, bpm_weight=0.5, harmonic_weight=0.5):
        self.bpm_weight = bpm_weight
        self.harmonic_weight = harmonic_weight

    def calculate_bpm_score(self, bpm1, bpm2):
        """Higher score for closer BPMs. 100 is perfect."""
        diff_percent = (abs(bpm1 - bpm2) / bpm1) * 100
        # Score drops to 0 at 15% difference
        score = max(0, 100 - (diff_percent * 6.66))
        return score

    def calculate_harmonic_score(self, key1, key2):
        """Scores based on distance on the Circle of Fifths."""
        if key1 not in self.CIRCLE_OF_FIFTHS or key2 not in self.CIRCLE_OF_FIFTHS:
            return 50 # Default middle score for unknown keys
            
        pos1 = self.CIRCLE_OF_FIFTHS[key1]
        pos2 = self.CIRCLE_OF_FIFTHS[key2]
        
        distance = abs(pos1 - pos2)
        if distance > 6:
            distance = 12 - distance
            
        # Perfect match: 100, Adjacent: 80, others lower
        if distance == 0: return 100
        if distance == 1: return 80
        return max(0, 60 - (distance * 10))

    def get_total_score(self, track1, track2):
        """Combines all scores into a single 0-100 value."""
        bpm_s = self.calculate_bpm_score(track1['bpm'], track2['bpm'])
        har_s = self.calculate_harmonic_score(track1['harmonic_key'], track2['harmonic_key'])
        
        total = (bpm_s * self.bpm_weight) + (har_s * self.harmonic_weight)
        return {
            "total": round(total, 2),
            "bpm_score": round(bpm_s, 2),
            "harmonic_score": round(har_s, 2)
        }
