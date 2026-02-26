import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestrator import FullMixOrchestrator
from src.scoring import CompatibilityScorer

class TestRobustnessPro(unittest.TestCase):
    def test_null_value_handling(self):
        """Verify that the orchestrator can handle database tracks with NULL/None values."""
        orch = FullMixOrchestrator()
        orch.dm = MagicMock()
        
        # Track with explicitly None/NULL values (simulating a messy DB)
        bad_track = {
            'id': 1,
            'filename': "none_track.wav",
            'file_path': "none.wav",
            'bpm': None,
            'harmonic_key': None,
            'energy': None,
            'vocal_energy': None,
            'onset_density': None,
            'stems_path': None
        }
        
        dummy_tracks = [bad_track] * 10
        
        with patch.object(orch.dm, 'get_conn') as mock_conn:
            mock_cursor = mock_conn.return_value.cursor.return_value
            mock_cursor.fetchall.return_value = dummy_tracks
            
            # This should NOT crash with TypeError
            try:
                segments = orch.get_hyper_segments(depth=0)
                print("✅ Robustness Test: Successfully handled NULL database values without crash.")
            except TypeError as e:
                self.fail(f"Robustness Test FAILED: Orchestrator crashed on NULL values: {e}")

    def test_scorer_null_resilience(self):
        """Verify the scorer doesn't crash when comparing tracks with missing data."""
        scorer = CompatibilityScorer()
        t1 = {'bpm': 120, 'energy': 0.5}
        t2 = {'bpm': None, 'energy': None} # Messy track
        
        try:
            score = scorer.get_total_score(t1, t2)
            self.assertIn('total', score)
            print("✅ Scorer Resilience Test: Handled None values in scoring calculation.")
        except Exception as e:
            self.fail(f"Scorer crashed on None values: {e}")

if __name__ == "__main__":
    unittest.main()
