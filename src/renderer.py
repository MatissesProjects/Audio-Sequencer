from pydub import AudioSegment
import os

class FlowRenderer:
    """Handles mixing, layering, and crossfading multiple tracks."""
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def mix_tracks(self, track1_path, track2_path, output_path, gain2=-3.0, position_ms=0):
        """
        Layers track2 on top of track1 at a specific millisecond offset.
        """
        s1 = AudioSegment.from_file(track1_path)
        s2 = AudioSegment.from_file(track2_path)
        
        s1 = s1.set_frame_rate(self.sr).set_channels(2)
        s2 = s2.set_frame_rate(self.sr).set_channels(2)

        s1 = s1 - 3.0
        s2 = s2 + gain2

        # Overlay at specific position
        mixed = s1.overlay(s2, position=position_ms)
        
        mixed.export(output_path, format="wav")
        return output_path

if __name__ == "__main__":
    # Test mixing the original and the stretched version
    renderer = FlowRenderer()
    # We will use this in preview_mix.py
