from pydub import AudioSegment
import os

class FlowRenderer:
    """Handles mixing, layering, and crossfading multiple tracks."""
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def stitch_tracks(self, track_paths, output_path, crossfade_ms=2000):
        """
        Sequences multiple tracks horizontally with crossfades.
        """
        if not track_paths:
            return None
            
        combined = AudioSegment.from_file(track_paths[0])
        combined = combined.set_frame_rate(self.sr).set_channels(2)
        
        for next_track_path in track_paths[1:]:
            next_seg = AudioSegment.from_file(next_track_path)
            next_seg = next_seg.set_frame_rate(self.sr).set_channels(2)
            
            # Append with crossfade
            combined = combined.append(next_seg, crossfade=crossfade_ms)
            
        combined.export(output_path, format="wav")
        return output_path

if __name__ == "__main__":
    # Test mixing the original and the stretched version
    renderer = FlowRenderer()
    # We will use this in preview_mix.py
