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
            
        combined.export(output_path, format="mp3", bitrate="320k")
        return output_path

    def dj_stitch(self, track_paths, output_path, overlay_ms=8000):
        """
        Creates a 'DJ Mix' style sequence with long overlapping sections.
        """
        if not track_paths:
            return None
            
        combined = AudioSegment.from_file(track_paths[0])
        combined = combined.set_frame_rate(self.sr).set_channels(2)
        
        for next_track_path in track_paths[1:]:
            next_seg = AudioSegment.from_file(next_track_path)
            next_seg = next_seg.set_frame_rate(self.sr).set_channels(2)
            
            # Fade out previous, fade in next
            # We overlay the start of next_seg onto the end of combined
            fade_out_seg = combined[-overlay_ms:].fade_out(overlay_ms)
            main_body = combined[:-overlay_ms]
            
            fade_in_next = next_seg[:overlay_ms].fade_in(overlay_ms)
            rest_of_next = next_seg[overlay_ms:]
            
            # Layer the overlap
            transition = fade_out_seg.overlay(fade_in_next)
            
            combined = main_body + transition + rest_of_next
            
        combined.export(output_path, format="mp3", bitrate="320k")
        return output_path

if __name__ == "__main__":
    # Test mixing the original and the stretched version
    renderer = FlowRenderer()
    # We will use this in preview_mix.py
