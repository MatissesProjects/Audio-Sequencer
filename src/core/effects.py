import pedalboard
from pedalboard import Reverb, Distortion, HighpassFilter, LowpassFilter, Delay, Chorus
import numpy as np

class AudioEffect:
    """Base class for all modular audio effects."""
    def apply(self, samples, sr, params):
        raise NotImplementedError

class DelayEffect(AudioEffect):
    def apply(self, samples, sr, params):
        amt = params.get('delay', 0.0)
        if amt <= 0: return samples
        # Feedback and mix based on user amount
        board = pedalboard.Pedalboard([
            Delay(delay_seconds=0.375, feedback=amt * 0.6, mix=amt * 0.5)
        ])
        return board(samples, sr)

class ChorusEffect(AudioEffect):
    def apply(self, samples, sr, params):
        amt = params.get('chorus', 0.0)
        if amt <= 0: return samples
        board = pedalboard.Pedalboard([
            Chorus(rate_hz=1.5, depth=amt * 0.5, mix=amt * 0.5)
        ])
        return board(samples, sr)

class ReverbEffect(AudioEffect):
    def apply(self, samples, sr, params):
        rev_amt = params.get('reverb', 0.0)
        if rev_amt <= 0: return samples
        board = pedalboard.Pedalboard([
            Reverb(room_size=0.8, wet_level=rev_amt * 0.6, dry_level=1.0 - (rev_amt * 0.2))
        ])
        return board(samples, sr)

class DistortionEffect(AudioEffect):
    """Used for harmonics and saturation."""
    def apply(self, samples, sr, params):
        harm_amt = params.get('harmonics', 0.0)
        if harm_amt <= 0: return samples
        # Map user preference to Distortion drive
        board = pedalboard.Pedalboard([
            Distortion(drive_db=harm_amt * 15)
        ])
        return board(samples, sr)

class FilterEffect(AudioEffect):
    def apply(self, samples, sr, params):
        lc = params.get('low_cut', 20)
        hc = params.get('high_cut', 20000)
        if lc <= 20 and hc >= 20000: return samples
        board = pedalboard.Pedalboard([
            HighpassFilter(cutoff_frequency_hz=lc),
            LowpassFilter(cutoff_frequency_hz=hc)
        ])
        return board(samples, sr)

class FXChain:
    """Manages a sequence of effects."""
    def __init__(self):
        self.effects = [
            DistortionEffect(), # Saturation first
            FilterEffect(),     # EQ
            ChorusEffect(),     # Thickening
            DelayEffect(),      # Echo
            ReverbEffect()      # Reverb last
        ]

    def process(self, samples, sr, params):
        for effect in self.effects:
            samples = effect.apply(samples, sr, params)
        return samples
