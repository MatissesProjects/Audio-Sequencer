from PyQt6.QtGui import QColor

class TrackSegment:
    KEY_COLORS = {
        'C': QColor(255, 50, 50), 'C#': QColor(255, 100, 200),
        'D': QColor(255, 150, 50), 'D#': QColor(255, 50, 255),
        'E': QColor(255, 255, 50), 'F': QColor(255, 50, 150),
        'F#': QColor(50, 255, 255), 'G': QColor(50, 255, 50),
        'G#': QColor(150, 50, 255), 'A': QColor(50, 150, 255),
        'A#': QColor(200, 100, 255), 'B': QColor(100, 255, 100)
    }
    def __init__(self, track_data, start_ms=0, duration_ms=20000, lane=0, offset_ms=0):
        self.id = track_data['id']
        self.filename = track_data['filename']
        self.file_path = track_data['file_path']
        self.bpm = track_data['bpm']
        self.key = track_data['harmonic_key']
        self.start_ms = start_ms
        self.duration_ms = duration_ms
        self.offset_ms = offset_ms
        self.volume = 1.0
        self.pan = 0.0 # -1.0 (Left) to 1.0 (Right)
        self.low_cut = 20 # Hz
        self.high_cut = 20000 # Hz
        self.is_ambient = False # NEW: Background/Ambient role
        self.lane = lane
        self.is_primary = False
        self.waveform = []
        self.fade_in_ms = 2000
        self.fade_out_ms = 2000
        self.pitch_shift = 0
        self.reverb = 0.0 # 0.0 to 1.0 (Wet amount)
        self.harmonics = 0.0 # 0.0 to 1.0 (Saturation/Harmonic excitement)
        base_color = self.KEY_COLORS.get(self.key, QColor(70, 130, 180))
        self.color = QColor(base_color.red(), base_color.green(), base_color.blue(), 200)
        self.onsets = []
        if 'onsets_json' in track_data and track_data['onsets_json']:
            try:
                self.onsets = [float(x) * 1000.0 for x in track_data['onsets_json'].split(',')]
            except:
                pass

    def to_dict(self):
        d = {
            'id': self.id, 'filename': self.filename, 'file_path': self.file_path, 
            'bpm': self.bpm, 'key': self.key, 'start_ms': self.start_ms, 
            'duration_ms': self.duration_ms, 'offset_ms': self.offset_ms, 
            'volume': self.volume, 'pan': self.pan, 
            'low_cut': self.low_cut, 'high_cut': self.high_cut,
            'is_ambient': self.is_ambient,
            'lane': self.lane, 'is_primary': self.is_primary, 
            'fade_in_ms': self.fade_in_ms, 'fade_out_ms': self.fade_out_ms, 
            'pitch_shift': self.pitch_shift,
            'reverb': self.reverb, 'harmonics': self.harmonics
        }
        d['onsets_json'] = ",".join([str(x/1000.0) for x in self.onsets])
        return d
