from PyQt6.QtGui import QColor
import json
from typing import Dict, List, Optional, Any, Tuple, Union, TypedDict

# Strong Typing for Audio Metadata from Database
class TrackMetadata(TypedDict, total=False):
    id: int
    file_path: str
    filename: str
    duration: float
    sample_rate: int
    bpm: float
    harmonic_key: str
    key: str
    energy: float
    vocal_energy: float
    onset_density: float
    stems_path: Optional[str]
    vocal_lyrics: Optional[str]
    vocal_gender: Optional[str]
    onsets_json: str
    sections_json: str
    clp_embedding_id: Optional[str]

class MusicalSection(TypedDict):
    start: float
    end: float
    label: str
    energy: float

class TrackSegment:
    KEY_COLORS: Dict[str, QColor] = {
        'C': QColor(255, 50, 50), 'C#': QColor(255, 100, 200),
        'D': QColor(255, 150, 50), 'D#': QColor(255, 50, 255),
        'E': QColor(255, 255, 50), 'F': QColor(255, 50, 150),
        'F#': QColor(50, 255, 255), 'G': QColor(50, 255, 50),
        'G#': QColor(150, 50, 255), 'A': QColor(50, 150, 255),
        'A#': QColor(200, 100, 255), 'B': QColor(100, 255, 100)
    }

    def __init__(self, track_data: Union[TrackMetadata, Dict[str, Any]], start_ms: int = 0, duration_ms: int = 20000, lane: int = 0, offset_ms: float = 0):
        self.id: int = track_data.get('id', -1)
        self.filename: str = track_data.get('filename', "Unknown")
        self.file_path: str = track_data.get('file_path', "")
        self.bpm: float = float(track_data.get('bpm', 120.0))
        self.key: str = str(track_data.get('harmonic_key') or track_data.get('key', 'N/A'))
        self.start_ms: int = start_ms
        self.duration_ms: int = duration_ms
        self.offset_ms: float = offset_ms
        self.volume: float = 1.0
        self.pan: float = 0.0 # -1.0 (Left) to 1.0 (Right)
        self.low_cut: float = 20.0 # Hz
        self.high_cut: float = 20000.0 # Hz
        self.is_ambient: bool = False 
        self.lane: int = lane
        self.is_primary: bool = False
        self.waveform: List[float] = []
        self.stem_waveforms: Dict[str, List[float]] = {} 
        self.fade_in_ms: int = 2000
        self.fade_out_ms: int = 2000
        self.pitch_shift: int = 0
        self.reverb: float = 0.0 # 0.0 to 1.0
        self.harmonics: float = 0.0 # 0.0 to 1.0
        self.delay: float = 0.0 
        self.chorus: float = 0.0 
        self.stems_path: Optional[str] = track_data.get('stems_path')
        self.vocal_energy: float = float(track_data.get('vocal_energy') or 0.0)
        self.vocal_lyrics: Optional[str] = track_data.get('vocal_lyrics')
        self.vocal_gender: Optional[str] = track_data.get('vocal_gender')
        self.sections: List[MusicalSection] = []
        if 'sections_json' in track_data and track_data['sections_json']:
            try:
                self.sections = json.loads(track_data['sections_json'])
            except:
                pass
        self.vocal_shift: int = 0
        self.bass_shift: int = 0
        self.drum_shift: int = 0
        self.instr_shift: int = 0
        self.gender_swap: str = "none" # "none", "male", "female"
        self.harmony_level: float = 0.0
        self.vocal_vol: float = 1.0
        self.drum_vol: float = 1.0
        self.bass_vol: float = 1.0
        self.instr_vol: float = 1.0
        self.ducking_depth: float = 0.7 
        self.duck_low: float = 1.0 
        self.duck_mid: float = 1.0
        self.duck_high: float = 1.0
        
        # KEYFRAMES: { 'volume': [(0, 1.0), (5000, 0.5)] }
        self.keyframes: Dict[str, List[Tuple[float, float]]] = {} 
        
        base_color = self.KEY_COLORS.get(self.key, QColor(70, 130, 180))
        self.color: QColor = QColor(base_color.red(), base_color.green(), base_color.blue(), 200)
        self.onsets: List[float] = []
        if 'onsets_json' in track_data and track_data['onsets_json']:
            try:
                self.onsets = [float(x) * 1000.0 for x in track_data['onsets_json'].split(',')]
            except:
                pass

    def add_keyframe(self, param: str, relative_ms: float, value: float) -> None:
        """Adds a keyframe for a parameter. Overwrites if exists at same time."""
        if param not in self.keyframes:
            self.keyframes[param] = []
        
        # Remove existing at same time
        self.keyframes[param] = [k for k in self.keyframes[param] if abs(k[0] - relative_ms) > 10]
        self.keyframes[param].append((relative_ms, value))
        self.keyframes[param].sort(key=lambda x: x[0])

    def get_value_at(self, param: str, relative_ms: float, default_val: float) -> float:
        """Linearly interpolates value for a parameter at a given time."""
        if param not in self.keyframes or not self.keyframes[param]:
            return default_val
            
        points = self.keyframes[param]
        
        # Before first point
        if relative_ms <= points[0][0]:
            return points[0][1]
        # After last point
        if relative_ms >= points[-1][0]:
            return points[-1][1]
            
        # Interpolate
        for i in range(len(points) - 1):
            t1, v1 = points[i]
            t2, v2 = points[i+1]
            if t1 <= relative_ms <= t2:
                ratio = (relative_ms - t1) / (t2 - t1)
                return v1 + (v2 - v1) * ratio
        return default_val

    def get_end_ms(self) -> float:
        """Returns the absolute end time of the segment on the timeline."""
        return self.start_ms + self.duration_ms

    def overlaps_with(self, other: 'TrackSegment') -> bool:
        """Checks if this segment overlaps with another segment in time."""
        return max(self.start_ms, other.start_ms) < min(self.get_end_ms(), other.get_end_ms())

    def to_dict(self) -> Dict[str, Any]:
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
            'reverb': self.reverb, 'harmonics': self.harmonics,
            'delay': self.delay, 'chorus': self.chorus,
            'stems_path': self.stems_path,
            'vocal_energy': self.vocal_energy,
            'vocal_lyrics': self.vocal_lyrics,
            'vocal_gender': self.vocal_gender,
            'gender_swap': self.gender_swap,
            'sections_json': json.dumps(self.sections),
            'vocal_shift': self.vocal_shift,
            'bass_shift': self.bass_shift,
            'drum_shift': self.drum_shift,
            'instr_shift': self.instr_shift,
            'harmony_level': self.harmony_level,
            'vocal_vol': self.vocal_vol,
            'drum_vol': self.drum_vol,
            'bass_vol': self.bass_vol,
            'instr_vol': self.instr_vol,
            'ducking_depth': self.ducking_depth,
            'duck_low': self.duck_low,
            'duck_mid': self.duck_mid,
            'duck_high': self.duck_high,
            'keyframes': self.keyframes
        }
        d['onsets_json'] = ",".join([str(x/1000.0) for x in self.onsets])
        return d
