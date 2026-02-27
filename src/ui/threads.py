from PyQt6.QtCore import QThread, pyqtSignal
import os
import time
from typing import List, Dict, Optional, Any, Union, Tuple
from src.database import DataManager
from src.core.models import TrackSegment
from src.processor import AudioProcessor

class SearchThread(QThread):
    resultsFound = pyqtSignal(list)
    errorOccurred = pyqtSignal(str)
    
    def __init__(self, query: str, dm: DataManager) -> None:
        super().__init__()
        self.query: str = query
        self.dm: DataManager = dm
        
    def run(self) -> None:
        try:
            from src.embeddings import EmbeddingEngine
            engine = EmbeddingEngine()
            text_emb = engine.get_text_embedding(self.query)
            results = self.dm.search_embeddings(text_emb, n_results=20)
            self.resultsFound.emit(results)
        except Exception as e:
            self.errorOccurred.emit(str(e))

class AIInitializerThread(QThread):
    """Background thread to warm up heavy AI models without blocking UI."""
    finished = pyqtSignal(object, object, object) # scorer, generator, orchestrator
    error = pyqtSignal(str)
    
    def run(self) -> None:
        try:
            from src.core.config import AppConfig
            import requests
            
            print(f"[BOOT] AI Warm-up started in background... (Remote AI: {AppConfig.REMOTE_AI_HOST})")
            start = time.time()
            
            try:
                requests.get(f"http://{AppConfig.REMOTE_AI_HOST}:{AppConfig.REMOTE_AI_PORT}/", timeout=2)
            except:
                print(f"[BOOT] Warning: Remote AI Server ({AppConfig.REMOTE_AI_HOST}) seems offline.")

            from src.scoring import CompatibilityScorer
            from src.generator import TransitionGenerator
            from src.orchestrator import FullMixOrchestrator
            from src.embeddings import EmbeddingEngine
            
            s = CompatibilityScorer()
            g = TransitionGenerator()
            o = FullMixOrchestrator()
            
            print("[BOOT] Pre-loading CLAP model...")
            _ = EmbeddingEngine()
            
            elapsed = time.time() - start
            print(f"[BOOT] AI Engine Ready ({elapsed:.2f}s)")
            self.finished.emit(s, g, o)
        except Exception as e:
            self.error.emit(str(e))

class WaveformLoader(QThread):
    waveformLoaded = pyqtSignal(object, list, dict) # segment, full_waveform, stem_waveforms
    
    def __init__(self, segment: TrackSegment, processor: AudioProcessor) -> None:
        super().__init__()
        self.segment: TrackSegment = segment
        self.processor: AudioProcessor = processor
        
    def run(self) -> None:
        try:
            w = self.processor.get_waveform_envelope(self.segment.file_path)
            sw: Dict[str, List[float]] = {}
            if self.segment.stems_path and os.path.exists(self.segment.stems_path):
                for s in ["vocals", "drums", "bass", "other"]:
                    sp = os.path.join(self.segment.stems_path, f"{s}.wav")
                    if os.path.exists(sp):
                        sw[s] = self.processor.get_waveform_envelope(sp)
            
            self.waveformLoaded.emit(self.segment, w, sw)
        except: pass

class IngestionThread(QThread):
    finished = pyqtSignal()
    
    def __init__(self, paths: List[str], dm: DataManager) -> None:
        super().__init__()
        self.paths: List[str] = paths
        self.dm: DataManager = dm
        
    def run(self) -> None:
        try:
            from src.ingestion import IngestionEngine
            ie = IngestionEngine(db_path=self.dm.db_path)
            for p in self.paths:
                if os.path.isdir(p):
                    ie.scan_directory(p)
                else:
                    ie.ingest_single_file(p)
            self.finished.emit()
        except: pass

class StemSeparationThread(QThread):
    finished = pyqtSignal(str) # stems_dir
    error = pyqtSignal(str)
    
    def __init__(self, segment: TrackSegment, processor: AudioProcessor) -> None:
        super().__init__()
        self.segment: TrackSegment = segment
        self.processor: AudioProcessor = processor
        
    def run(self) -> None:
        try:
            from src.core.config import AppConfig
            stems_dir = AppConfig.get_stems_path(self.segment.filename)
            self.processor.separate_stems(self.segment.file_path, stems_dir)
            self.finished.emit(stems_dir)
        except Exception as e:
            self.error.emit(str(e))
