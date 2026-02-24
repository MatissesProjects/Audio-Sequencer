from PyQt6.QtCore import QThread, pyqtSignal
import os
import time

class SearchThread(QThread):
    resultsFound = pyqtSignal(list)
    errorOccurred = pyqtSignal(str)
    
    def __init__(self, query, dm):
        super().__init__()
        self.query = query
        self.dm = dm
        
    def run(self):
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
    
    def run(self):
        try:
            print("[BOOT] AI Warm-up started in background...")
            start = time.time()
            from src.scoring import CompatibilityScorer
            from src.generator import TransitionGenerator
            from src.orchestrator import FullMixOrchestrator
            from src.embeddings import EmbeddingEngine
            
            s = CompatibilityScorer()
            # TransitionGenerator now uses remote network calls
            g = TransitionGenerator()
            o = FullMixOrchestrator()
            
            # Warm up CLAP model in background
            print("[BOOT] Pre-loading CLAP model...")
            _ = EmbeddingEngine()
            
            elapsed = time.time() - start
            print(f"[BOOT] AI Engine Ready ({elapsed:.2f}s)")
            self.finished.emit(s, g, o)
        except Exception as e:
            self.error.emit(str(e))

class WaveformLoader(QThread):
    waveformLoaded = pyqtSignal(object, list)
    
    def __init__(self, segment, processor):
        super().__init__()
        self.segment = segment
        self.processor = processor
        
    def run(self):
        try:
            w = self.processor.get_waveform_envelope(self.segment.file_path)
            self.waveformLoaded.emit(self.segment, w)
        except:
            pass

class IngestionThread(QThread):
    finished = pyqtSignal()
    
    def __init__(self, paths, dm):
        super().__init__()
        self.paths = paths
        self.dm = dm
        
    def run(self):
        try:
            from src.ingestion import IngestionEngine
            ie = IngestionEngine(db_path=self.dm.db_path)
            for p in self.paths:
                if os.path.isdir(p):
                    ie.scan_directory(p)
                else:
                    ie.analyze_and_store(p)
            self.finished.emit()
        except:
            pass
