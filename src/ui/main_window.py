import sys
import os
import sqlite3
import json
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTableWidgetItem, QLineEdit, QLabel, QPushButton, 
                             QFrame, QMessageBox, QScrollArea, QFileDialog,
                             QSlider, QComboBox, QCheckBox, QStatusBar, QApplication)
from PyQt6.QtCore import Qt, QSize, QTimer, QUrl
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# Project Imports (Lightweight)
from src.database import DataManager
from src.processor import AudioProcessor
from src.renderer import FlowRenderer

# Modular Imports
from src.core.models import TrackSegment
from src.core.undo import UndoManager
from src.ui.dialogs import show_error
from src.ui.threads import SearchThread, IngestionThread, WaveformLoader
from src.ui.widgets import TimelineWidget, DraggableTable, LibraryWaveformPreview, LoadingOverlay

def sqlite3_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

class AudioSequencerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dm = DataManager()
        self.processor = AudioProcessor()
        self.renderer = FlowRenderer()
        self.undo_manager = UndoManager()
        
        # Lazy load heavy AI components
        try:
            from src.scoring import CompatibilityScorer
            from src.generator import TransitionGenerator
            from src.orchestrator import FullMixOrchestrator
            self.scorer = CompatibilityScorer()
            self.generator = TransitionGenerator()
            self.orchestrator = FullMixOrchestrator()
            self.ai_enabled = True
        except Exception as e:
            print(f"AI Loading Error (Check network/models): {e}")
            self.scorer = None
            self.generator = None
            self.orchestrator = None
            self.ai_enabled = False
        
        self.selected_library_track = None
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        self.preview_path = "temp_preview.wav"
        self.preview_dirty = True
        
        self.play_timer = QTimer()
        self.play_timer.setInterval(20)
        self.play_timer.timeout.connect(self.update_playback_cursor)
        self.is_playing = False
        
        self.waveform_loaders = []
        
        self.init_ui()
        self.load_library()
        self.loading_overlay = LoadingOverlay(self.centralWidget())
        self.setAcceptDrops(True)
        
    def init_ui(self):
        self.setWindowTitle("AudioSequencer AI - The Pro Flow")
        self.setMinimumSize(QSize(1400, 950))
        
        cw = QWidget()
        self.setCentralWidget(cw)
        ml = QVBoxLayout(cw)
        
        # Top Panel
        tp = QHBoxLayout()
        
        # 1. Library Panel
        lp = QFrame()
        lp.setFixedWidth(450)
        ll = QVBoxLayout(lp)
        ll.addWidget(QLabel("<h2>📁 Audio Library</h2>"))
        
        la = QHBoxLayout()
        self.scan_btn = QPushButton("📂 Scan Folder")
        self.scan_btn.clicked.connect(self.scan_folder)
        la.addWidget(self.scan_btn)
        
        self.embed_btn = QPushButton("🧠 AI Index")
        self.embed_btn.clicked.connect(self.run_embedding)
        la.addWidget(self.embed_btn)
        ll.addLayout(la)
        
        sl = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("🔍 Semantic Search...")
        self.search_bar.textChanged.connect(self.on_search_text_changed)
        self.search_bar.returnPressed.connect(self.trigger_semantic_search)
        sl.addWidget(self.search_bar)
        
        rsb = QPushButton("↺")
        rsb.setFixedWidth(30)
        rsb.clicked.connect(self.load_library)
        sl.addWidget(rsb)
        ll.addLayout(sl)
        
        self.library_table = DraggableTable(0, 3)
        self.library_table.setHorizontalHeaderLabels(["Track Name", "BPM", "Key"])
        self.library_table.setColumnWidth(0, 250)
        self.library_table.itemSelectionChanged.connect(self.on_library_track_selected)
        ll.addWidget(self.library_table)
        
        self.lib_preview_group = QFrame()
        self.lib_preview_group.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; border-radius: 4px; margin-top: 5px;")
        lpl = QVBoxLayout(self.lib_preview_group)
        self.l_wave = LibraryWaveformPreview()
        lpl.addWidget(self.l_wave)
        self.l_wave_label = QLabel("Select track to preview")
        self.l_wave_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.l_wave_label.setStyleSheet("color: #666; font-size: 10px;")
        lpl.addWidget(self.l_wave_label)
        ll.addWidget(self.lib_preview_group)
        
        tp.addWidget(lp)
        
        # 2. Middle/Right Panel (Controls & Recs)
        mp = QFrame()
        mp.setFixedWidth(250)
        mlayout = QVBoxLayout(mp)
        
        ag = QFrame()
        ag.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 5px;")
        al = QVBoxLayout(ag)
        al.addWidget(QLabel("<h3>📊 Analytics Board</h3>"))
        
        self.mod_toggle = QPushButton("🔍 Hide Markers")
        self.mod_toggle.setCheckable(True)
        self.mod_toggle.clicked.connect(self.toggle_analytics)
        al.addWidget(self.mod_toggle)
        
        self.grid_toggle = QPushButton("📏 Grid Snap: ON")
        self.grid_toggle.setCheckable(True)
        self.grid_toggle.setChecked(True)
        self.grid_toggle.clicked.connect(self.toggle_grid)
        al.addWidget(self.grid_toggle)
        
        ur = QHBoxLayout()
        self.ub = QPushButton("↶ Undo")
        self.ub.clicked.connect(self.undo)
        ur.addWidget(self.ub)
        self.rb = QPushButton("↷ Redo")
        self.rb.clicked.connect(self.redo)
        ur.addWidget(self.rb)
        al.addLayout(ur)
        
        self.stats_label = QLabel("Timeline empty")
        self.stats_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: bold;")
        al.addWidget(self.stats_label)
        
        save_btn = QPushButton("💾 Save Journey")
        save_btn.clicked.connect(self.save_project)
        al.addWidget(save_btn)
        load_btn = QPushButton("📂 Load Journey")
        load_btn.clicked.connect(self.load_project)
        al.addWidget(load_btn)
        
        mlayout.addWidget(ag)
        mlayout.addSpacing(10)
        
        mlayout.addWidget(QLabel("<h3>⚡ Actions</h3>"))
        self.atb = QPushButton("➕ Add to Timeline")
        self.atb.clicked.connect(self.add_selected_to_timeline)
        mlayout.addWidget(self.atb)
        
        self.pb = QPushButton("▶ Preview")
        self.pb.clicked.connect(self.play_selected)
        mlayout.addWidget(self.pb)
        mlayout.addSpacing(10)
        
        self.prop_group = QFrame()
        self.prop_group.setStyleSheet("background-color: #252525; border: 1px solid #444; border-radius: 8px; padding: 10px;")
        self.prop_layout = QVBoxLayout(self.prop_group)
        self.prop_layout.addWidget(QLabel("<b>Track Properties</b>"))
        
        v_lay = QHBoxLayout()
        v_lay.addWidget(QLabel("Vol:"))
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 150)
        self.vol_slider.valueChanged.connect(self.on_prop_changed)
        v_lay.addWidget(self.vol_slider)
        self.prop_layout.addLayout(v_lay)
        
        p_lay = QHBoxLayout()
        p_lay.addWidget(QLabel("Pitch:"))
        self.pitch_combo = QComboBox()
        for i in range(-6, 7):
            self.pitch_combo.addItem(f"{i:+} st", i)
        self.pitch_combo.currentIndexChanged.connect(self.on_prop_changed)
        p_lay.addWidget(self.pitch_combo)
        self.prop_layout.addLayout(p_lay)
        
        self.prim_check = QCheckBox("Set as Primary Track")
        self.prim_check.stateChanged.connect(self.on_prop_changed)
        self.prop_layout.addWidget(self.prim_check)
        
        self.prop_group.setVisible(False)
        mlayout.addWidget(self.prop_group)
        mlayout.addStretch()
        
        tp.addWidget(mp)
        
        # 3. Recommendations
        rp = QFrame()
        rp.setFixedWidth(450)
        rl = QVBoxLayout(rp)
        rl.addWidget(QLabel("<h3>✨ Smart Suggestions</h3>"))
        self.rec_list = DraggableTable(0, 2)
        self.rec_list.setHorizontalHeaderLabels(["Match %", "Track"])
        self.rec_list.itemDoubleClicked.connect(self.on_rec_double_clicked)
        rl.addWidget(self.rec_list)
        tp.addWidget(rp)
        
        ml.addLayout(tp, stretch=1)
        
        # Timeline Section
        th = QHBoxLayout()
        th.addWidget(QLabel("<h2>🎞 Timeline Journey</h2>"))
        
        th.addSpacing(20)
        th.addWidget(QLabel("H-Zoom:"))
        self.zs = QSlider(Qt.Orientation.Horizontal)
        self.zs.setRange(10, 200)
        self.zs.setValue(50)
        self.zs.setFixedWidth(100)
        self.zs.valueChanged.connect(self.on_zoom_changed)
        th.addWidget(self.zs)
        
        th.addSpacing(10)
        th.addWidget(QLabel("V-Zoom:"))
        self.vs = QSlider(Qt.Orientation.Horizontal)
        self.vs.setRange(40, 250)
        self.vs.setValue(120)
        self.vs.setFixedWidth(100)
        self.vs.valueChanged.connect(self.on_vzoom_changed)
        th.addWidget(self.vs)
        
        th.addStretch()
        
        self.new_btn = QPushButton("📄 New")
        self.new_btn.clicked.connect(self.new_project)
        th.addWidget(self.new_btn)
        
        self.sb = QPushButton("⏹")
        self.sb.setFixedWidth(40)
        self.sb.clicked.connect(self.jump_to_start)
        th.addWidget(self.sb)
        
        self.ptb = QPushButton("▶ Play Journey")
        self.ptb.setFixedWidth(120)
        self.ptb.clicked.connect(self.toggle_playback)
        th.addWidget(self.ptb)
        
        self.agb = QPushButton("🪄 Auto-Generate Path")
        self.agb.clicked.connect(self.auto_populate_timeline)
        th.addWidget(self.agb)
        
        self.cb = QPushButton("🗑 Clear")
        self.cb.clicked.connect(self.clear_timeline)
        th.addWidget(self.cb)
        
        th.addWidget(QLabel("Target BPM:"))
        self.tbe = QLineEdit("124")
        self.tbe.setFixedWidth(60)
        self.tbe.textChanged.connect(self.on_bpm_changed)
        th.addWidget(self.tbe)
        
        th.addSpacing(10)
        th.addWidget(QLabel("Master:"))
        self.mv_s = QSlider(Qt.Orientation.Horizontal)
        self.mv_s.setRange(0, 100)
        self.mv_s.setValue(80)
        self.mv_s.setFixedWidth(100)
        self.mv_s.valueChanged.connect(self.on_master_vol_changed)
        th.addWidget(self.mv_s)
        
        self.render_btn = QPushButton("🚀 RENDER FINAL MIX")
        self.render_btn.setStyleSheet("background-color: #007acc; padding: 12px 25px; color: white; font-weight: bold;")
        self.render_btn.clicked.connect(self.render_timeline)
        th.addWidget(self.render_btn)
        
        self.stems_btn = QPushButton("📦 EXPORT STEMS")
        self.stems_btn.setStyleSheet("background-color: #444; padding: 12px 15px; color: white; font-weight: bold;")
        self.stems_btn.clicked.connect(self.export_stems)
        th.addWidget(self.stems_btn)
        
        ml.addLayout(th)
        
        t_s = QScrollArea()
        t_s.setWidgetResizable(True)
        t_s.setStyleSheet("QScrollArea { background-color: #1a1a1a; border: 1px solid #333; } QScrollBar:horizontal { height: 12px; background: #222; } QScrollBar::handle:horizontal { background: #444; border-radius: 6px; }")
        
        self.timeline_widget = TimelineWidget()
        t_s.setWidget(self.timeline_widget)
        ml.addWidget(t_s, stretch=1)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        if self.ai_enabled:
            self.status_bar.showMessage("Ready. AI Engine Online.")
        else:
            self.status_bar.showMessage("Ready. AI Engine OFFLINE (Check model downloads).")
        
        # Connections
        self.timeline_widget.segmentSelected.connect(self.on_segment_selected)
        self.timeline_widget.timelineChanged.connect(self.update_status)
        self.timeline_widget.undoRequested.connect(self.push_undo)
        self.timeline_widget.cursorJumped.connect(self.on_cursor_jump)
        self.timeline_widget.bridgeRequested.connect(self.find_bridge_for_gap)
        self.timeline_widget.aiTransitionRequested.connect(self.generate_ai_transition)
        self.timeline_widget.duplicateRequested.connect(self.duplicate_segment)
        self.timeline_widget.captureRequested.connect(self.capture_segment_to_library)
        self.timeline_widget.zoomChanged.connect(lambda v: self.zs.setValue(v))
        
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI'; }
            QLabel { color: #ffffff; }
            QTableWidget { background-color: #1e1e1e; gridline-color: #333; color: white; border: 1px solid #333; }
            QHeaderView::section { background-color: #333; color: white; border: 1px solid #444; padding: 5px; font-weight: bold; }
            QPushButton { background-color: #333; color: #fff; padding: 8px; border-radius: 4px; border: 1px solid #444; }
            QPushButton:hover { background-color: #444; border: 1px solid #666; }
            QLineEdit { background-color: #222; color: white; border: 1px solid #444; padding: 5px; border-radius: 4px; }
            QComboBox { background-color: #333; color: white; border: 1px solid #444; padding: 5px; border-radius: 4px; }
            QCheckBox { color: white; spacing: 10px; }
            QScrollBar:vertical { width: 12px; background: #222; }
            QScrollBar::handle:vertical { background: #444; border-radius: 6px; }
        """)

    # ... Methods ...
    
    def load_waveform_async(self, seg):
        l = WaveformLoader(seg, self.processor)
        l.waveformLoaded.connect(self.on_waveform_loaded)
        self.waveform_loaders.append(l)
        l.start()
        
    def on_waveform_loaded(self, seg, w):
        seg.waveform = w
        self.timeline_widget.update()
        self.waveform_loaders = [l for l in self.waveform_loaders if l.isRunning()]
        
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            
    def dropEvent(self, e):
        urls = e.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        if paths:
            self.loading_overlay.show_loading("Ingesting Files...")
            self.it = IngestionThread(paths, self.dm)
            self.it.finished.connect(self.on_ingestion_finished)
            self.it.start()
            
    def on_ingestion_finished(self):
        self.load_library()
        self.loading_overlay.hide_loading()
        self.status_bar.showMessage("Ingestion complete.")
        
    def update_playback_cursor(self):
        if self.is_playing:
            p = self.player.position()
            if self.timeline_widget.loop_enabled and p >= self.timeline_widget.loop_end_ms:
                self.player.setPosition(int(self.timeline_widget.loop_start_ms))
                p = self.timeline_widget.loop_start_ms
                
            self.timeline_widget.cursor_pos_ms = p
            self.timeline_widget.update()
            
            t_e = 0
            for s in self.timeline_widget.segments:
                if s.start_ms <= p <= s.start_ms + s.duration_ms:
                    any_s = any(self.timeline_widget.solos)
                    is_a = (self.timeline_widget.solos[s.lane] if any_s else not self.timeline_widget.mutes[s.lane])
                    if is_a:
                        t_e += s.volume
                        
            mw = int(min(1.0, t_e / 3.0) * 20)
            ms = "█" * mw + "░" * (20 - mw)
            self.status_bar.showMessage(f"Playing | Energy: [{ms}] | {p/1000:.1f}s")
            
            if not self.timeline_widget.loop_enabled and p >= self.player.duration() and self.player.duration() > 0:
                self.stop_playback()

    def stop_playback(self):
        self.player.stop()
        self.play_timer.stop()
        self.is_playing = False
        self.ptb.setText("▶ Play Journey")
        self.status_bar.showMessage("Stopped.")
        
    def toggle_playback(self):
        if not self.timeline_widget.segments:
            return
            
        if self.is_playing:
            self.player.pause()
            self.play_timer.stop()
            self.is_playing = False
            self.ptb.setText("▶ Play Journey")
        else:
            if self.preview_dirty:
                self.render_preview_for_playback()
            self.player.setPosition(int(self.timeline_widget.cursor_pos_ms))
            self.player.play()
            self.play_timer.start()
            self.is_playing = True
            self.ptb.setText("⏸ Pause Preview")
            
    def render_preview_for_playback(self):
        self.loading_overlay.show_loading("Building Preview...")
        try:
            ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
            tb = float(self.tbe.text()) if self.tbe.text() else 124.0
            rd = [s.to_dict() for s in ss]
            self.renderer.render_timeline(rd, self.preview_path, target_bpm=tb, mutes=self.timeline_widget.mutes, solos=self.timeline_widget.solos)
            self.player.setSource(QUrl.fromLocalFile(os.path.abspath(self.preview_path)))
            self.preview_dirty = False
        except Exception as e:
            show_error(self, "Preview Error", "Failed to build audio.", e)
        finally:
            self.loading_overlay.hide_loading()
            
    def jump_to_start(self):
        self.timeline_widget.cursor_pos_ms = 0
        if self.is_playing:
            self.player.setPosition(0)
        self.timeline_widget.update()
        
    def push_undo(self):
        self.preview_dirty = True
        self.undo_manager.push_state(self.timeline_widget.segments)
        
    def undo(self):
        ns = self.undo_manager.undo(self.timeline_widget.segments)
        if ns:
            self.apply_state(ns)
            
    def redo(self):
        ns = self.undo_manager.redo(self.timeline_widget.segments)
        if ns:
            self.apply_state(ns)
            
    def apply_state(self, sl):
        self.timeline_widget.segments = []
        for sj in sl:
            s = json.loads(sj)
            td = {'id': s['id'], 'filename': s['filename'], 'file_path': s['file_path'], 'bpm': s['bpm'], 'harmonic_key': s['key'], 'onsets_json': s.get('onsets_json', "")}
            seg = TrackSegment(td, start_ms=s['start_ms'], duration_ms=s['duration_ms'], lane=s['lane'], offset_ms=s['offset_ms'])
            seg.volume = s['volume']
            seg.is_primary = s['is_primary']
            seg.fade_in_ms = s['fade_in_ms']
            seg.fade_out_ms = s['fade_out_ms']
            seg.pitch_shift = s.get('pitch_shift', 0)
            self.load_waveform_async(seg)
            self.timeline_widget.segments.append(seg)
        self.timeline_widget.update_geometry()
        self.update_status()

    def on_segment_selected(self, s):
        if s:
            self.status_bar.showMessage(f"Selected: {s.filename}")
            self.prop_group.setVisible(True)
            self.vol_slider.blockSignals(True)
            self.vol_slider.setValue(int(s.volume * 100))
            self.vol_slider.blockSignals(False)
            
            self.pitch_combo.blockSignals(True)
            idx = self.pitch_combo.findData(s.pitch_shift)
            self.pitch_combo.setCurrentIndex(idx)
            self.pitch_combo.blockSignals(False)
            
            self.prim_check.blockSignals(True)
            self.prim_check.setChecked(s.is_primary)
            self.prim_check.blockSignals(False)
        else:
            self.prop_group.setVisible(False)
            self.update_status()

    def on_prop_changed(self):
        sel = self.timeline_widget.selected_segment
        if sel:
            self.push_undo()
            sel.volume = self.vol_slider.value() / 100.0
            sel.pitch_shift = self.pitch_combo.currentData()
            sel.is_primary = self.prim_check.isChecked()
            self.timeline_widget.update()
            self.update_status()

    def duplicate_segment(self, ts):
        td = {'id': ts.id, 'filename': ts.filename, 'file_path': ts.file_path, 'bpm': ts.bpm, 'harmonic_key': ts.key, 'onsets_json': ",".join([str(x/1000.0) for x in ts.onsets])}
        ns = self.timeline_widget.add_track(td, start_ms=ts.start_ms + ts.duration_ms, lane=ts.lane)
        ns.duration_ms = ts.duration_ms
        ns.offset_ms = ts.offset_ms
        ns.volume = ts.volume
        ns.pitch_shift = ts.pitch_shift
        ns.is_primary = ts.is_primary
        ns.fade_in_ms = ts.fade_in_ms
        ns.fade_out_ms = ts.fade_out_ms
        ns.waveform = ts.waveform
        self.timeline_widget.update_geometry()
        self.timeline_widget.update()

    def capture_segment_to_library(self, ts):
        self.loading_overlay.show_loading("Capturing Processed Loop...")
        try:
            out_name = f"captured_{int(ts.start_ms)}_{ts.filename}.mp3"
            os.makedirs("captured_loops", exist_ok=True)
            out_path = os.path.abspath(os.path.join("captured_loops", out_name))
            
            tb = float(self.tbe.text()) if self.tbe.text() else 124.0
            
            self.renderer.render_timeline([ts.to_dict()], out_path, target_bpm=tb)
            
            from src.ingestion import IngestionEngine
            ie = IngestionEngine(db_path=self.dm.db_path)
            ie.scan_directory(os.path.dirname(out_path))
            
            self.load_library()
            self.loading_overlay.hide_loading()
            QMessageBox.information(self, "Captured", f"Clip captured and added to library:\\n{out_name}")
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Capture Error", "Failed to capture loop.", e)

    def on_cursor_jump(self, ms):
        if self.is_playing:
            self.player.setPosition(int(ms))
        self.update_status()

    def generate_ai_transition(self, x):
        if not self.ai_enabled:
            self.status_bar.showMessage("AI features disabled. Transition generation unavailable.")
            return
        gm = x / self.timeline_widget.pixels_per_ms
        ps = ns = None
        ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        for s in ss:
            if s.start_ms + s.duration_ms <= gm:
                ps = s
            elif s.start_ms >= gm:
                if ns is None:
                    ns = s
                    
        if not ps or not ns:
            self.status_bar.showMessage("Need track before AND after gap.")
            return
            
        self.loading_overlay.show_loading("✨ Gemini: Orchestrating Transition...")
        os.makedirs("generated_assets", exist_ok=True)
        op = os.path.abspath(f"generated_assets/trans_{int(gm)}.wav")
        
        try:
            p = self.generator.get_transition_params(ps.__dict__, ns.__dict__)
            self.generator.generate_riser(duration_sec=4.0, bpm=self.timeline_widget.target_bpm, output_path=op, params=p)
            
            td = {'id': -1, 'filename': f"AI Sweep ({p.get('description', 'Procedural')})", 'file_path': op, 'bpm': self.timeline_widget.target_bpm, 'harmonic_key': 'N/A', 'onsets_json': ""}
            sm = ns.start_ms - 4000
            seg = self.timeline_widget.add_track(td, start_ms=sm, lane=4)
            seg.duration_ms = 4000
            seg.fade_in_ms = 3500
            seg.fade_out_ms = 500
            self.load_waveform_async(seg)
            
            self.loading_overlay.hide_loading()
            self.status_bar.showMessage("AI Transition generated.")
            self.timeline_widget.update()
            self.push_undo()
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "AI Generation Error", "Failed to generate transition.", e)

    def find_bridge_for_gap(self, x):
        if not self.ai_enabled:
            self.status_bar.showMessage("AI features disabled. Bridge search unavailable.")
            return
        gm = x / self.timeline_widget.pixels_per_ms
        ps = ns = None
        ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        for s in ss:
            if s.start_ms + s.duration_ms <= gm:
                ps = s
            elif s.start_ms >= gm:
                if ns is None:
                    ns = s
                    
        if not ps or not ns:
            self.status_bar.showMessage("Need track before AND after gap.")
            return
            
        self.loading_overlay.show_loading("Finding bridge...")
        try:
            conn = self.dm.get_conn()
            conn.row_factory = sqlite3_factory
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE id NOT IN (?, ?)", (ps.id, ns.id))
            cs = cursor.fetchall()
            results = []
            
            for c in cs:
                cd = dict(c)
                ce = self.dm.get_embedding(cd['clp_embedding_id']) if cd['clp_embedding_id'] else None
                sc = self.scorer.calculate_bridge_score(ps.__dict__, ns.__dict__, cd, c_emb=ce)
                results.append((sc, cd))
                
            results.sort(key=lambda x: x[0], reverse=True)
            self.rec_list.setRowCount(0)
            
            for sc, ot in results[:15]:
                ri = self.rec_list.rowCount()
                self.rec_list.insertRow(ri)
                si = QTableWidgetItem(f"{sc}% (BRIDGE)")
                si.setData(Qt.ItemDataRole.UserRole, ot['id'])
                self.rec_list.setItem(ri, 0, si)
                self.rec_list.setItem(ri, 1, QTableWidgetItem(ot['filename']))
                
            self.loading_overlay.hide_loading()
            self.status_bar.showMessage(f"AI found {len(results)} potential bridges.")
            conn.close()
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Bridge Error", "AI Bridge search failed.", e)

    def new_project(self):
        if QMessageBox.question(self, "New Project", "Discard current journey?") == QMessageBox.StandardButton.Yes:
            self.push_undo()
            self.timeline_widget.segments = []
            self.timeline_widget.cursor_pos_ms = 0
            self.timeline_widget.loop_enabled = False
            self.preview_dirty = True
            self.timeline_widget.update_geometry()
            self.update_status()

    def on_vzoom_changed(self, v):
        self.timeline_widget.lane_height = v
        self.timeline_widget.update_geometry()

    def on_zoom_changed(self, v):
        self.timeline_widget.pixels_per_ms = v / 1000.0
        self.timeline_widget.update_geometry()

    def clear_timeline(self):
        if QMessageBox.question(self, "Clear", "Clear journey?") == QMessageBox.StandardButton.Yes:
            self.push_undo()
            self.timeline_widget.segments = []
            self.timeline_widget.update_geometry()
            self.update_status()

    def on_search_text_changed(self, t):
        if not t:
            for r in range(self.library_table.rowCount()):
                self.library_table.setRowHidden(r, False)
            return
        q = t.lower()
        for r in range(self.library_table.rowCount()):
            self.library_table.setRowHidden(r, q not in self.library_table.item(r, 0).text().lower())

    def trigger_semantic_search(self):
        if not self.ai_enabled:
            QMessageBox.warning(self, "AI Disabled", "AI features are unavailable. This often happens if model assets could not be downloaded from HuggingFace.")
            return
        q = self.search_bar.text()
        if len(q) < 3:
            return
        self.loading_overlay.show_loading(f"AI Search: '{q}'...")
        self.st = SearchThread(q, self.dm)
        self.st.resultsFound.connect(self.on_semantic_results)
        self.st.errorOccurred.connect(self.on_search_error)
        self.st.start()

    def on_semantic_results(self, res):
        self.loading_overlay.hide_loading()
        self.library_table.setRowCount(0)
        for r in res:
            ri = self.library_table.rowCount()
            self.library_table.insertRow(ri)
            match = int(max(0, 1.0 - r.get('distance', 1.0)) * 100)
            ni = QTableWidgetItem(r['filename'])
            ni.setData(Qt.ItemDataRole.UserRole, r['id'])
            if match > 70:
                ni.setForeground(QBrush(QColor(0, 255, 200)))
            self.library_table.setItem(ri, 0, ni)
            self.library_table.setItem(ri, 1, QTableWidgetItem(f"{r['bpm']:.1f}"))
            self.library_table.setItem(ri, 2, QTableWidgetItem(r['harmonic_key']))

    def on_search_error(self, e):
        self.loading_overlay.hide_loading()
        QMessageBox.warning(self, "AI Error", e)

    def save_project(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save Journey", "", "JSON Files (*.json)")
        if p:
            data = {'target_bpm': self.timeline_widget.target_bpm, 'segments': [s.to_dict() for s in self.timeline_widget.segments]}
            with open(p, 'w') as f:
                json.dump(data, f)

    def load_project(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Journey", "", "JSON Files (*.json)")
        if p:
            self.push_undo()
            with open(p, 'r') as f:
                data = json.load(f)
            self.timeline_widget.segments = []
            self.tbe.setText(str(data['target_bpm']))
            for s in data['segments']:
                td = {'id': s['id'], 'filename': s['filename'], 'file_path': s['file_path'], 'bpm': s['bpm'], 'harmonic_key': s['key'], 'onsets_json': s.get('onsets_json', "")}
                seg = TrackSegment(td, start_ms=s['start_ms'], duration_ms=s['duration_ms'], lane=s['lane'], offset_ms=s['offset_ms'])
                seg.volume = s['volume']
                seg.is_primary = s['is_primary']
                seg.fade_in_ms = s['fade_in_ms']
                seg.fade_out_ms = s['fade_out_ms']
                seg.pitch_shift = s.get('pitch_shift', 0)
                self.load_waveform_async(seg)
                self.timeline_widget.segments.append(seg)
            self.timeline_widget.update_geometry()
            self.update_status()

    def on_bpm_changed(self, t):
        try:
            self.timeline_widget.target_bpm = float(t)
            self.preview_dirty = True
            self.timeline_widget.update()
            self.update_status()
        except:
            pass

    def on_master_vol_changed(self, v):
        self.audio_output.setVolume(v / 100.0)
        self.status_bar.showMessage(f"Master Volume: {v}%")

    def toggle_analytics(self):
        self.timeline_widget.show_modifications = not self.mod_toggle.isChecked()
        self.mod_toggle.setText("🔍 Show Markers" if self.mod_toggle.isChecked() else "🔍 Hide Markers")
        self.timeline_widget.update()

    def toggle_grid(self):
        self.timeline_widget.snap_to_grid = self.grid_toggle.isChecked()
        self.grid_toggle.setText(f"📏 Grid Snap: {'ON' if self.timeline_widget.snap_to_grid else 'OFF'}")
        self.timeline_widget.update()

    def update_status(self):
        count = len(self.timeline_widget.segments)
        if count > 0:
            tdur = max(s.start_ms + s.duration_ms for s in self.timeline_widget.segments)
            abpm = sum(s.bpm for s in self.timeline_widget.segments) / count
            bdiff = abs(abpm - self.timeline_widget.target_bpm)
            self.status_bar.showMessage(f"Timeline: {count} tracks | {tdur/1000:.1f}s total mix")
            at = (f"<b>Journey Stats</b><br>Tracks: {count}<br>Duration: {tdur/1000:.1f}s<br>Avg BPM: {abpm:.1f}<br>")
            ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
            fs = 100
            for i in range(len(ss) - 1):
                s1, s2 = ss[i], ss[i+1]
                if s2.start_ms < (s1.start_ms + s1.duration_ms + 2000):
                    if self.scorer and self.scorer.calculate_harmonic_score(s1.key, s2.key) < 60:
                        fs -= 10
            at += f"<b>Flow Health:</b> {max(0, fs)}%<br>"
            if self.timeline_widget.selected_segment:
                sel = self.timeline_widget.selected_segment
                at += f"<hr><b>Selected Clip:</b><br>{sel.filename[:20]}<br>Key: {sel.key}"
                if self.scorer:
                    for o in self.timeline_widget.segments:
                        if o != sel and max(sel.start_ms, o.start_ms) < min(sel.start_ms + sel.duration_ms, o.start_ms + o.duration_ms):
                            hs = self.scorer.calculate_harmonic_score(sel.key, o.key)
                            color = "#00ff66" if hs >= 100 else "#ccff00" if hs >= 80 else "#ff5555"
                            at += f"<br>vs '{o.filename[:8]}...': <span style='color: {color};'>{hs}%</span>"
            self.stats_label.setText(at)
        else:
            self.status_bar.showMessage("Ready.")
            self.stats_label.setText("Timeline empty")

    def load_library(self):
        try:
            conn = self.dm.get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id, filename, bpm, harmonic_key FROM tracks")
            rows = cursor.fetchall()
            self.library_table.setRowCount(0)
            for r in rows:
                ri = self.library_table.rowCount()
                self.library_table.insertRow(ri)
                ni = QTableWidgetItem(r[1])
                ni.setData(Qt.ItemDataRole.UserRole, r[0])
                self.library_table.setItem(ri, 0, ni)
                self.library_table.setItem(ri, 1, QTableWidgetItem(f"{r[2]:.1f}"))
                self.library_table.setItem(ri, 2, QTableWidgetItem(r[3]))
            conn.close()
        except Exception as e:
            show_error(self, "Library Error", "Failed to load library.", e)

    def on_library_track_selected(self):
        si = self.library_table.selectedItems()
        if si:
            tid = self.library_table.item(si[0].row(), 0).data(Qt.ItemDataRole.UserRole)
            self.add_track_by_id(tid, only_update_recs=True)
            try:
                conn = self.dm.get_conn()
                cursor = conn.cursor()
                cursor.execute("SELECT file_path FROM tracks WHERE id = ?", (tid,))
                fp = cursor.fetchone()[0]
                conn.close()
                w = self.processor.get_waveform_envelope(fp)
                self.l_wave.set_waveform(w)
                self.l_wave_label.setText(os.path.basename(fp))
                self.player.setSource(QUrl.fromLocalFile(os.path.abspath(fp)))
            except:
                pass

    def add_track_by_id(self, tid, x=None, only_update_recs=False, lane=0):
        try:
            conn = self.dm.get_conn()
            conn.row_factory = sqlite3_factory
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE id = ?", (tid,))
            track = dict(cursor.fetchone())
            conn.close()
            
            if not only_update_recs:
                self.push_undo()
                sm = x / self.timeline_widget.pixels_per_ms if x is not None else None
                seg = self.timeline_widget.add_track(track, start_ms=sm)
                
                # Apply Smart Loop if available
                if track.get('loop_duration', 0) > 0:
                    seg.offset_ms = track['loop_start'] * 1000.0
                    seg.duration_ms = track['loop_duration'] * 1000.0
                
                if x is not None:
                    seg.lane = lane
                self.load_waveform_async(seg)
                self.timeline_widget.update()
                
            self.selected_library_track = track
            self.update_recommendations(tid)
        except Exception as e:
            show_error(self, "Data Error", "Failed to retrieve track.", e)

    def add_selected_to_timeline(self):
        if self.selected_library_track:
            self.add_track_by_id(self.selected_library_track['id'])

    def on_rec_double_clicked(self, i):
        self.add_track_by_id(self.rec_list.item(i.row(), 0).data(Qt.ItemDataRole.UserRole))

    def auto_populate_timeline(self):
        if not self.ai_enabled:
            QMessageBox.warning(self, "AI Disabled", "AI orchestration requires the local model engine, which failed to load.")
            return
        if not self.selected_library_track:
            self.status_bar.showMessage("Select seed track.")
            return
            
        self.push_undo()
        self.loading_overlay.show_loading("AI Orchestrating...")
        seq = self.orchestrator.find_curated_sequence(max_tracks=8, seed_track=self.selected_library_track)
        
        if seq:
            self.timeline_widget.segments = []
            cm = 0
            for i, t in enumerate(seq):
                is_f = (i % 2 == 0)
                lane = 0 if is_f else (1 if i % 4 == 1 else 2)
                dur = 30000 if is_f else 15000
                sm = cm
                
                if is_f and i > 0:
                    sm -= 8000
                elif not is_f:
                    sm = cm - 25000
                    
                seg = self.timeline_widget.add_track(t, start_ms=max(0, sm), lane=lane)
                seg.duration_ms = dur
                seg.is_primary = is_f
                seg.fade_in_ms = seg.fade_out_ms = 4000
                self.load_waveform_async(seg)
                
                if is_f:
                    cm = sm + dur
                    
            self.timeline_widget.update_geometry()
            
        self.loading_overlay.hide_loading()

    def render_timeline(self):
        if not self.timeline_widget.segments:
            return
            
        ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        tb = float(self.tbe.text()) if self.tbe.text() else 124.0
        self.loading_overlay.show_loading("Rendering Mix...")
        
        try:
            out = "timeline_mix.mp3"
            rd = [s.to_dict() for s in ss]
            self.renderer.render_timeline(rd, out, target_bpm=tb, mutes=self.timeline_widget.mutes, solos=self.timeline_widget.solos)
            self.loading_overlay.hide_loading()
            QMessageBox.information(self, "Success", f"Mix rendered: {out}")
            os.startfile(out)
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Render Error", "Failed to render.", e)

    def export_stems(self):
        if not self.timeline_widget.segments:
            return
            
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return
            
        tb = float(self.tbe.text()) if self.tbe.text() else 124.0
        self.loading_overlay.show_loading("Exporting Stems...")
        try:
            rd = [s.to_dict() for s in self.timeline_widget.segments]
            paths = self.renderer.render_stems(rd, folder, target_bpm=tb)
            self.loading_overlay.hide_loading()
            QMessageBox.information(self, "Stems Exported", f"Successfully exported {len(paths)} stems to:\\n{folder}")
            os.startfile(folder)
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Stem Export Error", "Failed to export stems.", e)

    def scan_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select Folder")
        if f:
            self.loading_overlay.show_loading("Scanning...")
            try:
                from src.ingestion import IngestionEngine
                e = IngestionEngine(db_path=self.dm.db_path)
                e.scan_directory(f)
                self.load_library()
                self.loading_overlay.hide_loading()
            except Exception as e:
                self.loading_overlay.hide_loading()
                show_error(self, "Scan Error", "Failed to scan.", e)

    def run_embedding(self):
        self.loading_overlay.show_loading("AI Indexing...")
        try:
            from src.embeddings import EmbeddingEngine
            ee = EmbeddingEngine()
            conn = self.dm.get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id, file_path, clp_embedding_id FROM tracks")
            tracks = cursor.fetchall()
            for tid, fp, ex in tracks:
                if not ex:
                    eb = ee.get_embedding(fp)
                    self.dm.add_embedding(tid, eb, metadata={"file_path": fp})
            conn.close()
            self.loading_overlay.hide_loading()
            QMessageBox.information(self, "AI Complete", "Indexed!")
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "AI Error", "Indexing failed.", e)

    def update_recommendations(self, tid):
        try:
            conn = self.dm.get_conn()
            conn.row_factory = sqlite3_factory
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE id = ?", (tid,))
            target = dict(cursor.fetchone())
            te = self.dm.get_embedding(target['clp_embedding_id']) if target['clp_embedding_id'] else None
            
            cursor.execute("SELECT * FROM tracks WHERE id != ?", (tid,))
            others = cursor.fetchall()
            results = []
            
            for o in others:
                od = dict(o)
                oe = self.dm.get_embedding(od['clp_embedding_id']) if od['clp_embedding_id'] else None
                sd = self.scorer.get_total_score(target, od, te, oe)
                results.append((sd, od))
                
            results.sort(key=lambda x: x[0]['total'], reverse=True)
            self.rec_list.setRowCount(0)
            
            for sc, ot in results[:15]:
                ri = self.rec_list.rowCount()
                self.rec_list.insertRow(ri)
                si = QTableWidgetItem(f"{sc['total']}%")
                si.setData(Qt.ItemDataRole.UserRole, ot['id'])
                
                # Enhanced Tooltip with Groove and Energy
                tooltip = (f"BPM: {sc['bpm_score']}% | Har: {sc['harmonic_score']}% | Sem: {sc['semantic_score']}%\n"
                           f"Groove: {sc.get('groove_score', 0)}% | Energy: {sc.get('energy_score', 0)}%")
                si.setToolTip(tooltip)
                
                self.rec_list.setItem(ri, 0, si)
                ni = QTableWidgetItem(ot['filename'])
                if sc['harmonic_score'] >= 100:
                    ni.setForeground(QBrush(QColor(0, 255, 100)))
                elif sc['harmonic_score'] >= 80:
                    ni.setForeground(QBrush(QColor(200, 255, 0)))
                self.rec_list.setItem(ri, 1, ni)
            conn.close()
        except Exception as e:
            print(f"Rec Engine Error: {e}")

    def play_selected(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_playback()
        elif event.key() == Qt.Key.Key_M:
            sel = self.timeline_widget.selected_segment
            if sel:
                self.timeline_widget.mutes[sel.lane] = not self.timeline_widget.mutes[sel.lane]
                self.timeline_widget.update()
                self.preview_dirty = True
        elif event.key() == Qt.Key.Key_S:
            sel = self.timeline_widget.selected_segment
            if sel:
                self.timeline_widget.solos[sel.lane] = not self.timeline_widget.solos[sel.lane]
                self.timeline_widget.update()
                self.preview_dirty = True
        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            sel = self.timeline_widget.selected_segment
            if sel:
                self.push_undo()
                self.timeline_widget.segments.remove(sel)
                self.timeline_widget.selected_segment = None
                self.on_segment_selected(None)
                self.timeline_widget.update_geometry()
                self.update_status()
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_Z:
                self.undo()
            elif event.key() == Qt.Key.Key_Y:
                self.redo()
        else:
            super().keyPressEvent(event)
