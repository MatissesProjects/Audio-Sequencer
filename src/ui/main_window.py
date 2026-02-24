import sys
import os
import sqlite3
import json
import time
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTableWidgetItem, QLineEdit, QLabel, QPushButton, 
                             QFrame, QMessageBox, QScrollArea, QFileDialog,
                             QSlider, QComboBox, QCheckBox, QStatusBar, QApplication,
                             QSplitter, QFormLayout)
from PyQt6.QtCore import Qt, QSize, QTimer, QUrl, QMimeData
from PyQt6.QtGui import QBrush, QColor, QDrag
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# Project Imports (Lightweight)
from src.database import DataManager
from src.processor import AudioProcessor
from src.renderer import FlowRenderer

# Modular Imports
from src.core.config import AppConfig
from src.core.models import TrackSegment
from src.core.undo import UndoManager
from src.ui.dialogs import show_error
from src.ui.threads import SearchThread, IngestionThread, WaveformLoader, AIInitializerThread
from src.ui.widgets import TimelineWidget, DraggableTable, LibraryWaveformPreview, LoadingOverlay

def sqlite3_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

class AudioSequencerApp(QMainWindow):
    def __init__(self):
        boot_start = time.time()
        super().__init__()
        print("[BOOT] Main Window Class Initialized")
        
        AppConfig.ensure_dirs()
        self.dm = DataManager()
        self.processor = AudioProcessor(sample_rate=AppConfig.SAMPLE_RATE)
        self.renderer = FlowRenderer(sample_rate=AppConfig.SAMPLE_RATE)
        self.undo_manager = UndoManager()
        
        # Placeholder AI state
        self.scorer = None
        self.generator = None
        self.orchestrator = None
        self.ai_enabled = False
        self.ai_loading = True
        
        self.selected_library_track = None
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        self.preview_path = os.path.join(AppConfig.GENERATED_ASSETS_DIR, "temp_preview.wav")
        self.preview_dirty = True
        
        self.play_timer = QTimer()
        self.play_timer.setInterval(20)
        self.play_timer.timeout.connect(self.update_playback_cursor)
        self.is_playing = False
        
        self.waveform_loaders = []
        self.copy_buffer = None
        self.is_library_preview = False
        
        print(f"[BOOT] Core components ready ({time.time() - boot_start:.3f}s)")
        ui_start = time.time()
        self.init_ui()
        print(f"[BOOT] UI Layout built ({time.time() - ui_start:.3f}s)")
        
        self.load_library()
        self.loading_overlay = LoadingOverlay(self.centralWidget())
        self.setAcceptDrops(True)
        
        # Start AI in background
        self.start_ai_warmup()
        print(f"[BOOT] Total window ready in {time.time() - boot_start:.3f}s")

    def start_ai_warmup(self):
        """Dispatches AI model loading to background thread."""
        self.status_bar.showMessage("Warming up AI Engine (Loading Models)...")
        self.ai_thread = AIInitializerThread()
        self.ai_thread.finished.connect(self.on_ai_ready)
        self.ai_thread.error.connect(self.on_ai_error)
        self.ai_thread.start()

    def on_ai_ready(self, s, g, o):
        self.scorer = s
        self.generator = g
        self.orchestrator = o
        self.ai_enabled = True
        self.ai_loading = False
        self.status_bar.showMessage("AI Engine Online.")
        print("[BOOT] Background AI load complete.")

    def on_ai_error(self, e):
        self.ai_enabled = False
        self.ai_loading = False
        self.status_bar.showMessage("AI Engine Offline (Check internet/models).")
        print(f"[BOOT] AI background error: {e}")
        
    def init_ui(self):
        self.setWindowTitle("AudioSequencer AI - The Pro Flow")
        self.setMinimumSize(QSize(1400, 950))
        
        cw = QWidget()
        self.setCentralWidget(cw)
        main_layout = QVBoxLayout(cw)
        
        # Main Vertical Splitter
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.setStyleSheet("QSplitter::handle { background-color: #333; }")
        
        # --- TOP AREA ---
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 5)
        
        # 1. Library Panel (Left)
        lp = QFrame()
        lp.setFixedWidth(400)
        ll = QVBoxLayout(lp)
        ll.addWidget(QLabel("<h2>📁 Library</h2>"))
        
        la = QHBoxLayout()
        self.scan_btn = QPushButton("📂 Scan")
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
        self.library_table.setHorizontalHeaderLabels(["Name", "BPM", "Key"])
        self.library_table.setColumnWidth(0, 200)
        self.library_table.itemSelectionChanged.connect(self.on_library_track_selected)
        ll.addWidget(self.library_table)
        
        pl_btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("➕ Add")
        self.add_btn.clicked.connect(self.add_selected_to_timeline)
        pl_btn_layout.addWidget(self.add_btn)
        
        self.preview_clip_btn = QPushButton("▶ Preview Clip")
        self.preview_clip_btn.clicked.connect(self.play_library_preview)
        pl_btn_layout.addWidget(self.preview_clip_btn)
        ll.addLayout(pl_btn_layout)
        
        self.l_preview = LibraryWaveformPreview()
        self.l_preview.dragStarted.connect(self.on_library_preview_drag)
        ll.addWidget(self.l_preview)
        
        self.l_wave_label = QLabel("Select range to drag specific section")
        self.l_wave_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.l_wave_label.setStyleSheet("color: #666; font-size: 10px;")
        ll.addWidget(self.l_wave_label)
        top_layout.addWidget(lp)
        
        # 2. Production Mixer & Analytics (Middle)
        mp = QFrame()
        mp.setFixedWidth(320)
        ml = QVBoxLayout(mp)
        mix_group = QFrame()
        mix_group.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;")
        mix_l = QVBoxLayout(mix_group)
        mix_l.addWidget(QLabel("<h3>🎚 Production Mixer</h3>"))
        
        self.meter_frame = QFrame()
        self.meter_frame.setFixedHeight(80)
        self.meter_frame.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff3333, stop:0.3 #33ff33, stop:1 #111); border: 1px solid #444;")
        mix_l.addWidget(self.meter_frame)
        
        mix_l.addWidget(QLabel("<b>Master FX Chain</b>"))
        self.comp_check = QCheckBox("Master Compressor")
        self.comp_check.setChecked(True)
        mix_l.addWidget(self.comp_check)
        
        self.lim_check = QCheckBox("Master Limiter")
        self.lim_check.setChecked(True)
        mix_l.addWidget(self.lim_check)
        ml.addWidget(mix_group)
        
        ag = QFrame()
        ag.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px; margin-top: 5px;")
        al = QVBoxLayout(ag)
        al.addWidget(QLabel("<b>📊 Journey Data</b>"))
        self.stats_label = QLabel("Ready.")
        self.stats_label.setWordWrap(True)
        al.addWidget(self.stats_label)
        
        lab = QHBoxLayout()
        sb = QPushButton("💾 Save")
        sb.clicked.connect(self.save_project)
        lab.addWidget(sb)
        
        lb = QPushButton("📂 Load")
        lb.clicked.connect(self.load_project)
        lab.addWidget(lb)
        al.addLayout(lab)
        ml.addWidget(ag)
        ml.addStretch()
        top_layout.addWidget(mp)
        
        # 3. Suggestions & Inspector (Right)
        rp = QFrame()
        rl = QVBoxLayout(rp)
        rl.addWidget(QLabel("<h3>✨ Smart Suggestions</h3>"))
        self.rec_list = DraggableTable(0, 2)
        self.rec_list.setFixedHeight(250)
        self.rec_list.itemDoubleClicked.connect(self.on_rec_double_clicked)
        rl.addWidget(self.rec_list)
        
        self.prop_group = QFrame()
        self.prop_group.setStyleSheet("""
            QFrame#InspectorFrame { background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; }
            QLabel { color: #ffffff; font-weight: bold; font-size: 11px; border: none; background: transparent; }
            QSlider::handle:horizontal { background: #007acc; border: 1px solid #444; width: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider::groove:horizontal { border: 1px solid #333; height: 4px; background: #252525; margin: 2px 0; }
            QCheckBox { color: #e0e0e0; }
        """)
        self.prop_group.setObjectName("InspectorFrame")
        
        # Create a container widget for the scroll area
        inspector_content = QWidget()
        inspector_content.setStyleSheet("background: transparent;")
        inspector_layout = QVBoxLayout(inspector_content)
        inspector_layout.setContentsMargins(10, 10, 10, 10)
        inspector_layout.setSpacing(10)
        
        header = QLabel("TRACK INSPECTOR")
        header.setStyleSheet("font-size: 16px; color: #00ffcc; font-weight: bold; margin-bottom: 5px;")
        inspector_layout.addWidget(header)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(10)
        
        # --- CORE PROPS ---
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 150)
        self.vol_slider.valueChanged.connect(self.on_prop_changed)
        form.addRow("Main Volume:", self.vol_slider)
        
        self.pan_slider = QSlider(Qt.Orientation.Horizontal)
        self.pan_slider.setRange(-100, 100)
        self.pan_slider.setValue(0)
        self.pan_slider.valueChanged.connect(self.on_prop_changed)
        form.addRow("Stereo Pan:", self.pan_slider)
        
        self.pitch_combo = QComboBox()
        for i in range(-6, 7):
            self.pitch_combo.addItem(f"{i:+} st", i)
        self.pitch_combo.currentIndexChanged.connect(self.on_prop_changed)
        form.addRow("Master Pitch:", self.pitch_combo)
        
        self.rev_slider = QSlider(Qt.Orientation.Horizontal)
        self.rev_slider.setRange(0, 100)
        self.rev_slider.valueChanged.connect(self.on_prop_changed)
        form.addRow("Reverb Space:", self.rev_slider)
        
        self.harm_slider = QSlider(Qt.Orientation.Horizontal)
        self.harm_slider.setRange(0, 100)
        self.harm_slider.valueChanged.connect(self.on_prop_changed)
        form.addRow("Saturation:", self.harm_slider)
        
        self.delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.delay_slider.setRange(0, 100)
        self.delay_slider.valueChanged.connect(self.on_prop_changed)
        form.addRow("Echo/Delay:", self.delay_slider)
        
        self.chorus_slider = QSlider(Qt.Orientation.Horizontal)
        self.chorus_slider.setRange(0, 100)
        self.chorus_slider.valueChanged.connect(self.on_prop_changed)
        form.addRow("Chorus/Air:", self.chorus_slider)
        
        self.vocal_shift_combo = QComboBox()
        for i in range(-12, 13):
            self.vocal_shift_combo.addItem(f"{i:+} st", i)
        self.vocal_shift_combo.setCurrentIndex(12)
        self.vocal_shift_combo.currentIndexChanged.connect(self.on_prop_changed)
        form.addRow("Vocal Shift:", self.vocal_shift_combo)
        
        self.harmony_slider = QSlider(Qt.Orientation.Horizontal)
        self.harmony_slider.setRange(0, 100)
        self.harmony_slider.valueChanged.connect(self.on_prop_changed)
        form.addRow("Voc Rhythm:", self.harmony_slider)
        
        inspector_layout.addLayout(form)

        # --- STEM MIXER ---
        inspector_layout.addSpacing(5)
        lbl_mix = QLabel("🎛 STEM MIXER")
        lbl_mix.setStyleSheet("color: #00ffcc; font-size: 11px; font-weight: bold; margin-top: 5px;")
        inspector_layout.addWidget(lbl_mix)
        
        mix_form = QFormLayout()
        mix_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.v_vol_s = QSlider(Qt.Orientation.Horizontal)
        self.v_vol_s.setRange(0, 150)
        self.v_vol_s.setValue(100)
        self.v_vol_s.valueChanged.connect(self.on_prop_changed)
        mix_form.addRow("Vocal Vol:", self.v_vol_s)
        
        self.d_vol_s = QSlider(Qt.Orientation.Horizontal)
        self.d_vol_s.setRange(0, 150)
        self.d_vol_s.setValue(100)
        self.d_vol_s.valueChanged.connect(self.on_prop_changed)
        mix_form.addRow("Drum Vol:", self.d_vol_s)
        
        self.i_vol_s = QSlider(Qt.Orientation.Horizontal)
        self.i_vol_s.setRange(0, 150)
        self.i_vol_s.setValue(100)
        self.i_vol_s.valueChanged.connect(self.on_prop_changed)
        mix_form.addRow("Instr Vol:", self.i_vol_s)
        
        inspector_layout.addLayout(mix_form)

        # --- DUCKING ---
        inspector_layout.addSpacing(5)
        lbl_duck = QLabel("🌊 AUTO-DUCKING")
        lbl_duck.setStyleSheet("color: #00ffcc; font-size: 11px; font-weight: bold; margin-top: 5px;")
        inspector_layout.addWidget(lbl_duck)
        
        duck_form = QFormLayout()
        duck_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.duck_depth_s = QSlider(Qt.Orientation.Horizontal)
        self.duck_depth_s.setRange(0, 100)
        self.duck_depth_s.setValue(70)
        self.duck_depth_s.valueChanged.connect(self.on_prop_changed)
        duck_form.addRow("Duck Depth:", self.duck_depth_s)
        
        inspector_layout.addLayout(duck_form)

        # --- CHECKBOXES ---
        cb_layout = QHBoxLayout()
        self.prim_check = QCheckBox("Primary Track")
        self.prim_check.stateChanged.connect(self.on_prop_changed)
        cb_layout.addWidget(self.prim_check)
        
        self.amb_check = QCheckBox("Ambient Track")
        self.amb_check.stateChanged.connect(self.on_prop_changed)
        cb_layout.addWidget(self.amb_check)
        
        inspector_layout.addLayout(cb_layout)
        
        # Wrap everything in a scroll area
        self.inspector_scroll = QScrollArea()
        self.inspector_scroll.setWidgetResizable(True)
        self.inspector_scroll.setWidget(inspector_content)
        self.inspector_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        # Add the scroll area to the prop_group frame
        prop_group_main_layout = QVBoxLayout(self.prop_group)
        prop_group_main_layout.setContentsMargins(0, 0, 0, 0)
        prop_group_main_layout.addWidget(self.inspector_scroll)
        
        self.prop_group.setVisible(False)
        rl.addWidget(self.prop_group)
        rl.addStretch()
        top_layout.addWidget(rp, stretch=1)
        
        self.main_splitter.addWidget(top_widget)
        
        # --- BOTTOM AREA ---
        bt = QWidget()
        bl = QVBoxLayout(bt)
        bt.setMinimumHeight(400)
        th = QHBoxLayout()
        th.addWidget(QLabel("<h2>🎞 Timeline Journey</h2>"))
        th.addSpacing(20)
        th.addWidget(QLabel("H-Zoom:"))
        self.zs = QSlider(Qt.Orientation.Horizontal)
        self.zs.setRange(10, 200)
        self.zs.setValue(50)
        self.zs.setFixedWidth(80)
        self.zs.valueChanged.connect(self.on_zoom_changed)
        th.addWidget(self.zs)
        
        th.addSpacing(10)
        th.addWidget(QLabel("V-Zoom:"))
        self.vs = QSlider(Qt.Orientation.Horizontal)
        self.vs.setRange(40, 250)
        self.vs.setValue(120)
        self.vs.setFixedWidth(80)
        self.vs.valueChanged.connect(self.on_vzoom_changed)
        th.addWidget(self.vs)
        th.addStretch()
        
        self.new_btn = QPushButton("📄 New")
        self.new_btn.clicked.connect(self.new_project)
        th.addWidget(self.new_btn)
        
        self.sb = QPushButton("⏹")
        self.sb.clicked.connect(self.jump_to_start)
        th.addWidget(self.sb)
        
        self.ptb = QPushButton("▶ Play Journey")
        self.ptb.clicked.connect(self.toggle_playback)
        th.addWidget(self.ptb)
        
        self.agb = QPushButton("🪄 Path")
        self.agb.clicked.connect(self.auto_populate_timeline)
        th.addWidget(self.agb)
        
        self.h_mix_btn = QPushButton("💥 Hyper-Mix")
        self.h_mix_btn.setStyleSheet("background-color: #528;")
        self.h_mix_btn.clicked.connect(self.auto_populate_hyper_mix)
        th.addWidget(self.h_mix_btn)
        
        self.fill_btn = QPushButton("🩹 Fill Gaps")
        self.fill_btn.setStyleSheet("background-color: #264;")
        self.fill_btn.clicked.connect(self.smart_fill_all_gaps)
        th.addWidget(self.fill_btn)
        
        th.addWidget(QLabel("BPM:"))
        self.tbe = QLineEdit("124")
        self.tbe.setFixedWidth(40)
        self.tbe.textChanged.connect(self.on_bpm_changed)
        th.addWidget(self.tbe)
        
        th.addWidget(QLabel("Master:"))
        self.mv_s = QSlider(Qt.Orientation.Horizontal)
        self.mv_s.setRange(0, 100)
        self.mv_s.setValue(80)
        self.mv_s.setFixedWidth(80)
        self.mv_s.valueChanged.connect(self.on_master_vol_changed)
        th.addWidget(self.mv_s)
        
        self.render_btn = QPushButton("🚀 RENDER")
        self.render_btn.setStyleSheet("background-color: #07c; font-weight: bold;")
        self.render_btn.clicked.connect(self.render_timeline)
        th.addWidget(self.render_btn)
        bl.addLayout(th)
        
        t_s = QScrollArea()
        t_s.setWidgetResizable(True)
        t_s.setStyleSheet("background-color: #1a1a1a;")
        self.timeline_widget = TimelineWidget()
        t_s.setWidget(self.timeline_widget)
        bl.addWidget(t_s)
        self.main_splitter.addWidget(bt)
        
        main_layout.addWidget(self.main_splitter)
        self.main_splitter.setSizes([500, 500])
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Signals
        self.timeline_widget.segmentSelected.connect(self.on_segment_selected)
        self.timeline_widget.timelineChanged.connect(self.update_status)
        self.timeline_widget.undoRequested.connect(self.push_undo)
        self.timeline_widget.cursorJumped.connect(self.on_cursor_jump)
        self.timeline_widget.bridgeRequested.connect(self.find_bridge_for_gap)
        self.timeline_widget.aiTransitionRequested.connect(self.generate_ai_transition)
        self.timeline_widget.duplicateRequested.connect(self.duplicate_segment)
        self.timeline_widget.captureRequested.connect(self.capture_segment_to_library)
        self.timeline_widget.zoomChanged.connect(lambda v: self.zs.setValue(v))
        self.timeline_widget.trackDropped.connect(self.on_track_dropped)
        self.timeline_widget.fillRangeRequested.connect(self.smart_fill_all_gaps)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI'; } 
            QLabel { color: #ffffff; } 
            QTableWidget { background-color: #1e1e1e; gridline-color: #333; color: white; border: 1px solid #333; } 
            QHeaderView::section { background-color: #333; color: white; border: 1px solid #444; padding: 5px; } 
            QPushButton { background-color: #333; color: #fff; padding: 6px; border-radius: 4px; border: 1px solid #444; } 
            QPushButton:hover { background-color: #444; } 
            QLineEdit { background-color: #222; color: white; border: 1px solid #444; } 
            QComboBox { background-color: #333; color: white; } 
            QCheckBox { color: white; } 
            QScrollBar:vertical { width: 12px; background: #222; } 
            QScrollBar::handle:vertical { background: #444; border-radius: 6px; }
        """)

    def on_track_dropped(self, tid_str, x, y):
        # tid_str might be "tid" or "tid:start:end"
        parts = str(tid_str).split(':')
        tid = int(parts[0])
        selection_range = None
        if len(parts) == 3:
            selection_range = (float(parts[1]), float(parts[2]))

        # Header is 40px tall, then lanes are (height + spacing)
        # We use the widget's internal coordinates (x,y)
        lane = max(0, min(7, int((y - 40) // (self.timeline_widget.lane_height + self.timeline_widget.lane_spacing))))
        print(f"[UI] Track {tid} dropped at x={x}, y={y} -> Lane {lane} (Range: {selection_range})")
        self.add_track_by_id(tid, x=x, lane=lane, selection_range=selection_range)
        self.timeline_widget.update_geometry()
        self.timeline_widget.update()

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
        if self.is_library_preview:
            p = self.player.position()
            if self.l_preview.selection_end is not None:
                dur = self.player.duration()
                end_ms = int(self.l_preview.selection_end * dur)
                if p >= end_ms:
                    self.player.pause()
                    self.is_library_preview = False
                    self.play_timer.stop()
            elif p >= self.player.duration() and self.player.duration() > 0:
                self.player.stop()
                self.is_library_preview = False
                self.play_timer.stop()
            return

        if self.is_playing:
            p = self.player.position()
            if self.timeline_widget.loop_enabled and p >= self.timeline_widget.loop_end_ms:
                self.player.setPosition(int(self.timeline_widget.loop_start_ms))
                p = self.timeline_widget.loop_start_ms
            
            self.timeline_widget.cursor_pos_ms = p
            self.timeline_widget.update()
            
            t_e = 0
            for s in self.timeline_widget.segments:
                if s.start_ms <= p <= s.get_end_ms():
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
        self.is_library_preview = False
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
        ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        self.loading_overlay.show_loading("Building Sonic Preview...", total=len(ss))
        try:
            tb = float(self.tbe.text()) if self.tbe.text() else 124.0
            rd = [s.to_dict() for s in ss]
            self.renderer.render_timeline(rd, self.preview_path, target_bpm=tb, 
                                          mutes=self.timeline_widget.mutes, solos=self.timeline_widget.solos,
                                          progress_cb=self.loading_overlay.set_progress)
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
            td = {
                'id': s['id'], 
                'filename': s['filename'], 
                'file_path': s['file_path'], 
                'bpm': s['bpm'], 
                'harmonic_key': s['key'], 
                'onsets_json': s.get('onsets_json', "")
            }
            seg = TrackSegment(td, start_ms=s['start_ms'], duration_ms=s['duration_ms'], lane=s['lane'], offset_ms=s['offset_ms'])
            seg.volume = s['volume']
            seg.pan = s.get('pan', 0.0)
            seg.is_primary = s['is_primary']
            seg.fade_in_ms = s['fade_in_ms']
            seg.fade_out_ms = s['fade_out_ms']
            seg.pitch_shift = s.get('pitch_shift', 0)
            seg.reverb = s.get('reverb', 0.0)
            seg.harmonics = s.get('harmonics', 0.0)
            seg.vocal_shift = s.get('vocal_shift', 0)
            seg.harmony_level = s.get('harmony_level', 0.0)
            seg.vocal_vol = s.get('vocal_vol', 1.0)
            seg.drum_vol = s.get('drum_vol', 1.0)
            seg.instr_vol = s.get('instr_vol', 1.0)
            seg.ducking_depth = s.get('ducking_depth', 0.7)
            
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
            
            self.pan_slider.blockSignals(True)
            self.pan_slider.setValue(int(s.pan * 100))
            self.pan_slider.blockSignals(False)
            
            self.pitch_combo.blockSignals(True)
            idx = self.pitch_combo.findData(s.pitch_shift)
            self.pitch_combo.setCurrentIndex(idx)
            self.pitch_combo.blockSignals(False)
            
            self.rev_slider.blockSignals(True)
            self.rev_slider.setValue(int(s.reverb * 100))
            self.rev_slider.blockSignals(False)
            
            self.harm_slider.blockSignals(True)
            self.harm_slider.setValue(int(s.harmonics * 100))
            self.harm_slider.blockSignals(False)
            
            self.delay_slider.blockSignals(True)
            self.delay_slider.setValue(int(s.delay * 100))
            self.delay_slider.blockSignals(False)
            
            self.chorus_slider.blockSignals(True)
            self.chorus_slider.setValue(int(s.chorus * 100))
            self.chorus_slider.blockSignals(False)
            
            self.vocal_shift_combo.blockSignals(True)
            idx = self.vocal_shift_combo.findData(s.vocal_shift)
            self.vocal_shift_combo.setCurrentIndex(idx)
            self.vocal_shift_combo.blockSignals(False)
            
            self.harmony_slider.blockSignals(True)
            self.harmony_slider.setValue(int(s.harmony_level * 100))
            self.harmony_slider.blockSignals(False)
            
            self.v_vol_s.blockSignals(True)
            self.v_vol_s.setValue(int(s.vocal_vol * 100))
            self.v_vol_s.blockSignals(False)
            
            self.d_vol_s.blockSignals(True)
            self.d_vol_s.setValue(int(s.drum_vol * 100))
            self.d_vol_s.blockSignals(False)
            
            self.i_vol_s.blockSignals(True)
            self.i_vol_s.setValue(int(s.instr_vol * 100))
            self.i_vol_s.blockSignals(False)
            
            self.duck_depth_s.blockSignals(True)
            self.duck_depth_s.setValue(int(s.ducking_depth * 100))
            self.duck_depth_s.blockSignals(False)
            
            self.prim_check.blockSignals(True)
            self.prim_check.setChecked(s.is_primary)
            self.prim_check.blockSignals(False)
            
            self.amb_check.blockSignals(True)
            self.amb_check.setChecked(s.is_ambient)
            self.amb_check.blockSignals(False)
        else:
            self.prop_group.setVisible(False)
            self.update_status()

    def on_prop_changed(self):
        sel = self.timeline_widget.selected_segment
        if sel:
            self.push_undo()
            sel.volume = self.vol_slider.value() / 100.0
            sel.pan = self.pan_slider.value() / 100.0
            sel.pitch_shift = self.pitch_combo.currentData()
            sel.reverb = self.rev_slider.value() / 100.0
            sel.harmonics = self.harm_slider.value() / 100.0
            sel.delay = self.delay_slider.value() / 100.0
            sel.chorus = self.chorus_slider.value() / 100.0
            sel.vocal_shift = self.vocal_shift_combo.currentData()
            sel.harmony_level = self.harmony_slider.value() / 100.0
            
            sel.vocal_vol = self.v_vol_s.value() / 100.0
            sel.drum_vol = self.d_vol_s.value() / 100.0
            sel.instr_vol = self.i_vol_s.value() / 100.0
            sel.ducking_depth = self.duck_depth_s.value() / 100.0
            
            sel.is_primary = self.prim_check.isChecked()
            sel.is_ambient = self.amb_check.isChecked()
            self.timeline_widget.update()
            self.update_status()

    def duplicate_segment(self, ts):
        td = {
            'id': ts.id, 
            'filename': ts.filename, 
            'file_path': ts.file_path, 
            'bpm': ts.bpm, 
            'harmonic_key': ts.key, 
            'onsets_json': ",".join([str(x/1000.0) for x in ts.onsets])
        }
        ns = self.timeline_widget.add_track(td, start_ms=ts.get_end_ms(), lane=ts.lane)
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
        self.loading_overlay.show_loading("Capturing...")
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
            QMessageBox.information(self, "Captured", f"Clip captured:\n{out_name}")
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Capture Error", "Failed.", e)

    def on_cursor_jump(self, ms):
        if self.is_playing:
            self.player.setPosition(int(ms))
        self.update_status()

    def generate_ai_transition(self, x):
        if not self.ai_enabled:
            self.status_bar.showMessage("AI features disabled.")
            return
        gm = x / self.timeline_widget.pixels_per_ms
        ps = ns = None
        ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        for s in ss:
            if s.get_end_ms() <= gm:
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
            td = {
                'id': -1, 
                'filename': f"AI Sweep ({p.get('description', 'Procedural')})", 
                'file_path': op, 
                'bpm': self.timeline_widget.target_bpm, 
                'harmonic_key': 'N/A', 
                'onsets_json': ""
            }
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
            show_error(self, "AI Generation Error", "Failed.", e)

    def find_bridge_for_gap(self, x):
        if not self.ai_enabled:
            self.status_bar.showMessage("AI features disabled.")
            return
        gm = x / self.timeline_widget.pixels_per_ms
        ps = ns = None
        ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        for s in ss:
            if s.get_end_ms() <= gm:
                ps = s
            elif s.start_ms >= gm:
                if ns is None:
                    ns = s
        if not ps or not ns:
            self.status_bar.showMessage("Need track before AND after.")
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
                ni = QTableWidgetItem(ot['filename'])
                self.rec_list.setItem(ri, 1, ni)
            self.loading_overlay.hide_loading()
            self.status_bar.showMessage(f"AI found {len(results)} potential bridges.")
            conn.close()
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Bridge Error", "Failed.", e)

    def new_project(self):
        if QMessageBox.question(self, "New Project", "Discard current journey?") == QMessageBox.StandardButton.Yes:
            self.push_undo()
            self.timeline_widget.segments = []
            self.timeline_widget.cursor_pos_ms = 0
            self.timeline_widget.loop_enabled = False
            self.preview_dirty = True
            self.timeline_widget.update_geometry()
            self.update_status()

    def smart_fill_all_gaps(self, range_start=None, range_end=None):
        if not self.ai_enabled:
            QMessageBox.warning(self, "AI Disabled", "AI Gap Filling requires the AI Engine.")
            return
        
        self.push_undo()
        self.loading_overlay.show_loading("AI Healing Gaps...")
        
        try:
            if range_start is not None and range_end is not None:
                gaps = [(range_start, range_end)]
            else:
                gaps = self.timeline_widget.find_silence_regions()

            if not gaps and self.timeline_widget.segments:
                self.loading_overlay.hide_loading()
                self.status_bar.showMessage("No significant gaps detected.")
                return
            
            # If timeline is empty and no gaps found, create an initial gap from 0 to 30s
            if not self.timeline_widget.segments and not gaps:
                gaps = [(0.0, 30000.0)]
                
            # Determine the absolute end of the arrangement
            abs_end = max([s.get_end_ms() for s in self.timeline_widget.segments]) if self.timeline_widget.segments else 0
            
            filled_count = 0
            for start, end in gaps:
                # Rule: Don't fill if the gap is at the very end (taper out) and it's not a forced range
                if range_start is None and end >= abs_end - 500 and self.timeline_widget.segments:
                    continue
                
                # Find surrounding tracks for context
                prev_track = None
                next_track = None
                
                for s in self.timeline_widget.segments:
                    if s.get_end_ms() <= start + 500:
                        if not prev_track or (s.get_end_ms() > prev_track.get_end_ms()):
                            prev_track = s
                    if s.start_ms >= end - 500:
                        if not next_track or s.start_ms < next_track.start_ms:
                            next_track = s
                
                # If no surrounding tracks (empty timeline case), use selected library track as seed
                seed_id = self.selected_library_track['id'] if self.selected_library_track else None
                
                # Find best filler
                filler_data = self.orchestrator.find_best_filler_for_gap(
                    prev_track_id=prev_track.id if prev_track else seed_id,
                    next_track_id=next_track.id if next_track else None
                )
                
                if filler_data:
                    # Place filler to cover gap + 2s overlap on each side
                    f_dur = (end - start) + 4000
                    f_start = start - 2000
                    
                    # Ensure start is not negative
                    if f_start < 0:
                        f_dur += f_start
                        f_start = 0
                        
                    # Find a free lane or use lane 7
                    busy_lanes = set()
                    for s in self.timeline_widget.segments:
                        if max(f_start, s.start_ms) < min(f_start + f_dur, s.get_end_ms()):
                            busy_lanes.add(s.lane)
                    
                    lane = 7
                    for l in [7, 6, 5, 4, 3, 2, 1]: # Prefer higher lanes for fill
                        if l not in busy_lanes:
                            lane = l
                            break
                            
                    seg = self.timeline_widget.add_track(filler_data, start_ms=f_start, lane=lane)
                    seg.duration_ms = f_dur
                    seg.volume = 0.5 # Default ducked
                    seg.is_ambient = True
                    seg.fade_in_ms = 2000
                    seg.fade_out_ms = 2000
                    self.load_waveform_async(seg)
                    filled_count += 1
            
            self.timeline_widget.update_geometry()
            self.timeline_widget.find_silence_regions() # Refresh warnings
            self.loading_overlay.hide_loading()
            self.status_bar.showMessage(f"AI: Healed {filled_count} energy gaps.")
            
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Gap Fill Error", "Failed to fill gaps.", e)
    
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
            QMessageBox.warning(self, "AI Disabled", "AI features are unavailable.")
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
            data = {
                'target_bpm': self.timeline_widget.target_bpm, 
                'segments': [s.to_dict() for s in self.timeline_widget.segments]
            }
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
                td = {
                    'id': s['id'], 
                    'filename': s['filename'], 
                    'file_path': s['file_path'], 
                    'bpm': s['bpm'], 
                    'harmonic_key': s['key'], 
                    'onsets_json': s.get('onsets_json', "")
                }
                seg = TrackSegment(td, start_ms=s['start_ms'], duration_ms=s['duration_ms'], lane=s['lane'], offset_ms=s['offset_ms'])
                seg.volume = s['volume']
                seg.pan = s.get('pan', 0.0)
                seg.is_primary = s['is_primary']
                seg.fade_in_ms = s['fade_in_ms']
                seg.fade_out_ms = s['fade_out_ms']
                seg.pitch_shift = s.get('pitch_shift', 0)
                seg.reverb = s.get('reverb', 0.0)
                seg.harmonics = s.get('harmonics', 0.0)
                seg.vocal_shift = s.get('vocal_shift', 0)
                seg.harmony_level = s.get('harmony_level', 0.0)
                seg.delay = s.get('delay', 0.0)
                seg.chorus = s.get('chorus', 0.0)
                seg.vocal_vol = s.get('vocal_vol', 1.0)
                seg.drum_vol = s.get('drum_vol', 1.0)
                seg.instr_vol = s.get('instr_vol', 1.0)
                seg.ducking_depth = s.get('ducking_depth', 0.7)
                
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
            tdur = max(s.get_end_ms() for s in self.timeline_widget.segments)
            abpm = sum(s.bpm for s in self.timeline_widget.segments) / count
            self.status_bar.showMessage(f"Timeline: {count} tracks | {tdur/1000:.1f}s total mix")
            
            at = (f"<b>Journey Stats</b><br>Tracks: {count}<br>Duration: {tdur/1000:.1f}s<br>Avg BPM: {abpm:.1f}<br>")
            ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
            fs = 100
            for i in range(len(ss) - 1):
                s1, s2 = ss[i], ss[i+1]
                if s2.start_ms < (s1.get_end_ms() + 2000):
                    if self.scorer and self.scorer.calculate_harmonic_score(s1.key, s2.key) < 60:
                        fs -= 10
            at += f"<b>Flow Health:</b> {max(0, fs)}%<br>"
            
            # Silence Guard Check
            gaps = self.timeline_widget.find_silence_regions()
            if gaps:
                at += f"<br><span style='color: #ff5555;'>⚠ <b>Silence Guard:</b> {len(gaps)} gaps detected!</span>"
            
            if self.timeline_widget.selected_segment:
                sel = self.timeline_widget.selected_segment
                at += f"<hr><b>Selected Clip:</b><br>{sel.filename[:20]}<br>Key: {sel.key}"
                if self.scorer:
                    for o in self.timeline_widget.segments:
                        if o != sel and sel.overlaps_with(o):
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
                self.l_preview.set_waveform(w)
                self.l_wave_label.setText(os.path.basename(fp))
                self.player.setSource(QUrl.fromLocalFile(os.path.abspath(fp)))
            except:
                pass

    def play_library_preview(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_timer.stop()
            self.is_library_preview = False
        else:
            self.is_library_preview = True
            if self.l_preview.selection_start is not None:
                dur = self.player.duration()
                if dur > 0:
                    start_ms = int(self.l_preview.selection_start * dur)
                    self.player.setPosition(start_ms)
            self.player.play()
            self.play_timer.start()

    def on_library_preview_drag(self, start_pct, end_pct):
        si = self.library_table.selectedItems()
        if si:
            tid = self.library_table.item(si[0].row(), 0).data(Qt.ItemDataRole.UserRole)
            # Standard Qt Drag
            drag = QDrag(self)
            mime = QMimeData()
            # Special format for range: tid:start_pct:end_pct
            mime.setText(f"{tid}:{start_pct}:{end_pct}")
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.CopyAction)

    def add_track_by_id(self, tid, x=None, only_update_recs=False, lane=0, selection_range=None):
        try:
            conn = self.dm.get_conn()
            conn.row_factory = sqlite3_factory
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE id = ?", (tid,))
            row = cursor.fetchone()
            if not row:
                if not only_update_recs:
                    print(f"[UI] Track ID {tid} not found in database.")
                conn.close()
                return
            track = dict(row)
            conn.close()
            
            if not only_update_recs:
                self.push_undo()
                sm = x / self.timeline_widget.pixels_per_ms if x is not None else None
                seg = self.timeline_widget.add_track(track, start_ms=sm)
                loop_dur = track.get('loop_duration') or 0
                loop_start = track.get('loop_start') or 0
                
                if loop_dur > 0:
                    seg.offset_ms = loop_start * 1000.0
                    seg.duration_ms = loop_dur * 1000.0
                
                # Apply selection range if provided
                if selection_range:
                    s_pct, e_pct = selection_range
                    base_dur = seg.duration_ms
                    seg.offset_ms = s_pct * base_dur
                    seg.duration_ms = (e_pct - s_pct) * base_dur

                if x is not None:
                    seg.lane = lane
                self.load_waveform_async(seg)
                self.timeline_widget.update()
            
            self.selected_library_track = track
            self.update_recommendations(tid)
        except Exception as e:
            show_error(self, "Data Error", "Failed.", e)

    def copy_selected_segment(self):
        sel = self.timeline_widget.selected_segment
        if sel:
            self.copy_buffer = sel.to_dict()
            self.status_bar.showMessage(f"Copied: {sel.filename}")

    def paste_segment(self):
        if self.copy_buffer:
            self.push_undo()
            # Paste at cursor position
            start_ms = self.timeline_widget.cursor_pos_ms
            # Find a free lane at this position or use lane 0
            lane = 0
            
            seg = self.timeline_widget.add_track(self.copy_buffer, start_ms=start_ms, lane=lane)
            # Restore properties from buffer
            seg.duration_ms = self.copy_buffer['duration_ms']
            seg.offset_ms = self.copy_buffer['offset_ms']
            seg.volume = self.copy_buffer['volume']
            seg.pan = self.copy_buffer.get('pan', 0.0)
            seg.pitch_shift = self.copy_buffer.get('pitch_shift', 0)
            seg.reverb = self.copy_buffer.get('reverb', 0.0)
            seg.harmonics = self.copy_buffer.get('harmonics', 0.0)
            seg.delay = self.copy_buffer.get('delay', 0.0)
            seg.chorus = self.copy_buffer.get('chorus', 0.0)
            seg.vocal_shift = self.copy_buffer.get('vocal_shift', 0)
            seg.harmony_level = self.copy_buffer.get('harmony_level', 0.0)
            seg.vocal_vol = self.copy_buffer.get('vocal_vol', 1.0)
            seg.drum_vol = self.copy_buffer.get('drum_vol', 1.0)
            seg.instr_vol = self.copy_buffer.get('instr_vol', 1.0)
            seg.ducking_depth = self.copy_buffer.get('ducking_depth', 0.7)
            seg.is_primary = self.copy_buffer['is_primary']
            seg.is_ambient = self.copy_buffer.get('is_ambient', False)
            
            self.load_waveform_async(seg)
            self.timeline_widget.update()
            self.status_bar.showMessage(f"Pasted: {seg.filename}")

    def add_selected_to_timeline(self):
        if self.selected_library_track:
            self.add_track_by_id(self.selected_library_track['id'])

    def on_rec_double_clicked(self, i):
        self.add_track_by_id(self.rec_list.item(i.row(), 0).data(Qt.ItemDataRole.UserRole))

    def auto_populate_timeline(self):
        if not self.ai_enabled:
            QMessageBox.warning(self, "AI Disabled", "AI Engine Offline.")
            return
        
        # Use existing timeline state if available
        seed = self.selected_library_track
        start_ms = 0
        
        if self.timeline_widget.segments:
            # Find the latest track to continue from it
            last_seg = max(self.timeline_widget.segments, key=lambda s: s.get_end_ms())
            start_ms = last_seg.get_end_ms() - 8000 # 8s overlap
            # Use last track as seed if nothing specifically selected in library
            if not seed:
                try:
                    conn = self.dm.get_conn()
                    conn.row_factory = sqlite3_factory
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM tracks WHERE id = ?", (last_seg.id,))
                    seed = dict(cursor.fetchone())
                    conn.close()
                except:
                    pass

        if not seed:
            self.status_bar.showMessage("Select a track in the library to start the path.")
            return

        self.push_undo()
        self.loading_overlay.show_loading("AI Continuing Path...")
        # Get sequence starting from seed
        seq = self.orchestrator.find_curated_sequence(max_tracks=4, seed_track=seed)
        
        if seq:
            # If we used the last track as seed, skip it in the sequence to avoid duplicates
            if self.timeline_widget.segments and seq[0]['id'] == last_seg.id:
                seq = seq[1:]
                
            cm = start_ms
            for i, t in enumerate(seq):
                is_f = (i % 2 == 0)
                lane = 0 if is_f else (1 if i % 4 == 1 else 2)
                dur = 30000 if is_f else 15000
                sm = cm
                # Overlap logic for continuation
                if i > 0:
                    if is_f:
                        sm -= 8000
                    else:
                        sm = cm - 25000
                
                seg = self.timeline_widget.add_track(t, start_ms=max(0, sm), lane=lane)
                seg.duration_ms = dur
                seg.is_primary = is_f
                seg.fade_in_ms = seg.fade_out_ms = 4000
                self.load_waveform_async(seg)
                if is_f:
                    cm = sm + dur
            
            self.timeline_widget.update_geometry()
            self.status_bar.showMessage(f"AI: Added {len(seq)} compatible tracks to the journey.")
        
        self.loading_overlay.hide_loading()

    def auto_populate_hyper_mix(self):
        if not self.ai_enabled:
            QMessageBox.warning(self, "AI Disabled", "AI Engine Offline.")
            return
        
        start_ms = 0
        if self.timeline_widget.segments:
            last_seg = max(self.timeline_widget.segments, key=lambda s: s.get_end_ms())
            start_ms = last_seg.get_end_ms()
            
        self.push_undo()
        self.loading_overlay.show_loading("Synthesizing Hyper-Mix...")
        
        try:
            h_segs = self.orchestrator.get_hyper_segments(seed_track=self.selected_library_track, start_time_ms=start_ms)
            if h_segs:
                for sd in h_segs:
                    seg = self.timeline_widget.add_track(sd, start_ms=sd['start_ms'], lane=sd['lane'])
                    seg.duration_ms = sd['duration_ms']
                    seg.offset_ms = sd['offset_ms']
                    seg.volume = sd['volume']
                    seg.pan = sd.get('pan', 0.0)
                    seg.is_primary = sd.get('is_primary', False)
                    seg.pitch_shift = sd.get('pitch_shift', 0)
                    seg.low_cut = sd.get('low_cut', 20)
                    seg.high_cut = sd.get('high_cut', 20000)
                    seg.fade_in_ms = sd['fade_in_ms']
                    seg.fade_out_ms = sd['fade_out_ms']
                    # New props
                    seg.vocal_vol = sd.get('vocal_vol', 1.0)
                    seg.drum_vol = sd.get('drum_vol', 1.0)
                    seg.instr_vol = sd.get('instr_vol', 1.0)
                    seg.ducking_depth = sd.get('ducking_depth', 0.7)
                    seg.reverb = sd.get('reverb', 0.0)
                    seg.harmony_level = sd.get('harmony_level', 0.0)
                    seg.vocal_shift = sd.get('vocal_shift', 0)
                    
                    self.load_waveform_async(seg)
                self.timeline_widget.update_geometry()
                self.status_bar.showMessage(f"AI: Appended Hyper-Mix structure to the journey.")
            self.loading_overlay.hide_loading()
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Hyper Error", "Failed.", e)

    def render_timeline(self):
        if not self.timeline_widget.segments:
            return
        
        # Check if we should render a specific region
        time_range = None
        if self.timeline_widget.loop_enabled:
            msg = QMessageBox(self)
            msg.setWindowTitle("Render Options")
            msg.setText("Would you like to render the entire journey or just the selected loop region?")
            full_btn = msg.addButton("Entire Journey", QMessageBox.ButtonRole.ActionRole)
            sel_btn = msg.addButton("Selected Region", QMessageBox.ButtonRole.ActionRole)
            msg.addButton(QMessageBox.StandardButton.Cancel)
            msg.exec()
            
            if msg.clickedButton() == sel_btn:
                time_range = (self.timeline_widget.loop_start_ms, self.timeline_widget.loop_end_ms)
            elif msg.clickedButton() != full_btn:
                return # Cancelled

        ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        self.loading_overlay.show_loading("Rendering Mix...", total=len(ss))
        try:
            tb = float(self.tbe.text()) if self.tbe.text() else 124.0
            rd = [s.to_dict() for s in ss]
            self.renderer.render_timeline(rd, "timeline_mix.mp3", target_bpm=tb, 
                                          mutes=self.timeline_widget.mutes, solos=self.timeline_widget.solos,
                                          progress_cb=self.loading_overlay.set_progress,
                                          time_range=time_range)
            self.loading_overlay.hide_loading()
            QMessageBox.information(self, "Success", "Mix rendered: timeline_mix.mp3")
            os.startfile("timeline_mix.mp3")
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Render Error", "Failed.", e)

    def export_stems(self):
        if not self.timeline_widget.segments:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        
        self.loading_overlay.show_loading("Exporting Multi-Lane Stems...", total=len(self.timeline_widget.segments))
        try:
            tb = float(self.tbe.text()) if self.tbe.text() else 124.0
            rd = [s.to_dict() for s in self.timeline_widget.segments]
            self.renderer.render_stems(rd, folder, target_bpm=tb, progress_cb=self.loading_overlay.set_progress)
            self.loading_overlay.hide_loading()
            QMessageBox.information(self, "Exported", f"Stems exported to:\n{folder}")
            os.startfile(folder)
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Export Error", "Failed.", e)

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
                show_error(self, "Scan Error", "Failed.", e)

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
            QMessageBox.information(self, "Complete", "Indexed!")
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "AI Error", "Failed.", e)

    def update_recommendations(self, tid):
        if not self.scorer:
            self.rec_list.setRowCount(0)
            return
        try:
            tid = int(tid)
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
                si.setToolTip(f"BPM: {sc['bpm_score']}% | Har: {sc['harmonic_score']}% | Sem: {sc['semantic_score']}\nGroove: {sc.get('groove_score', 0)}% | Energy: {sc.get('energy_score', 0)}%")
                self.rec_list.setItem(ri, 0, si)
                ni = QTableWidgetItem(ot['filename'])
                ni.setForeground(QBrush(QColor(0, 255, 100)) if sc['harmonic_score'] >= 80 else QBrush(QColor(255, 255, 255)))
                self.rec_list.setItem(ri, 1, ni)
            conn.close()
        except Exception as e:
            print(f"[RECS] Error updating recommendations: {e}")

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
            elif event.key() == Qt.Key.Key_C:
                self.copy_selected_segment()
            elif event.key() == Qt.Key.Key_V or event.key() == Qt.Key.Key_P:
                self.paste_segment()
            elif event.key() == Qt.Key.Key_B:
                # Blade Tool: Split selected at cursor
                sel = self.timeline_widget.selected_segment
                if sel:
                    cur_x = self.timeline_widget.cursor_pos_ms * self.timeline_widget.pixels_per_ms
                    self.push_undo()
                    self.timeline_widget.split_segment(sel, cur_x)
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioSequencerApp()
    window.show()
    sys.exit(app.exec())
