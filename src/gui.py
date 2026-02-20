import sys
import os
import sqlite3
import traceback
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTableWidget, QTableWidgetItem, 
                             QLineEdit, QLabel, QPushButton, QFrame, QMessageBox,
                             QScrollArea, QMenu, QDialog, QTextEdit, QStatusBar, QFileDialog,
                             QSlider)
from PyQt6.QtCore import Qt, QSize, QRect, pyqtSignal, QPoint, QMimeData, QThread, QTimer
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont, QDrag

# Project Imports
from src.database import DataManager
from src.scoring import CompatibilityScorer
from src.processor import AudioProcessor
from src.renderer import FlowRenderer
from src.generator import TransitionGenerator
from src.orchestrator import FullMixOrchestrator
from src.embeddings import EmbeddingEngine

class DetailedErrorDialog(QDialog):
    def __init__(self, title, message, details, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout(self)
        msg_layout = QHBoxLayout()
        icon_label = QLabel("❌"); icon_label.setStyleSheet("font-size: 32px;"); msg_layout.addWidget(icon_label)
        msg_label = QLabel(message); msg_label.setWordWrap(True); msg_label.setStyleSheet("font-size: 14px; font-weight: bold;"); msg_layout.addWidget(msg_label, stretch=1)
        layout.addLayout(msg_layout); layout.addWidget(QLabel("Technical Details:"))
        self.details_box = QTextEdit(); self.details_box.setReadOnly(True); self.details_box.setText(details)
        self.details_box.setStyleSheet("background-color: #1a1a1a; color: #ff5555; font-family: Consolas, monospace;"); layout.addWidget(self.details_box)
        btn_layout = QHBoxLayout()
        copy_btn = QPushButton("📋 Copy to Clipboard"); copy_btn.clicked.connect(self.copy_to_clipboard)
        btn_layout.addWidget(copy_btn); close_btn = QPushButton("Close"); close_btn.clicked.connect(self.accept); close_btn.setDefault(True)
        btn_layout.addWidget(close_btn); layout.addLayout(btn_layout)
        self.setStyleSheet("QDialog { background-color: #252525; color: white; } QLabel { color: white; } QPushButton { background-color: #444; color: white; padding: 8px; border-radius: 4px; }")
    def copy_to_clipboard(self): QApplication.clipboard().setText(self.details_box.toPlainText()); QMessageBox.information(self, "Copied", "Error details copied to clipboard.")

def show_error(parent, title, message, exception):
    details = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    dialog = DetailedErrorDialog(title, message, details, parent); dialog.exec()

class SearchThread(QThread):
    resultsFound = pyqtSignal(list)
    errorOccurred = pyqtSignal(str)
    def __init__(self, query, dm): super().__init__(); self.query = query; self.dm = dm
    def run(self):
        try:
            engine = EmbeddingEngine(); text_emb = engine.get_text_embedding(self.query)
            results = self.dm.search_embeddings(text_emb, n_results=20); self.resultsFound.emit(results)
        except Exception as e: self.errorOccurred.emit(str(e))

class TrackSegment:
    def __init__(self, track_data, start_ms=0, duration_ms=20000, lane=0, offset_ms=0):
        self.id = track_data['id']; self.filename = track_data['filename']; self.file_path = track_data['file_path']
        self.bpm = track_data['bpm']; self.key = track_data['harmonic_key']
        self.start_ms = start_ms; self.duration_ms = duration_ms; self.offset_ms = offset_ms
        self.volume = 1.0; self.lane = lane; self.is_primary = False; self.waveform = []
        self.fade_in_ms = 2000; self.fade_out_ms = 2000; self.pitch_shift = 0
        self.color = QColor(70, 130, 180, 200)
    def to_dict(self):
        return {'id': self.id, 'filename': self.filename, 'file_path': self.file_path, 'bpm': self.bpm, 'key': self.key, 'start_ms': self.start_ms, 'duration_ms': self.duration_ms, 'offset_ms': self.offset_ms, 'volume': self.volume, 'lane': self.lane, 'is_primary': self.is_primary, 'fade_in_ms': self.fade_in_ms, 'fade_out_ms': self.fade_out_ms, 'pitch_shift': self.pitch_shift}

class UndoManager:
    def __init__(self): self.undo_stack = []; self.redo_stack = []
    def push_state(self, segments):
        state = [json.dumps(s.to_dict()) for s in segments]
        self.undo_stack.append(state); self.redo_stack.clear()
        if len(self.undo_stack) > 50: self.undo_stack.pop(0)
    def undo(self, current_segments):
        if not self.undo_stack: return None
        self.redo_stack.append([json.dumps(s.to_dict()) for s in current_segments]); return self.undo_stack.pop()
    def redo(self, current_segments):
        if not self.redo_stack: return None
        self.undo_stack.append([json.dumps(s.to_dict()) for s in current_segments]); return self.redo_stack.pop()

class DraggableTable(QTableWidget):
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if item:
                row = item.row(); tid = self.item(row, 0).data(Qt.ItemDataRole.UserRole)
                if tid is not None:
                    drag = QDrag(self); mime = QMimeData(); mime.setText(str(tid)); drag.setMimeData(mime); drag.exec(Qt.DropAction.CopyAction)
        super().mousePressEvent(event)

class TimelineWidget(QWidget):
    segmentSelected = pyqtSignal(object)
    timelineChanged = pyqtSignal()
    def __init__(self):
        super().__init__(); self.segments = []; self.setMinimumHeight(550); self.setAcceptDrops(True); self.pixels_per_ms = 0.05
        self.selected_segment = None; self.dragging = self.resizing = self.vol_dragging = self.fade_in_dragging = self.fade_out_dragging = False
        self.drag_start_pos = None; self.drag_start_ms = self.drag_start_dur = self.drag_start_fade = 0; self.drag_start_vol = 1.0; self.drag_start_lane = 0
        self.lane_height = 120; self.lane_spacing = 10; self.snap_threshold_ms = 2000; self.target_bpm = 124.0
        self.show_modifications = True; self.cursor_pos_ms = 0; self.show_waveforms = True; self.snap_to_grid = True; self.update_geometry()
    
    def update_geometry(self):
        max_ms = 600000; 
        if self.segments: max_ms = max(max_ms, max(s.start_ms + s.duration_ms for s in self.segments) + 60000)
        self.setMinimumWidth(int(max_ms * self.pixels_per_ms)); self.update()
    
    def get_ms_per_beat(self): return (60.0 / self.target_bpm) * 1000.0
    def get_seg_rect(self, seg):
        x = int(seg.start_ms * self.pixels_per_ms); w = int(seg.duration_ms * self.pixels_per_ms); h = int((self.lane_height - 20) * seg.volume)
        y_center = (seg.lane * (self.lane_height + self.lane_spacing)) + (self.lane_height // 2) + 40
        return QRect(x, y_center - (h // 2), w, h)

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing); painter.fillRect(self.rect(), QColor(25, 25, 25))
        painter.setPen(QPen(QColor(45, 45, 45), 1))
        for i in range(5): 
            y = i * (self.lane_height + self.lane_spacing) + 40; painter.fillRect(0, y, self.width(), self.lane_height, QColor(32, 32, 32))
            painter.setPen(QColor(100, 100, 100)); painter.drawText(5, y + 15, f"LANE {i+1}")
        
        mpb = self.get_ms_per_beat(); mpbar = mpb * 4
        for i in range(0, 3600000, int(mpb)):
            x = int(i * self.pixels_per_ms); 
            if x > self.width(): break
            if (i % int(mpbar)) < 10:
                painter.setPen(QPen(QColor(80, 80, 80), 1)); painter.drawLine(x, 0, x, self.height())
                painter.setPen(QColor(150, 150, 150)); painter.drawText(x + 5, 25, f"BAR {int(i // mpbar) + 1}")
            else: painter.setPen(QPen(QColor(50, 50, 50), 1, Qt.PenStyle.DotLine)); painter.drawLine(x, 40, x, self.height())

        for seg in self.segments:
            rect = self.get_seg_rect(seg); color = QColor(seg.color); color.setAlpha(int(120 + 135 * (min(seg.volume, 1.5) / 1.5)))
            if seg == self.selected_segment: painter.setBrush(QBrush(color.lighter(130))); painter.setPen(QPen(Qt.GlobalColor.white, 3))
            elif seg.is_primary: painter.setBrush(QBrush(color)); painter.setPen(QPen(QColor(255, 215, 0), 3))
            else: painter.setBrush(QBrush(color)); painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawRoundedRect(rect, 6, 6)
            if self.show_waveforms and seg.waveform:
                painter.setPen(QPen(QColor(255, 255, 255, 80), 1)); pts = len(seg.waveform); mid_y = rect.center().y(); max_h = rect.height() // 2
                for i in range(0, rect.width(), 2):
                    idx = int((i / rect.width()) * pts); 
                    if idx < pts: val = seg.waveform[idx] * max_h; painter.drawLine(rect.left() + i, int(mid_y - val), rect.left() + i, int(mid_y + val))
            fi_w = int(seg.fade_in_ms * self.pixels_per_ms); fo_w = int(seg.fade_out_ms * self.pixels_per_ms)
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1, Qt.PenStyle.DashLine)); painter.drawLine(rect.left(), rect.bottom(), rect.left() + fi_w, rect.top()); painter.drawLine(rect.right() - fo_w, rect.top(), rect.right(), rect.bottom())
            painter.setBrush(QBrush(Qt.GlobalColor.white)); painter.setPen(Qt.PenStyle.NoPen); painter.drawEllipse(rect.left() + fi_w - 4, rect.top() - 4, 8, 8); painter.drawEllipse(rect.right() - fo_w - 4, rect.top() - 4, 8, 8)
            painter.setPen(Qt.GlobalColor.white); painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold)); painter.drawText(rect.adjusted(8, 8, -8, -8), Qt.AlignmentFlag.AlignTop, seg.filename)
            by = rect.bottom() - 22; bx = rect.left() + 8
            if seg.is_primary:
                painter.setBrush(QBrush(QColor(255, 215, 0))); painter.setPen(Qt.PenStyle.NoPen); br = QRect(bx, by, 60, 16); painter.drawRoundedRect(br, 4, 4); painter.setPen(Qt.GlobalColor.black); painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold)); painter.drawText(br, Qt.AlignmentFlag.AlignCenter, "PRIMARY"); bx += 65
            if self.show_modifications:
                if abs(seg.bpm - self.target_bpm) > 0.1:
                    painter.setBrush(QBrush(QColor(255, 165, 0))); br = QRect(bx, by, 55, 16); painter.drawRoundedRect(br, 4, 4); painter.setPen(Qt.GlobalColor.black); painter.drawText(br, Qt.AlignmentFlag.AlignCenter, "STRETCH"); bx += 60
                if abs(seg.volume - 1.0) > 0.05:
                    painter.setBrush(QBrush(QColor(0, 200, 255))); br = QRect(bx, by, 40, 16); painter.drawRoundedRect(br, 4, 4); painter.setPen(Qt.GlobalColor.black); painter.drawText(br, Qt.AlignmentFlag.AlignCenter, f"{int(seg.volume*100)}%"); bx += 45
                if seg.pitch_shift != 0:
                    painter.setBrush(QBrush(QColor(200, 100, 255))); br = QRect(bx, by, 40, 16); painter.drawRoundedRect(br, 4, 4); painter.setPen(Qt.GlobalColor.black); painter.drawText(br, Qt.AlignmentFlag.AlignCenter, f"{seg.pitch_shift:+}st")
        cx = int(self.cursor_pos_ms * self.pixels_per_ms); painter.setPen(QPen(QColor(255, 50, 50), 2)); painter.drawLine(cx, 0, cx, self.height()); painter.setBrush(QBrush(QColor(255, 50, 50))); painter.drawPolygon(QPoint(cx - 8, 0), QPoint(cx + 8, 0), QPoint(cx, 12))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            clicked_seg = None
            for seg in reversed(self.segments):
                rect = self.get_seg_rect(seg); fi_x = rect.left() + int(seg.fade_in_ms * self.pixels_per_ms); fo_x = rect.right() - int(seg.fade_out_ms * self.pixels_per_ms)
                if QRect(fi_x-10, rect.top()-10, 20, 20).contains(event.pos()): self.selected_segment = seg; self.fade_in_dragging = True; self.drag_start_pos = event.pos(); self.drag_start_fade = seg.fade_in_ms; self.update(); return
                if QRect(fo_x-10, rect.top()-10, 20, 20).contains(event.pos()): self.selected_segment = seg; self.fade_out_dragging = True; self.drag_start_pos = event.pos(); self.drag_start_fade = seg.fade_out_ms; self.update(); return
                if rect.contains(event.pos()): clicked_seg = seg; break
            self.selected_segment = clicked_seg; self.segmentSelected.emit(clicked_seg)
            if self.selected_segment:
                self.drag_start_pos = event.pos(); self.drag_start_ms = self.selected_segment.start_ms; self.drag_start_dur = self.selected_segment.duration_ms; self.drag_start_vol = self.selected_segment.volume; self.drag_start_lane = self.selected_segment.lane
                rect = self.get_seg_rect(self.selected_segment); 
                if event.pos().x() > (rect.right() - 20): self.resizing = True
                elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier: self.vol_dragging = True
                else: self.dragging = True
            else: self.cursor_pos_ms = event.pos().x() / self.pixels_per_ms
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            target_seg = None
            for seg in reversed(self.segments):
                if self.get_seg_rect(seg).contains(event.pos()): target_seg = seg; break
            menu = QMenu(self)
            if target_seg:
                primary_text = "⭐ Unmark Primary" if target_seg.is_primary else "⭐ Set as Primary"; primary_action = menu.addAction(primary_text); split_action = menu.addAction("✂ Split at Cursor")
                pm = menu.addMenu("🎵 Shift Pitch")
                for i in range(-6, 7): t = f"{i:+} Semitones" if i != 0 else "Original Pitch"; pa = pm.addAction(t); pa.setData(i)
                menu.addSeparator(); da = menu.addAction("🗑 Remove Track"); action = menu.exec(self.mapToGlobal(event.pos()))
                if action == primary_action: self.window().push_undo(); target_seg.is_primary = not target_seg.is_primary
                elif action == split_action: self.window().push_undo(); self.split_segment(target_seg, event.pos().x())
                elif action in pm.actions(): self.window().push_undo(); target_seg.pitch_shift = action.data()
                elif action == da: self.window().push_undo(); self.segments.remove(target_seg); 
                if self.selected_segment == target_seg: self.selected_segment = None
                self.update_geometry(); self.timelineChanged.emit()
            else:
                ba = menu.addAction("🪄 Find Bridge Track here"); action = menu.exec(self.mapToGlobal(event.pos()))
                if action == ba: self.window().find_bridge_for_gap(event.pos().x())

    def mouseMoveEvent(self, event):
        if not self.selected_segment: return
        dx = event.pos().x() - self.drag_start_pos.x(); dy = event.pos().y() - self.drag_start_pos.y(); mpb = self.get_ms_per_beat()
        if self.fade_in_dragging:
            rf = self.drag_start_fade + dx/self.pixels_per_ms
            if self.snap_to_grid: rf = round(rf / mpb) * mpb
            self.selected_segment.fade_in_ms = max(0, min(self.selected_segment.duration_ms/2, rf))
        elif self.fade_out_dragging:
            rf = self.drag_start_fade - dx/self.pixels_per_ms
            if self.snap_to_grid: rf = round(rf / mpb) * mpb
            self.selected_segment.fade_out_ms = max(0, min(self.selected_segment.duration_ms/2, rf))
        elif self.resizing:
            rd = self.drag_start_dur + dx/self.pixels_per_ms
            if self.snap_to_grid: rd = round(rd / mpb) * mpb
            self.selected_segment.duration_ms = max(1000, rd)
        elif self.vol_dragging: self.selected_segment.volume = max(0.0, min(1.5, self.drag_start_vol - dy/150.0))
        elif self.dragging:
            ns = max(0, self.drag_start_ms + dx/self.pixels_per_ms)
            if self.snap_to_grid: ns = round(ns / mpb) * mpb
            for other in self.segments:
                if other == self.selected_segment: continue
                oe = other.start_ms + other.duration_ms
                if abs(ns - oe) < self.snap_threshold_ms: ns = oe
                elif abs(ns - other.start_ms) < self.snap_threshold_ms: ns = other.start_ms
            self.selected_segment.start_ms = ns; nl = max(0, min(4, int((event.pos().y() - 40) // (self.lane_height + self.lane_spacing))))
            self.selected_segment.lane = nl
        self.update_geometry(); self.timelineChanged.emit()

    def mouseReleaseEvent(self, event): self.dragging = self.resizing = self.vol_dragging = self.fade_in_dragging = self.fade_out_dragging = False; self.update_geometry()
    
    def split_segment(self, seg, x_pos):
        sm = x_pos / self.pixels_per_ms; rs = sm - seg.start_ms
        if rs < 500 or rs > (seg.duration_ms - 500): return 
        nd = seg.duration_ms - rs; no = seg.offset_ms + rs; seg.duration_ms = rs 
        td = {'id': seg.id, 'filename': seg.filename, 'file_path': seg.file_path, 'bpm': seg.bpm, 'harmonic_key': seg.key}
        ns = TrackSegment(td, start_ms=sm, duration_ms=nd, lane=seg.lane, offset_ms=no); ns.volume = seg.volume; ns.is_primary = seg.is_primary; ns.waveform = seg.waveform; ns.pitch_shift = seg.pitch_shift
        self.segments.append(ns); self.update_geometry(); self.timelineChanged.emit()

    def add_track(self, track_data, start_ms=None, lane=0):
        if start_ms is None:
            if self.segments: last_seg = max(self.segments, key=lambda s: s.start_ms + s.duration_ms); start_ms = last_seg.start_ms + last_seg.duration_ms - 5000; lane = last_seg.lane
            else: start_ms = 0
        ns = TrackSegment(track_data, start_ms=start_ms, lane=lane); self.segments.append(ns); self.update_geometry(); self.timelineChanged.emit(); return ns

    def dragEnterEvent(self, event):
        if event.mimeData().hasText(): event.acceptProposedAction()
    def dropEvent(self, event):
        tid = event.mimeData().text()
        if tid:
            pos = event.position(); lane = max(0, min(4, int((pos.y() - 40) // (self.lane_height + self.lane_spacing))))
            self.window().add_track_by_id(int(tid), pos.x(), lane=lane)
            event.acceptProposedAction()

class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents); self.message = "Processing Journey..."; self.hide()
    def paintEvent(self, event):
        p = QPainter(self); p.fillRect(self.rect(), QColor(0, 0, 0, 180)); p.setPen(Qt.GlobalColor.white); p.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold)); p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
    def show_loading(self, message="Processing..."): self.message = message; self.setGeometry(self.parent().rect()); self.raise_(); self.show(); QApplication.processEvents()
    def hide_loading(self): self.hide()

class AudioSequencerApp(QMainWindow):
    def __init__(self):
        super().__init__(); self.dm = DataManager(); self.scorer = CompatibilityScorer(); self.processor = AudioProcessor(); self.renderer = FlowRenderer(); self.generator = TransitionGenerator(); self.orchestrator = FullMixOrchestrator(); self.undo_manager = UndoManager(); self.selected_library_track = None; 
        self.play_timer = QTimer(); self.play_timer.setInterval(20); self.play_timer.timeout.connect(self.update_playback_cursor); self.is_playing = False
        self.init_ui(); self.load_library(); self.loading_overlay = LoadingOverlay(self.centralWidget())
    
    def update_playback_cursor(self):
        self.timeline_widget.cursor_pos_ms += 20; self.timeline_widget.update()
    
    def toggle_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing: self.play_timer.start(); self.ptb.setText("⏹ Stop Preview")
        else: self.play_timer.stop(); self.ptb.setText("▶ Play Journey")

    def push_undo(self): self.undo_manager.push_state(self.timeline_widget.segments)
    def undo(self): 
        ns = self.undo_manager.undo(self.timeline_widget.segments)
        if ns: self.apply_state(ns)
    def redo(self):
        ns = self.undo_manager.redo(self.timeline_widget.segments)
        if ns: self.apply_state(ns)
    def apply_state(self, state_list):
        self.timeline_widget.segments = []
        for sj in state_list:
            s = json.loads(sj); td = {'id': s['id'], 'filename': s['filename'], 'file_path': s['file_path'], 'bpm': s['bpm'], 'harmonic_key': s['key']}
            seg = TrackSegment(td, start_ms=s['start_ms'], duration_ms=s['duration_ms'], lane=s['lane'], offset_ms=s['offset_ms'])
            seg.volume = s['volume']; seg.is_primary = s['is_primary']; seg.fade_in_ms = s['fade_in_ms']; seg.fade_out_ms = s['fade_out_ms']; seg.pitch_shift = s.get('pitch_shift', 0)
            seg.waveform = self.processor.get_waveform_envelope(seg.file_path); self.timeline_widget.segments.append(seg)
        self.timeline_widget.update_geometry(); self.update_status()

    def init_ui(self):
        self.setWindowTitle("AudioSequencer AI - The Pro Flow"); self.setMinimumSize(QSize(1400, 950))
        cw = QWidget(); self.setCentralWidget(cw); ml = QVBoxLayout(cw); tp = QHBoxLayout()
        lp = QFrame(); lp.setFixedWidth(450); ll = QVBoxLayout(lp); ll.addWidget(QLabel("<h2>📁 Audio Library</h2>"))
        la = QHBoxLayout(); self.scan_btn = QPushButton("📂 Scan Folder"); self.scan_btn.clicked.connect(self.scan_folder); la.addWidget(self.scan_btn)
        self.embed_btn = QPushButton("🧠 AI Index"); self.embed_btn.clicked.connect(self.run_embedding); la.addWidget(self.embed_btn); ll.addLayout(la)
        sl = QHBoxLayout(); self.search_bar = QLineEdit(); self.search_bar.setPlaceholderText("🔍 Semantic Search..."); self.search_bar.textChanged.connect(self.on_search_text_changed); self.search_bar.returnPressed.connect(self.trigger_semantic_search); sl.addWidget(self.search_bar)
        rsb = QPushButton("↺"); rsb.setFixedWidth(30); rsb.clicked.connect(self.load_library); sl.addWidget(rsb); ll.addLayout(sl)
        self.library_table = DraggableTable(0, 3); self.library_table.setHorizontalHeaderLabels(["Track Name", "BPM", "Key"]); self.library_table.setColumnWidth(0, 250); self.library_table.itemSelectionChanged.connect(self.on_library_track_selected); ll.addWidget(self.library_table); tp.addWidget(lp)
        mp = QFrame(); mp.setFixedWidth(250); mlayout = QVBoxLayout(mp); ag = QFrame(); ag.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 5px;"); al = QVBoxLayout(ag); al.addWidget(QLabel("<h3>📊 Analytics Board</h3>"))
        self.mod_toggle = QPushButton("🔍 Hide Markers"); self.mod_toggle.setCheckable(True); self.mod_toggle.clicked.connect(self.toggle_analytics); al.addWidget(self.mod_toggle)
        self.grid_toggle = QPushButton("📏 Grid Snap: ON"); self.grid_toggle.setCheckable(True); self.grid_toggle.setChecked(True); self.grid_toggle.clicked.connect(self.toggle_grid); al.addWidget(self.grid_toggle)
        ur = QHBoxLayout(); self.ub = QPushButton("↶ Undo"); self.ub.clicked.connect(self.undo); ur.addWidget(self.ub); self.rb = QPushButton("↷ Redo"); self.rb.clicked.connect(self.redo); ur.addWidget(self.rb); al.addLayout(ur)
        self.stats_label = QLabel("Timeline empty"); self.stats_label.setStyleSheet("color: #aaa; font-size: 11px;"); al.addWidget(self.stats_label)
        save_btn = QPushButton("💾 Save Journey"); save_btn.clicked.connect(self.save_project); al.addWidget(save_btn); load_btn = QPushButton("📂 Load Journey"); load_btn.clicked.connect(self.load_project); al.addWidget(load_btn); mlayout.addWidget(ag); mlayout.addSpacing(10)
        mlayout.addWidget(QLabel("<h3>⚡ Actions</h3>")); self.atb = QPushButton("➕ Add to Timeline"); self.atb.clicked.connect(self.add_selected_to_timeline); mlayout.addWidget(self.atb)
        self.pb = QPushButton("▶ Preview"); self.pb.clicked.connect(self.play_selected); mlayout.addWidget(self.pb); mlayout.addStretch(); hb = QLabel("<b>💡 Pro Tips:</b><br><br>• <b>Drag circles</b> at top to adjust Fades.<br>• <b>Shift + Drag</b> for Volume.<br>• <b>Right-Click</b> to Split."); hb.setStyleSheet("color: #888; font-size: 11px; background: #1a1a1a; padding: 10px;"); mlayout.addWidget(hb); tp.addWidget(mp)
        rp = QFrame(); rp.setFixedWidth(450); rl = QVBoxLayout(rp); rl.addWidget(QLabel("<h3>✨ Smart Suggestions</h3>")); self.rec_list = DraggableTable(0, 2); self.rec_list.setHorizontalHeaderLabels(["Match %", "Track"]); self.rec_list.itemDoubleClicked.connect(self.on_rec_double_clicked); rl.addWidget(self.rec_list); tp.addWidget(rp); ml.addLayout(tp, stretch=1)
        th = QHBoxLayout(); th.addWidget(QLabel("<h2>🎞 Timeline Journey</h2>"))
        self.ptb = QPushButton("▶ Play Journey"); self.ptb.setFixedWidth(120); self.ptb.clicked.connect(self.toggle_playback); th.addWidget(self.ptb)
        th.addSpacing(20); th.addWidget(QLabel("Zoom:")); self.zs = QSlider(Qt.Orientation.Horizontal); self.zs.setRange(10, 200); self.zs.setValue(50); self.zs.setFixedWidth(150); self.zs.valueChanged.connect(self.on_zoom_changed); th.addWidget(self.zs); th.addStretch()
        self.agb = QPushButton("🪄 Auto-Generate Path"); self.agb.clicked.connect(self.auto_populate_timeline); th.addWidget(self.agb); self.cb = QPushButton("🗑 Clear"); self.cb.clicked.connect(self.clear_timeline); th.addWidget(self.cb)
        th.addWidget(QLabel("Target BPM:")); self.tbe = QLineEdit("124"); self.tbe.setFixedWidth(60); self.tbe.textChanged.connect(self.on_bpm_changed); th.addWidget(self.tbe)
        self.render_btn = QPushButton("🚀 RENDER FINAL MIX"); self.render_btn.setStyleSheet("background-color: #007acc; padding: 12px 25px; color: white; font-weight: bold;"); self.render_btn.clicked.connect(self.render_timeline); th.addWidget(self.render_btn); ml.addLayout(th)
        t_scroll = QScrollArea(); t_scroll.setWidgetResizable(True); self.timeline_widget = TimelineWidget(); t_scroll.setWidget(self.timeline_widget); ml.addWidget(t_scroll, stretch=1)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.status_bar.showMessage("Ready.")
        self.timeline_widget.segmentSelected.connect(self.on_segment_selected); self.timeline_widget.timelineChanged.connect(self.update_status)
        self.setStyleSheet("QMainWindow { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI'; } QLabel { color: #ffffff; } QTableWidget { background-color: #1e1e1e; gridline-color: #333; } QPushButton { background-color: #333; color: #fff; padding: 8px; border-radius: 4px; }")

    def find_bridge_for_gap(self, x_pos):
        gap_ms = x_pos / self.timeline_widget.pixels_per_ms; prev_seg = next_seg = None; sorted_segs = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        for s in sorted_segs:
            if s.start_ms + s.duration_ms <= gap_ms: prev_seg = s
            elif s.start_ms >= gap_ms:
                if next_seg is None: next_seg = s
        if not prev_seg or not next_seg: self.status_bar.showMessage("Need a track before AND after the gap to find a bridge."); return
        self.loading_overlay.show_loading(f"AI: Finding bridge..."); 
        try:
            conn = self.dm.get_conn(); conn.row_factory = sqlite3_factory; cursor = conn.cursor(); cursor.execute("SELECT * FROM tracks WHERE id NOT IN (?, ?)", (prev_seg.id, next_seg.id)); candidates = cursor.fetchall()
            results = []
            for c in candidates:
                cd = dict(c); ce = self.dm.get_embedding(cd['clp_embedding_id']) if cd['clp_embedding_id'] else None
                score = self.scorer.calculate_bridge_score(prev_seg.__dict__, next_seg.__dict__, cd, c_emb=ce); results.append((score, cd))
            results.sort(key=lambda x: x[0], reverse=True); self.rec_list.setRowCount(0)
            for sc, ot in results[:15]:
                ri = self.rec_list.rowCount(); self.rec_list.insertRow(ri); si = QTableWidgetItem(f"{sc}% (BRIDGE)"); si.setData(Qt.ItemDataRole.UserRole, ot['id']); self.rec_list.setItem(ri, 0, si); self.rec_list.setItem(ri, 1, QTableWidgetItem(ot['filename']))
            self.loading_overlay.hide_loading(); self.status_bar.showMessage(f"AI found {len(results)} potential bridges."); conn.close()
        except Exception as e: self.loading_overlay.hide_loading(); show_error(self, "Bridge Error", "AI Bridge search failed.", e)

    def on_zoom_changed(self, value): self.timeline_widget.pixels_per_ms = value / 1000.0; self.timeline_widget.update_geometry()
    def clear_timeline(self):
        if QMessageBox.question(self, "Clear", "Clear journey?") == QMessageBox.StandardButton.Yes: self.push_undo(); self.timeline_widget.segments = []; self.timeline_widget.update_geometry(); self.update_status()
    def on_search_text_changed(self, text):
        if not text:
            for r in range(self.library_table.rowCount()): self.library_table.setRowHidden(r, False)
            return
        q = text.lower(); 
        for r in range(self.library_table.rowCount()): self.library_table.setRowHidden(r, q not in self.library_table.item(r, 0).text().lower())
    def trigger_semantic_search(self):
        q = self.search_bar.text()
        if len(q) < 3: return
        self.loading_overlay.show_loading(f"AI Search: '{q}'..."); self.st = SearchThread(q, self.dm); self.st.resultsFound.connect(self.on_semantic_results); self.st.errorOccurred.connect(self.on_search_error); self.st.start()
    def on_semantic_results(self, res):
        self.loading_overlay.hide_loading(); self.library_table.setRowCount(0)
        for r in res:
            ri = self.library_table.rowCount(); self.library_table.insertRow(ri); match = int(max(0, 1.0 - r.get('distance', 1.0)) * 100)
            ni = QTableWidgetItem(r['filename']); ni.setData(Qt.ItemDataRole.UserRole, r['id'])
            if match > 70: ni.setForeground(QBrush(QColor(0, 255, 200)))
            self.library_table.setItem(ri, 0, ni); self.library_table.setItem(ri, 1, QTableWidgetItem(f"{r['bpm']:.1f}")); self.library_table.setItem(ri, 2, QTableWidgetItem(r['harmonic_key']))
    def on_search_error(self, e): self.loading_overlay.hide_loading(); QMessageBox.warning(self, "AI Error", e)
    def save_project(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save Journey", "", "JSON Files (*.json)")
        if p:
            data = {'target_bpm': self.timeline_widget.target_bpm, 'segments': [s.to_dict() for s in self.timeline_widget.segments]}
            with open(p, 'w') as f: json.dump(data, f)
    def load_project(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Journey", "", "JSON Files (*.json)")
        if p:
            self.push_undo(); 
            with open(p, 'r') as f: data = json.load(f)
            self.timeline_widget.segments = []; self.tbe.setText(str(data['target_bpm']))
            for s in data['segments']:
                td = {'id': s['id'], 'filename': s['filename'], 'file_path': s['file_path'], 'bpm': s['bpm'], 'harmonic_key': s['key']}
                seg = TrackSegment(td, start_ms=s['start_ms'], duration_ms=s['duration_ms'], lane=s['lane'], offset_ms=s['offset_ms'])
                seg.volume = s['volume']; seg.is_primary = s['is_primary']; seg.fade_in_ms = s['fade_in_ms']; seg.fade_out_ms = s['fade_out_ms']; seg.pitch_shift = s.get('pitch_shift', 0); seg.waveform = self.processor.get_waveform_envelope(seg.file_path); self.timeline_widget.segments.append(seg)
            self.timeline_widget.update_geometry(); self.update_status()
    def on_bpm_changed(self, t):
        try: self.timeline_widget.target_bpm = float(t); self.timeline_widget.update(); self.update_status()
        except: pass
    def toggle_analytics(self): self.timeline_widget.show_modifications = not self.mod_toggle.isChecked(); self.mod_toggle.setText("🔍 Show Markers" if self.mod_toggle.isChecked() else "🔍 Hide Markers"); self.timeline_widget.update()
    def toggle_grid(self): self.timeline_widget.snap_to_grid = self.grid_toggle.isChecked(); self.grid_toggle.setText(f"📏 Grid Snap: {'ON' if self.timeline_widget.snap_to_grid else 'OFF'}"); self.timeline_widget.update()
    def update_status(self):
        count = len(self.timeline_widget.segments)
        if count > 0:
            tdur = max(s.start_ms + s.duration_ms for s in self.timeline_widget.segments); abpm = sum(s.bpm for s in self.timeline_widget.segments) / count; bdiff = abs(abpm - self.timeline_widget.target_bpm); self.status_bar.showMessage(f"Timeline: {count} tracks | {tdur/1000:.1f}s")
            at = (f"<b>Tracks:</b> {count}<br><b>Duration:</b> {tdur/1000:.1f}s<br><b>Avg Track BPM:</b> {abpm:.1f}<br><b>BPM Variance:</b> {bdiff:.1f}<br>")
            if self.timeline_widget.selected_segment:
                sel = self.timeline_widget.selected_segment; at += f"<br><b>Selected Key:</b> {sel.key}"
                for o in self.timeline_widget.segments:
                    if o == sel: continue
                    if max(sel.start_ms, o.start_ms) < min(sel.start_ms + sel.duration_ms, o.start_ms + o.duration_ms):
                        hs = self.scorer.calculate_harmonic_score(sel.key, o.key); color = "#00ff66" if hs >= 100 else "#ccff00" if hs >= 80 else "#ff5555"; at += f"<br>Overlap with '{o.filename[:10]}...': <span style='color: {color};'>{hs}% Match</span>"
            self.stats_label.setText(at)
        else: self.status_bar.showMessage("Ready."); self.stats_label.setText("Timeline empty")
    def load_library(self):
        try:
            conn = self.dm.get_conn(); cursor = conn.cursor(); cursor.execute("SELECT id, filename, bpm, harmonic_key FROM tracks"); rows = cursor.fetchall(); self.library_table.setRowCount(0)
            for r in rows:
                ri = self.library_table.rowCount(); self.library_table.insertRow(ri); ni = QTableWidgetItem(r[1]); ni.setData(Qt.ItemDataRole.UserRole, row[0]); self.library_table.setItem(ri, 0, ni); self.library_table.setItem(ri, 1, QTableWidgetItem(f"{r[2]:.1f}")); self.library_table.setItem(ri, 2, QTableWidgetItem(r[3]))
            conn.close()
        except Exception as e: show_error(self, "Library Error", "Failed to load library.", e)
    def on_library_track_selected(self):
        si = self.library_table.selectedItems()
        if si: self.add_track_by_id(self.library_table.item(si[0].row(), 0).data(Qt.ItemDataRole.UserRole), only_update_recs=True)
    def add_track_by_id(self, tid, x=None, only_update_recs=False, lane=0):
        try:
            conn = self.dm.get_conn(); conn.row_factory = sqlite3_factory; cursor = conn.cursor(); cursor.execute("SELECT * FROM tracks WHERE id = ?", (tid,)); track = dict(cursor.fetchone()); conn.close()
            if not only_update_recs:
                self.push_undo(); sm = x / self.timeline_widget.pixels_per_ms if x is not None else None; seg = self.timeline_widget.add_track(track, start_ms=sm); 
                if x is not None: seg.lane = lane
                seg.waveform = self.processor.get_waveform_envelope(track['file_path']); self.timeline_widget.update()
            self.selected_library_track = track; self.update_recommendations(tid)
        except Exception as e: show_error(self, "Data Error", "Failed to retrieve track.", e)
    def add_selected_to_timeline(self):
        if self.selected_library_track: self.add_track_by_id(self.selected_library_track['id'])
    def on_rec_double_clicked(self, i): self.add_track_by_id(self.rec_list.item(i.row(), 0).data(Qt.ItemDataRole.UserRole))
    def auto_populate_timeline(self):
        if not self.selected_library_track: return
        self.push_undo(); self.loading_overlay.show_loading(); seq = self.orchestrator.find_curated_sequence(max_tracks=6, seed_track=self.selected_library_track)
        if sequence:
            self.timeline_widget.segments = []; cm = 0
            for i, t in enumerate(seq):
                d = 20000 if i % 2 == 0 else 30000; seg = self.timeline_widget.add_track(t, start_ms=cm); seg.waveform = self.processor.get_waveform_envelope(t['file_path']); cm += d - 8000 
            self.timeline_widget.update_geometry()
        self.loading_overlay.hide_loading()
    def on_segment_selected(self, s): self.update_status()
    def render_timeline(self):
        if not self.timeline_widget.segments: return
        ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        try: tbpm = float(self.tbe.text())
        except: tbpm = 124.0
        self.loading_overlay.show_loading("Rendering..."); 
        try:
            out = "timeline_mix.mp3"; rd = [s.to_dict() for s in ss]
            self.renderer.render_timeline(rd, out, target_bpm=tbpm); self.loading_overlay.hide_loading(); QMessageBox.information(self, "Success", f"Mix rendered: {out}"); os.startfile(out)
        except Exception as e: self.loading_overlay.hide_loading(); show_error(self, "Render Error", "Failed to render.", e)
    def scan_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select Folder")
        if f:
            self.loading_overlay.show_loading("Scanning..."); 
            try: from src.ingestion import IngestionEngine; e = IngestionEngine(db_path=self.dm.db_path); e.scan_directory(f); self.load_library(); self.loading_overlay.hide_loading()
            except Exception as e: self.loading_overlay.hide_loading(); show_error(self, "Scan Error", "Failed to scan.", e)
    def run_embedding(self):
        self.loading_overlay.show_loading("AI Indexing..."); 
        try:
            from src.embeddings import EmbeddingEngine; ee = EmbeddingEngine(); conn = self.dm.get_conn(); cursor = conn.cursor(); cursor.execute("SELECT id, file_path, clp_embedding_id FROM tracks"); tracks = cursor.fetchall()
            for tid, fp, ex in tracks:
                if not ex: eb = ee.get_embedding(fp); self.dm.add_embedding(tid, eb, metadata={"file_path": fp})
            conn.close(); self.loading_overlay.hide_loading(); QMessageBox.information(self, "AI Complete", "Indexed!")
        except Exception as e: self.loading_overlay.hide_loading(); show_error(self, "AI Error", "Indexing failed.", e)
    def update_recommendations(self, tid):
        try:
            conn = self.dm.get_conn(); conn.row_factory = sqlite3_factory; cursor = conn.cursor(); cursor.execute("SELECT * FROM tracks WHERE id = ?",(tid,)); target = dict(cursor.fetchone()); te = self.dm.get_embedding(target['clp_embedding_id']) if target['clp_embedding_id'] else None
            cursor.execute("SELECT * FROM tracks WHERE id != ?", (tid,)); others = cursor.fetchall(); results = []
            for o in others:
                od = dict(o); oe = self.dm.get_embedding(od['clp_embedding_id']) if od['clp_embedding_id'] else None; sd = self.scorer.get_total_score(target, od, te, oe); results.append((sd, od))
            results.sort(key=lambda x: x[0]['total'], reverse=True); self.rec_list.setRowCount(0)
            for sc, ot in results[:15]:
                ri = self.rec_list.rowCount(); self.rec_list.insertRow(ri); si = QTableWidgetItem(f"{sc['total']}%"); si.setData(Qt.ItemDataRole.UserRole, ot['id']); si.setToolTip(f"BPM: {sc['bpm_score']}% | Har: {sc['harmonic_score']}% | Sem: {sc['semantic_score']}%"); self.rec_list.setItem(ri, 0, si); ni = QTableWidgetItem(ot['filename'])
                if sc['harmonic_score'] >= 100: ni.setForeground(QBrush(QColor(0, 255, 100)))
                elif sc['harmonic_score'] >= 80: ni.setForeground(QBrush(QColor(200, 255, 0)))
                self.rec_list.setItem(ri, 1, ni)
            conn.close()
        except Exception as e: print(f"Rec Engine Error: {e}")
    def play_selected(self):
        if self.selected_library_track:
            try: os.startfile(self.selected_library_track['file_path'])
            except Exception as e: show_error(self, "Playback Error", "Failed to play.", e)

def sqlite3_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description): d[col[0]] = row[idx]
    return d

if __name__ == "__main__":
    app = QApplication(sys.argv); window = AudioSequencerApp(); window.show(); sys.exit(app.exec())
