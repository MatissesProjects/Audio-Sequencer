import sys
import os
import sqlite3
import traceback
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTableWidget, QTableWidgetItem, 
                             QLineEdit, QLabel, QPushButton, QFrame, QMessageBox,
                             QScrollArea, QMenu, QDialog, QTextEdit, QStatusBar, QFileDialog,
                             QSlider, QComboBox, QCheckBox, QHeaderView)
from PyQt6.QtCore import Qt, QSize, QRect, pyqtSignal, QPoint, QMimeData, QThread, QTimer, QUrl
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont, QDrag
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

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
        self.setWindowTitle(title); self.setMinimumSize(600, 400)
        layout = QVBoxLayout(self); msg_layout = QHBoxLayout()
        icon_label = QLabel("❌"); icon_label.setStyleSheet("font-size: 32px;"); msg_layout.addWidget(icon_label)
        msg_label = QLabel(message); msg_label.setWordWrap(True); msg_label.setStyleSheet("font-size: 14px; font-weight: bold; color: white;"); msg_layout.addWidget(msg_label, stretch=1)
        layout.addLayout(msg_layout); l = QLabel("Technical Details:"); l.setStyleSheet("color: #aaa;"); layout.addWidget(l)
        self.details_box = QTextEdit(); self.details_box.setReadOnly(True); self.details_box.setText(details)
        self.details_box.setStyleSheet("background-color: #1a1a1a; color: #ff5555; font-family: Consolas, monospace; border: 1px solid #333;"); layout.addWidget(self.details_box)
        btn_layout = QHBoxLayout()
        copy_btn = QPushButton("📋 Copy to Clipboard"); copy_btn.clicked.connect(self.copy_to_clipboard); btn_layout.addWidget(copy_btn)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(self.accept); close_btn.setDefault(True); btn_layout.addWidget(close_btn); layout.addLayout(btn_layout)
        self.setStyleSheet("QDialog { background-color: #252525; } QPushButton { background-color: #444; color: white; padding: 8px; border-radius: 4px; }")
    def copy_to_clipboard(self): QApplication.clipboard().setText(self.details_box.toPlainText()); QMessageBox.information(self, "Copied", "Error details copied to clipboard.")

def show_error(parent, title, message, exception):
    details = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    dialog = DetailedErrorDialog(title, message, details, parent); dialog.exec()

class SearchThread(QThread):
    resultsFound = pyqtSignal(list); errorOccurred = pyqtSignal(str)
    def __init__(self, query, dm): super().__init__(); self.query = query; self.dm = dm
    def run(self):
        try:
            engine = EmbeddingEngine(); text_emb = engine.get_text_embedding(self.query)
            results = self.dm.search_embeddings(text_emb, n_results=20); self.resultsFound.emit(results)
        except Exception as e: self.errorOccurred.emit(str(e))

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
        self.id = track_data['id']; self.filename = track_data['filename']; self.file_path = track_data['file_path']
        self.bpm = track_data['bpm']; self.key = track_data['harmonic_key']
        self.start_ms = start_ms; self.duration_ms = duration_ms; self.offset_ms = offset_ms
        self.volume = 1.0; self.lane = lane; self.is_primary = False; self.waveform = []
        self.fade_in_ms = 2000; self.fade_out_ms = 2000; self.pitch_shift = 0
        base_color = self.KEY_COLORS.get(self.key, QColor(70, 130, 180))
        self.color = QColor(base_color.red(), base_color.green(), base_color.blue(), 200)
        self.onsets = []
        if 'onsets_json' in track_data and track_data['onsets_json']:
            try: self.onsets = [float(x) * 1000.0 for x in track_data['onsets_json'].split(',')]
            except: pass
    def to_dict(self):
        d = {'id': self.id, 'filename': self.filename, 'file_path': self.file_path, 'bpm': self.bpm, 'key': self.key, 'start_ms': self.start_ms, 'duration_ms': self.duration_ms, 'offset_ms': self.offset_ms, 'volume': self.volume, 'lane': self.lane, 'is_primary': self.is_primary, 'fade_in_ms': self.fade_in_ms, 'fade_out_ms': self.fade_out_ms, 'pitch_shift': self.pitch_shift}
        d['onsets_json'] = ",".join([str(x/1000.0) for x in self.onsets]); return d

class UndoManager:
    def __init__(self): self.undo_stack = []; self.redo_stack = []
    def push_state(self, segments):
        state = [json.dumps(s.to_dict()) for s in segments]; self.undo_stack.append(state); self.redo_stack.clear()
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
    segmentSelected = pyqtSignal(object); timelineChanged = pyqtSignal()
    def __init__(self):
        super().__init__(); self.segments = []; self.setMinimumHeight(550); self.setAcceptDrops(True); self.pixels_per_ms = 0.05
        self.selected_segment = None; self.dragging = self.resizing = self.vol_dragging = self.fade_in_dragging = self.fade_out_dragging = self.slipping = self.setting_loop = False
        self.drag_start_pos = None; self.drag_start_ms = self.drag_start_dur = self.drag_start_fade = self.drag_start_offset = 0; self.drag_start_vol = 1.0; self.drag_start_lane = 0
        self.lane_height = 120; self.lane_spacing = 10; self.snap_threshold_ms = 2000; self.target_bpm = 124.0
        self.show_modifications = True; self.cursor_pos_ms = 0; self.show_waveforms = True; self.snap_to_grid = True
        self.mutes = [False] * 5; self.solos = [False] * 5; self.loop_start_ms = 0; self.loop_end_ms = 30000; self.loop_enabled = False
        self.scorer = CompatibilityScorer()
        self.update_geometry()
    def update_geometry(self):
        max_ms = 600000
        if self.segments: max_ms = max(max_ms, max(s.start_ms + s.duration_ms for s in self.segments) + 60000)
        self.setMinimumWidth(int(max_ms * self.pixels_per_ms)); self.update()
    def get_ms_per_beat(self): return (60.0 / self.target_bpm) * 1000.0
    def get_seg_rect(self, seg):
        x = int(seg.start_ms * self.pixels_per_ms); w = int(seg.duration_ms * self.pixels_per_ms); h = 100
        y_center = (seg.lane * (self.lane_height + self.lane_spacing)) + (self.lane_height // 2) + 40
        return QRect(x, y_center - (h // 2), w, h)
    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing); painter.fillRect(self.rect(), QColor(25, 25, 25))
        if self.loop_enabled:
            lx = int(self.loop_start_ms * self.pixels_per_ms); lw = int((self.loop_end_ms - self.loop_start_ms) * self.pixels_per_ms)
            painter.fillRect(lx, 0, lw, 40, QColor(0, 200, 255, 60))
            painter.setPen(QPen(QColor(0, 200, 255, 150), 2)); painter.drawLine(lx, 0, lx, 40); painter.drawLine(lx + lw, 0, lx + lw, 40)
        any_solo = any(self.solos)
        for i in range(5): 
            y = i * (self.lane_height + self.lane_spacing) + 40; bg = QColor(32, 32, 32)
            if self.solos[i]: bg = QColor(45, 45, 32)
            elif self.mutes[i] or (any_solo and not self.solos[i]): bg = QColor(20, 20, 20)
            painter.fillRect(0, y, self.width(), self.lane_height, bg); painter.setPen(QColor(150, 150, 150)); painter.drawText(5, y + 15, f"LANE {i+1}")
            mr = QRect(5, y + 25, 20, 20); painter.setBrush(QBrush(QColor(255, 50, 50) if self.mutes[i] else QColor(60, 60, 60))); painter.setPen(Qt.PenStyle.NoPen); painter.drawRoundedRect(mr, 3, 3); painter.setPen(Qt.GlobalColor.white); painter.drawText(mr, Qt.AlignmentFlag.AlignCenter, "M")
            sr = QRect(30, y + 25, 20, 20); painter.setBrush(QBrush(QColor(255, 200, 0) if self.solos[i] else QColor(60, 60, 60))); painter.setPen(Qt.PenStyle.NoPen); painter.drawRoundedRect(sr, 3, 3); painter.setPen(Qt.GlobalColor.white); painter.drawText(sr, Qt.AlignmentFlag.AlignCenter, "S")
        mpb = self.get_ms_per_beat(); mpbar = mpb * 4
        for i in range(0, 3600000, int(mpb)):
            x = int(i * self.pixels_per_ms); 
            if x > self.width(): break
            if (i % int(mpbar)) < 10:
                painter.setPen(QPen(QColor(80, 80, 80), 1)); painter.drawLine(x, 0, x, self.height()); painter.setPen(QColor(150, 150, 150)); painter.drawText(x + 5, 25, f"BAR {int(i // mpbar) + 1}")
            else: painter.setPen(QPen(QColor(50, 50, 50), 1, Qt.PenStyle.DotLine)); painter.drawLine(x, 40, x, self.height())
        for seg in self.segments:
            rect = self.get_seg_rect(seg); color = QColor(seg.color); is_ducked = False
            if not seg.is_primary:
                for o in self.segments:
                    if o != seg and o.is_primary and max(seg.start_ms, o.start_ms) < min(seg.start_ms + seg.duration_ms, o.start_ms + o.duration_ms): is_ducked = True; break
            hc = False
            for o in self.segments:
                if o != seg and max(seg.start_ms, o.start_ms) < min(seg.start_ms + seg.duration_ms, o.start_ms + o.duration_ms):
                    if self.scorer.calculate_harmonic_score(seg.key, o.key) < 60: hc = True; break
            dv = seg.volume * 0.63 if is_ducked else seg.volume; color.setAlpha(int(120 + 135 * (min(dv, 1.5) / 1.5)))
            if seg == self.selected_segment: painter.setBrush(QBrush(color.lighter(130))); painter.setPen(QPen(Qt.GlobalColor.white, 3))
            elif seg.is_primary: painter.setBrush(QBrush(color)); painter.setPen(QPen(QColor(255, 215, 0), 3))
            elif hc: painter.setBrush(QBrush(color)); painter.setPen(QPen(QColor(255, 50, 50), 3))
            else: painter.setBrush(QBrush(color)); painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawRoundedRect(rect, 6, 6)
            if self.show_waveforms and seg.waveform:
                painter.setPen(QPen(QColor(255, 255, 255, 80), 1)); pts = len(seg.waveform); mid_y = rect.center().y(); max_h = rect.height() // 2
                for i in range(0, rect.width(), 2):
                    ri = (i / rect.width()) * (seg.duration_ms / 30000.0); idx = int((ri + (seg.offset_ms / 30000.0)) * pts) % pts; val = seg.waveform[idx] * max_h; painter.drawLine(rect.left() + i, int(mid_y - val), rect.left() + i, int(mid_y + val))
            painter.setPen(QPen(QColor(255, 255, 255, 180), 2)); vy = rect.bottom() - int(rect.height() * (dv / 1.5)); painter.drawLine(rect.left(), vy, rect.right(), vy)
            if seg.onsets:
                painter.setPen(QPen(QColor(255, 255, 255, 120), 1)); s_f = self.target_bpm / seg.bpm
                for o_ms in seg.onsets:
                    adj = (o_ms - seg.offset_ms) * s_f
                    if 0 <= adj <= seg.duration_ms: tx = rect.left() + int(adj * self.pixels_per_ms); painter.drawLine(tx, rect.top() + 5, tx, rect.bottom() - 5)
            fi_w = int(seg.fade_in_ms * self.pixels_per_ms); fo_w = int(seg.fade_out_ms * self.pixels_per_ms)
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1, Qt.PenStyle.DashLine)); painter.drawLine(rect.left(), rect.bottom(), rect.left() + fi_w, rect.top()); painter.drawLine(rect.right() - fo_w, rect.top(), rect.right(), rect.bottom())
            painter.setBrush(QBrush(Qt.GlobalColor.white)); painter.setPen(Qt.PenStyle.NoPen); painter.drawEllipse(rect.left() + fi_w - 4, rect.top() - 4, 8, 8); painter.drawEllipse(rect.right() - fo_w - 4, rect.top() - 4, 8, 8)
            painter.setPen(Qt.GlobalColor.white); painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold)); painter.drawText(rect.adjusted(8, 8, -8, -8), Qt.AlignmentFlag.AlignTop, seg.filename)
            by = rect.bottom() - 22; bx = rect.left() + 8
            if seg.is_primary: painter.setBrush(QBrush(QColor(255, 215, 0))); painter.setPen(Qt.PenStyle.NoPen); br = QRect(bx, by, 60, 16); painter.drawRoundedRect(br, 4, 4); painter.setPen(Qt.GlobalColor.black); painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold)); painter.drawText(br, Qt.AlignmentFlag.AlignCenter, "PRIMARY"); bx += 65
            if self.show_modifications:
                if abs(seg.bpm - self.target_bpm) > 0.1: painter.setBrush(QBrush(QColor(255, 165, 0))); br = QRect(bx, by, 55, 16); painter.drawRoundedRect(br, 4, 4); painter.setPen(Qt.GlobalColor.black); painter.drawText(br, Qt.AlignmentFlag.AlignCenter, "STRETCH"); bx += 60
                if abs(seg.volume - 1.0) > 0.05: painter.setBrush(QBrush(QColor(0, 200, 255))); br = QRect(bx, by, 40, 16); painter.drawRoundedRect(br, 4, 4); painter.setPen(Qt.GlobalColor.black); painter.drawText(br, Qt.AlignmentFlag.AlignCenter, f"{int(seg.volume*100)}%"); bx += 45
                if seg.pitch_shift != 0: painter.setBrush(QBrush(QColor(200, 100, 255))); br = QRect(bx, by, 40, 16); painter.drawRoundedRect(br, 4, 4); painter.setPen(Qt.GlobalColor.black); painter.drawText(br, Qt.AlignmentFlag.AlignCenter, f"{seg.pitch_shift:+}st")
        cx = int(self.cursor_pos_ms * self.pixels_per_ms); painter.setPen(QPen(QColor(255, 50, 50), 2)); painter.drawLine(cx, 0, cx, self.height()); painter.setBrush(QBrush(QColor(255, 50, 50))); painter.drawPolygon(QPoint(cx - 8, 0), QPoint(cx + 8, 0), QPoint(cx, 12))

    def mousePressEvent(self, event):
        for i in range(5):
            y = i * (self.lane_height + self.lane_spacing) + 40
            m_r = QRect(5, y + 25, 20, 20); s_r = QRect(30, y + 25, 20, 20)
            if m_r.contains(event.pos()): self.mutes[i] = not self.mutes[i]; self.update(); self.timelineChanged.emit(); return
            if s_r.contains(event.pos()): self.solos[i] = not self.solos[i]; self.update(); self.timelineChanged.emit(); return
        if event.pos().y() < 40: self.setting_loop = True; self.loop_start_ms = event.pos().x() / self.pixels_per_ms; self.loop_end_ms = self.loop_start_ms; self.loop_enabled = True; self.update(); return
        if event.button() == Qt.MouseButton.LeftButton:
            cs = None
            for seg in reversed(self.segments):
                r = self.get_seg_rect(seg); fi = r.left() + int(seg.fade_in_ms * self.pixels_per_ms); fo = r.right() - int(seg.fade_out_ms * self.pixels_per_ms)
                if QRect(fi-10, r.top()-10, 20, 20).contains(event.pos()): self.selected_segment = seg; self.fade_in_dragging = True; self.drag_start_pos = event.pos(); self.drag_start_fade = seg.fade_in_ms; self.update(); return
                if QRect(fo-10, r.top()-10, 20, 20).contains(event.pos()): self.selected_segment = seg; self.fade_out_dragging = True; self.drag_start_pos = event.pos(); self.drag_start_fade = seg.fade_out_ms; self.update(); return
                if r.contains(event.pos()): cs = seg; break
            self.selected_segment = cs; self.segmentSelected.emit(cs)
            if self.selected_segment:
                self.window().push_undo(); self.drag_start_pos = event.pos(); self.drag_start_ms = self.selected_segment.start_ms; self.drag_start_dur = self.selected_segment.duration_ms; self.drag_start_vol = self.selected_segment.volume; self.drag_start_lane = self.selected_segment.lane; self.drag_start_offset = self.selected_segment.offset_ms; r = self.get_seg_rect(self.selected_segment)
                if event.modifiers() & Qt.KeyboardModifier.AltModifier: self.slipping = True
                elif event.pos().x() > (r.right() - 20): self.resizing = True
                elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier: self.vol_dragging = True
                else: self.dragging = True
            else: self.cursor_pos_ms = event.pos().x() / self.pixels_per_ms; self.window().on_cursor_jump(self.cursor_pos_ms)
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            ts = None
            for seg in reversed(self.segments):
                if self.get_seg_rect(seg).contains(event.pos()): ts = seg; break
            m = QMenu(self)
            if ts:
                pa = m.addAction("⭐ Unmark Primary" if ts.is_primary else "⭐ Set as Primary"); sa = m.addAction("✂ Split at Cursor"); qa = m.addAction("🪄 Quantize to Grid")
                pm = m.addMenu("🎵 Shift Pitch")
                for i in range(-6, 7): t = f"{i:+} st" if i != 0 else "Original"; p_act = pm.addAction(t); p_act.setData(i)
                m.addSeparator(); da = m.addAction("🗑 Remove Track"); act = m.exec(self.mapToGlobal(event.pos()))
                if act == pa: self.window().push_undo(); ts.is_primary = not ts.is_primary
                elif act == sa: self.window().push_undo(); self.split_segment(ts, event.pos().x())
                elif act == qa: self.window().push_undo(); self.quantize_segment(ts)
                elif act in pm.actions(): self.window().push_undo(); ts.pitch_shift = act.data()
                elif act == da: self.window().push_undo(); self.segments.remove(ts)
                if self.selected_segment == ts: self.selected_segment = None
                self.update_geometry(); self.timelineChanged.emit()
            else:
                ba = m.addAction("🪄 Find Bridge Track here"); act = m.exec(self.mapToGlobal(event.pos()))
                if act == ba: self.window().find_bridge_for_gap(event.pos().x())

    def mouseMoveEvent(self, event):
        if self.setting_loop: self.loop_end_ms = max(self.loop_start_ms, event.pos().x() / self.pixels_per_ms); self.update(); return
        if not self.selected_segment: return
        dx = event.pos().x() - self.drag_start_pos.x(); dy = event.pos().y() - self.drag_start_pos.y(); mpb = self.get_ms_per_beat()
        if self.slipping: self.selected_segment.offset_ms = max(0, self.drag_start_offset - dx/self.pixels_per_ms)
        elif self.fade_in_dragging:
            rf = self.drag_start_fade + dx/self.pixels_per_ms; 
            if self.snap_to_grid: rf = round(rf / mpb) * mpb
            self.selected_segment.fade_in_ms = max(0, min(self.selected_segment.duration_ms/2, rf))
        elif self.fade_out_dragging:
            rf = self.drag_start_fade - dx/self.pixels_per_ms; 
            if self.snap_to_grid: rf = round(rf / mpb) * mpb
            self.selected_segment.fade_out_ms = max(0, min(self.selected_segment.duration_ms/2, rf))
        elif self.resizing:
            rd = self.drag_start_dur + dx/self.pixels_per_ms; 
            if self.snap_to_grid: rd = round(rd / mpb) * mpb
            self.selected_segment.duration_ms = max(1000, rd)
        elif self.vol_dragging: self.selected_segment.volume = max(0.0, min(1.5, self.drag_start_vol - dy/150.0))
        elif self.dragging:
            ns = max(0, self.drag_start_ms + dx/self.pixels_per_ms)
            if self.snap_to_grid: ns = round(ns / mpb) * mpb
            for o in self.segments:
                if o == self.selected_segment: continue
                oe = o.start_ms + o.duration_ms
                if abs(ns - oe) < self.snap_threshold_ms: ns = oe
                elif abs(ns - o.start_ms) < self.snap_threshold_ms: ns = o.start_ms
            self.selected_segment.start_ms = ns; nl = max(0, min(4, int((event.pos().y() - 40) // (self.lane_height + self.lane_spacing)))); self.selected_segment.lane = nl
        self.update_geometry(); self.timelineChanged.emit()

    def mouseReleaseEvent(self, event): self.dragging = self.resizing = self.vol_dragging = self.fade_in_dragging = self.fade_out_dragging = self.slipping = self.setting_loop = False; self.update_geometry()
    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            nv = max(10, min(200, int(self.pixels_per_ms * 1000) + (event.angleDelta().y() // 120 * 10)))
            self.pixels_per_ms = nv / 1000.0; self.update_geometry(); self.window().zs.setValue(nv)
        else: super().wheelEvent(event)
    
    def split_segment(self, seg, x_pos):
        sm = x_pos / self.pixels_per_ms; rs = sm - seg.start_ms
        if rs < 500 or rs > (seg.duration_ms - 500): return 
        nd = seg.duration_ms - rs; no = seg.offset_ms + rs; seg.duration_ms = rs 
        td = {'id': seg.id, 'filename': seg.filename, 'file_path': seg.file_path, 'bpm': seg.bpm, 'harmonic_key': seg.key, 'onsets_json': ",".join([str(x/1000.0) for x in seg.onsets])}
        ns = TrackSegment(td, start_ms=sm, duration_ms=nd, lane=seg.lane, offset_ms=no); ns.volume = seg.volume; ns.is_primary = seg.is_primary; ns.waveform = seg.waveform; ns.pitch_shift = seg.pitch_shift; self.segments.append(ns); self.update_geometry(); self.timelineChanged.emit()

    def quantize_segment(self, seg):
        if not seg.onsets: return
        mpb = self.get_ms_per_beat(); stretch = self.target_bpm / seg.bpm; foc = (seg.onsets[0] - seg.offset_ms) * stretch; cwp = seg.start_ms + foc; twp = round(cwp / mpb) * mpb; seg.start_ms += (twp - cwp); self.update(); self.timelineChanged.emit()

    def add_track(self, td, start_ms=None, lane=0):
        if start_ms is None:
            if self.segments: l_s = max(self.segments, key=lambda s: s.start_ms + s.duration_ms); start_ms = l_s.start_ms + l_s.duration_ms - 5000; lane = l_s.lane
            else: start_ms = 0
        ns = TrackSegment(td, start_ms=start_ms, lane=lane); self.segments.append(ns); self.update_geometry(); self.timelineChanged.emit(); return ns

class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents); self.message = "Processing Journey..."; self.hide()
    def paintEvent(self, event):
        p = QPainter(self); p.fillRect(self.rect(), QColor(0, 0, 0, 180)); p.setPen(Qt.GlobalColor.white); p.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold)); p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
    def show_loading(self, m="Processing..."): self.message = m; self.setGeometry(self.parent().rect()); self.raise_(); self.show(); QApplication.processEvents()
    def hide_loading(self): self.hide()

class AudioSequencerApp(QMainWindow):
    def __init__(self):
        super().__init__(); self.dm = DataManager(); self.scorer = CompatibilityScorer(); self.processor = AudioProcessor(); self.renderer = FlowRenderer(); self.generator = TransitionGenerator(); self.orchestrator = FullMixOrchestrator(); self.undo_manager = UndoManager(); self.selected_library_track = None
        self.player = QMediaPlayer(); self.audio_output = QAudioOutput(); self.player.setAudioOutput(self.audio_output); self.audio_output.setVolume(0.8); self.preview_path = "temp_preview.wav"; self.preview_dirty = True
        self.play_timer = QTimer(); self.play_timer.setInterval(20); self.play_timer.timeout.connect(self.update_playback_cursor); self.is_playing = False
        self.init_ui(); self.load_library(); self.loading_overlay = LoadingOverlay(self.centralWidget())
    
    def update_playback_cursor(self):
        if self.is_playing:
            p = self.player.position()
            if self.timeline_widget.loop_enabled and p >= self.timeline_widget.loop_end_ms: self.player.setPosition(int(self.timeline_widget.loop_start_ms)); p = self.timeline_widget.loop_start_ms
            self.timeline_widget.cursor_pos_ms = p; self.timeline_widget.update()
            t_e = 0
            for s in self.timeline_widget.segments:
                if s.start_ms <= p <= s.start_ms + s.duration_ms:
                    any_s = any(self.timeline_widget.solos); is_a = (self.timeline_widget.solos[s.lane] if any_s else not self.timeline_widget.mutes[s.lane])
                    if is_a: t_e += s.volume
            mw = int(min(1.0, t_e / 3.0) * 20); ms = "█" * mw + "░" * (20 - mw); self.status_bar.showMessage(f"Playing | Energy: [{ms}] | {p/1000:.1f}s")
            if not self.timeline_widget.loop_enabled and p >= self.player.duration() and self.player.duration() > 0: self.stop_playback()

    def stop_playback(self): self.player.stop(); self.play_timer.stop(); self.is_playing = False; self.ptb.setText("▶ Play Journey"); self.status_bar.showMessage("Stopped.")
    def toggle_playback(self):
        if not self.timeline_widget.segments: return
        if self.is_playing: self.player.pause(); self.play_timer.stop(); self.is_playing = False; self.ptb.setText("▶ Play Journey")
        else:
            if self.preview_dirty: self.render_preview_for_playback()
            self.player.setPosition(int(self.timeline_widget.cursor_pos_ms)); self.player.play(); self.play_timer.start(); self.is_playing = True; self.ptb.setText("⏸ Pause Preview")

    def render_preview_for_playback(self):
        self.loading_overlay.show_loading("Building Preview..."); 
        try:
            ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms); tb = float(self.tbe.text()) if self.tbe.text() else 124.0
            rd = [s.to_dict() for s in ss]; self.renderer.render_timeline(rd, self.preview_path, target_bpm=tb, mutes=self.timeline_widget.mutes, solos=self.timeline_widget.solos); self.player.setSource(QUrl.fromLocalFile(os.path.abspath(self.preview_path))); self.preview_dirty = False
        except Exception as e: show_error(self, "Preview Error", "Failed to build audio.", e)
        finally: self.loading_overlay.hide_loading()

    def jump_to_start(self):
        self.timeline_widget.cursor_pos_ms = 0
        if self.is_playing: self.player.setPosition(0)
        self.timeline_widget.update()

    def push_undo(self): 
        self.preview_dirty = True
        self.undo_manager.push_state(self.timeline_widget.segments)

    def undo(self): 
        ns = self.undo_manager.undo(self.timeline_widget.segments)
        if ns: self.apply_state(ns)

    def redo(self): 
        ns = self.undo_manager.redo(self.timeline_widget.segments)
        if ns: self.apply_state(ns)

    def apply_state(self, sl):
        self.timeline_widget.segments = []
        for sj in sl:
            s = json.loads(sj); td = {'id': s['id'], 'filename': s['filename'], 'file_path': s['file_path'], 'bpm': s['bpm'], 'harmonic_key': s['key'], 'onsets_json': s.get('onsets_json', "")}
            seg = TrackSegment(td, start_ms=s['start_ms'], duration_ms=s['duration_ms'], lane=s['lane'], offset_ms=s['offset_ms']); seg.volume = s['volume']; seg.is_primary = s['is_primary']; seg.fade_in_ms = s['fade_in_ms']; seg.fade_out_ms = s['fade_out_ms']; seg.pitch_shift = s.get('pitch_shift', 0); seg.waveform = self.processor.get_waveform_envelope(seg.file_path); self.timeline_widget.segments.append(seg)
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
        mp = QFrame(); mp.setFixedWidth(250); mlayout = QVBoxLayout(mp); ag = QFrame(); ag.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px;"); al = QVBoxLayout(ag); al.addWidget(QLabel("<h3>📊 Analytics Board</h3>"))
        self.mod_toggle = QPushButton("🔍 Hide Markers"); self.mod_toggle.setCheckable(True); self.mod_toggle.clicked.connect(self.toggle_analytics); al.addWidget(self.mod_toggle)
        self.grid_toggle = QPushButton("📏 Grid Snap: ON"); self.grid_toggle.setCheckable(True); self.grid_toggle.setChecked(True); self.grid_toggle.clicked.connect(self.toggle_grid); al.addWidget(self.grid_toggle)
        ur = QHBoxLayout(); self.ub = QPushButton("↶ Undo"); self.ub.clicked.connect(self.undo); ur.addWidget(self.ub); self.rb = QPushButton("↷ Redo"); self.rb.clicked.connect(self.redo); ur.addWidget(self.rb); al.addLayout(ur)
        self.stats_label = QLabel("Timeline empty"); self.stats_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: bold;"); al.addWidget(self.stats_label)
        save_btn = QPushButton("💾 Save Journey"); save_btn.clicked.connect(self.save_project); al.addWidget(save_btn); load_btn = QPushButton("📂 Load Journey"); load_btn.clicked.connect(self.load_project); al.addWidget(load_btn); mlayout.addWidget(ag); mlayout.addSpacing(10)
        mlayout.addWidget(QLabel("<h3>⚡ Actions</h3>")); self.atb = QPushButton("➕ Add to Timeline"); self.atb.clicked.connect(self.add_selected_to_timeline); mlayout.addWidget(self.atb)
        self.pb = QPushButton("▶ Preview"); self.pb.clicked.connect(self.play_selected); mlayout.addWidget(self.pb); mlayout.addSpacing(10)
        self.prop_group = QFrame(); self.prop_group.setStyleSheet("background-color: #252525; border: 1px solid #444; border-radius: 8px; padding: 10px;"); self.prop_layout = QVBoxLayout(self.prop_group); self.prop_layout.addWidget(QLabel("<b>Track Properties</b>"))
        v_lay = QHBoxLayout(); v_lay.addWidget(QLabel("Vol:")); self.vol_slider = QSlider(Qt.Orientation.Horizontal); self.vol_slider.setRange(0, 150); self.vol_slider.valueChanged.connect(self.on_prop_changed); v_lay.addWidget(self.vol_slider); self.prop_layout.addLayout(v_lay)
        p_lay = QHBoxLayout(); p_lay.addWidget(QLabel("Pitch:")); self.pitch_combo = QComboBox()
        for i in range(-6, 7): self.pitch_combo.addItem(f"{i:+} st", i)
        self.pitch_combo.currentIndexChanged.connect(self.on_prop_changed); p_lay.addWidget(self.pitch_combo); self.prop_layout.addLayout(p_lay)
        self.prim_check = QCheckBox("Set as Primary Track"); self.prim_check.stateChanged.connect(self.on_prop_changed); self.prop_layout.addWidget(self.prim_check); self.prop_group.setVisible(False); mlayout.addWidget(self.prop_group); mlayout.addStretch(); tp.addWidget(mp)
        rp = QFrame(); rp.setFixedWidth(450); rl = QVBoxLayout(rp); rl.addWidget(QLabel("<h3>✨ Smart Suggestions</h3>")); self.rec_list = DraggableTable(0, 2); self.rec_list.setHorizontalHeaderLabels(["Match %", "Track"]); self.rec_list.itemDoubleClicked.connect(self.on_rec_double_clicked); rl.addWidget(self.rec_list); tp.addWidget(rp); ml.addLayout(tp, stretch=1)
        th = QHBoxLayout(); th.addWidget(QLabel("<h2>🎞 Timeline Journey</h2>"))
        self.sb = QPushButton("⏹"); self.sb.setFixedWidth(40); self.sb.clicked.connect(self.jump_to_start); th.addWidget(self.sb)
        self.ptb = QPushButton("▶ Play Journey"); self.ptb.setFixedWidth(120); self.ptb.clicked.connect(self.toggle_playback); th.addWidget(self.ptb)
        th.addSpacing(20); th.addWidget(QLabel("Zoom:")); self.zs = QSlider(Qt.Orientation.Horizontal); self.zs.setRange(10, 200); self.zs.setValue(50); self.zs.setFixedWidth(150); self.zs.valueChanged.connect(self.on_zoom_changed); th.addWidget(self.zs); th.addStretch()
        self.agb = QPushButton("🪄 Auto-Generate Path"); self.agb.clicked.connect(self.auto_populate_timeline); th.addWidget(self.agb); self.cb = QPushButton("🗑 Clear"); self.cb.clicked.connect(self.clear_timeline); th.addWidget(self.cb)
        th.addWidget(QLabel("Target BPM:")); self.tbe = QLineEdit("124"); self.tbe.setFixedWidth(60); self.tbe.textChanged.connect(self.on_bpm_changed); th.addWidget(self.tbe)
        self.render_btn = QPushButton("🚀 RENDER FINAL MIX"); self.render_btn.setStyleSheet("background-color: #007acc; padding: 12px 25px; color: white; font-weight: bold;"); self.render_btn.clicked.connect(self.render_timeline); th.addWidget(self.render_btn)
        self.stems_btn = QPushButton("📦 EXPORT STEMS"); self.stems_btn.setStyleSheet("background-color: #444; padding: 12px 15px; color: white; font-weight: bold;"); self.stems_btn.clicked.connect(self.export_stems); th.addWidget(self.stems_btn); ml.addLayout(th)
        t_s = QScrollArea(); t_s.setWidgetResizable(True); t_s.setStyleSheet("QScrollArea { background-color: #1a1a1a; border: 1px solid #333; } QScrollBar:horizontal { height: 12px; background: #222; } QScrollBar::handle:horizontal { background: #444; border-radius: 6px; }"); self.timeline_widget = TimelineWidget(); t_s.setWidget(self.timeline_widget); ml.addWidget(t_s, stretch=1)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.status_bar.showMessage("Ready.")
        self.timeline_widget.segmentSelected.connect(self.on_segment_selected); self.timeline_widget.timelineChanged.connect(self.update_status)
        
        # Comprehensive App Styling
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

    def on_segment_selected(self, s):
        if s:
            self.status_bar.showMessage(f"Selected: {s.filename}")
            self.prop_group.setVisible(True); self.vol_slider.blockSignals(True); self.vol_slider.setValue(int(s.volume * 100)); self.vol_slider.blockSignals(False)
            self.pitch_combo.blockSignals(True); idx = self.pitch_combo.findData(s.pitch_shift); self.pitch_combo.setCurrentIndex(idx); self.pitch_combo.blockSignals(False)
            self.prim_check.blockSignals(True); self.prim_check.setChecked(s.is_primary); self.prim_check.blockSignals(False)
        else: self.prop_group.setVisible(False); self.update_status()

    def on_prop_changed(self):
        sel = self.timeline_widget.selected_segment
        if sel: self.push_undo(); sel.volume = self.vol_slider.value() / 100.0; sel.pitch_shift = self.pitch_combo.currentData(); sel.is_primary = self.prim_check.isChecked(); self.timeline_widget.update(); self.update_status()

    def on_cursor_jump(self, ms):
        if self.is_playing: self.player.setPosition(int(ms))
        self.update_status()

    def find_bridge_for_gap(self, x):
        gm = x / self.timeline_widget.pixels_per_ms; ps = ns = None; ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        for s in ss:
            if s.start_ms + s.duration_ms <= gm: ps = s
            elif s.start_ms >= gm:
                if ns is None: ns = s
        if not ps or not ns: self.status_bar.showMessage("Need track before AND after gap."); return
        self.loading_overlay.show_loading("Finding bridge..."); 
        try:
            conn = self.dm.get_conn(); conn.row_factory = sqlite3_factory; cursor = conn.cursor(); cursor.execute("SELECT * FROM tracks WHERE id NOT IN (?, ?)", (ps.id, ns.id)); cs = cursor.fetchall(); results = []
            for c in cs:
                cd = dict(c); ce = self.dm.get_embedding(cd['clp_embedding_id']) if cd['clp_embedding_id'] else None; sc = self.scorer.calculate_bridge_score(ps.__dict__, ns.__dict__, cd, c_emb=ce); results.append((sc, cd))
            results.sort(key=lambda x: x[0], reverse=True); self.rec_list.setRowCount(0)
            for sc, ot in results[:15]:
                ri = self.rec_list.rowCount(); self.rec_list.insertRow(ri); si = QTableWidgetItem(f"{sc}% (BRIDGE)"); si.setData(Qt.ItemDataRole.UserRole, ot['id']); self.rec_list.setItem(ri, 0, si); self.rec_list.setItem(ri, 1, QTableWidgetItem(ot['filename']))
            self.loading_overlay.hide_loading(); self.status_bar.showMessage(f"AI found {len(results)} potential bridges."); conn.close()
        except Exception as e: self.loading_overlay.hide_loading(); show_error(self, "Bridge Error", "AI Bridge search failed.", e)

    def on_zoom_changed(self, v): self.timeline_widget.pixels_per_ms = v / 1000.0; self.timeline_widget.update_geometry()
    def clear_timeline(self):
        if QMessageBox.question(self, "Clear", "Clear journey?") == QMessageBox.StandardButton.Yes: self.push_undo(); self.timeline_widget.segments = []; self.timeline_widget.update_geometry(); self.update_status()
    def on_search_text_changed(self, t):
        if not t:
            for r in range(self.library_table.rowCount()): self.library_table.setRowHidden(r, False)
            return
        q = t.lower(); 
        for r in range(self.library_table.rowCount()): self.library_table.setRowHidden(r, q not in self.library_table.item(r, 0).text().lower())
    def trigger_semantic_search(self):
        q = self.search_bar.text()
        if len(q) < 3: return
        self.loading_overlay.show_loading(f"AI Search: '{q}'..."); self.st = SearchThread(q, self.dm); self.st.resultsFound.connect(self.on_semantic_results); self.st.errorOccurred.connect(self.on_search_error); self.st.start()
    def on_semantic_results(self, res):
        self.loading_overlay.hide_loading(); self.library_table.setRowCount(0)
        for r in res:
            ri = self.library_table.rowCount(); self.library_table.insertRow(ri); match = int(max(0, 1.0 - r.get('distance', 1.0)) * 100); ni = QTableWidgetItem(r['filename']); ni.setData(Qt.ItemDataRole.UserRole, r['id'])
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
                td = {'id': s['id'], 'filename': s['filename'], 'file_path': s['file_path'], 'bpm': s['bpm'], 'harmonic_key': s['key'], 'onsets_json': s.get('onsets_json', "")}; seg = TrackSegment(td, start_ms=s['start_ms'], duration_ms=s['duration_ms'], lane=s['lane'], offset_ms=s['offset_ms']); seg.volume = s['volume']; seg.is_primary = s['is_primary']; seg.fade_in_ms = s['fade_in_ms']; seg.fade_out_ms = s['fade_out_ms']; seg.pitch_shift = s.get('pitch_shift', 0); seg.waveform = self.processor.get_waveform_envelope(seg.file_path); self.timeline_widget.segments.append(seg)
            self.timeline_widget.update_geometry(); self.update_status()
    def on_bpm_changed(self, t):
        try: self.timeline_widget.target_bpm = float(t); self.preview_dirty = True; self.timeline_widget.update(); self.update_status()
        except: pass
    def toggle_analytics(self): self.timeline_widget.show_modifications = not self.mod_toggle.isChecked(); self.mod_toggle.setText("🔍 Show Markers" if self.mod_toggle.isChecked() else "🔍 Hide Markers"); self.timeline_widget.update()
    def toggle_grid(self): self.timeline_widget.snap_to_grid = self.grid_toggle.isChecked(); self.grid_toggle.setText(f"📏 Grid Snap: {'ON' if self.timeline_widget.snap_to_grid else 'OFF'}"); self.timeline_widget.update()
    def update_status(self):
        count = len(self.timeline_widget.segments)
        if count > 0:
            tdur = max(s.start_ms + s.duration_ms for s in self.timeline_widget.segments); abpm = sum(s.bpm for s in self.timeline_widget.segments) / count; bdiff = abs(abpm - self.timeline_widget.target_bpm); self.status_bar.showMessage(f"Timeline: {count} tracks | {tdur/1000:.1f}s total mix")
            at = (f"<b>Journey Stats</b><br>Tracks: {count}<br>Duration: {tdur/1000:.1f}s<br>Avg BPM: {abpm:.1f}<br>")
            ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms); fs = 100
            for i in range(len(ss) - 1):
                s1, s2 = ss[i], ss[i+1]
                if s2.start_ms < (s1.start_ms + s1.duration_ms + 2000):
                    if self.scorer.calculate_harmonic_score(s1.key, s2.key) < 60: fs -= 10
            at += f"<b>Flow Health:</b> {max(0, fs)}%<br>"
            if self.timeline_widget.selected_segment:
                sel = self.timeline_widget.selected_segment; at += f"<hr><b>Selected Clip:</b><br>{sel.filename[:20]}<br>Key: {sel.key}"
                for o in self.timeline_widget.segments:
                    if o != sel and max(sel.start_ms, o.start_ms) < min(sel.start_ms + sel.duration_ms, o.start_ms + o.duration_ms):
                        hs = self.scorer.calculate_harmonic_score(sel.key, o.key); color = "#00ff66" if hs >= 100 else "#ccff00" if hs >= 80 else "#ff5555"; at += f"<br>vs '{o.filename[:8]}...': <span style='color: {color};'>{hs}%</span>"
            self.stats_label.setText(at)
        else: self.status_bar.showMessage("Ready."); self.stats_label.setText("Timeline empty")
    def load_library(self):
        try:
            conn = self.dm.get_conn(); cursor = conn.cursor(); cursor.execute("SELECT id, filename, bpm, harmonic_key FROM tracks"); rows = cursor.fetchall(); self.library_table.setRowCount(0)
            for r in rows:
                ri = self.library_table.rowCount(); self.library_table.insertRow(ri); ni = QTableWidgetItem(r[1]); ni.setData(Qt.ItemDataRole.UserRole, r[0]); self.library_table.setItem(ri, 0, ni); self.library_table.setItem(ri, 1, QTableWidgetItem(f"{r[2]:.1f}")); self.library_table.setItem(ri, 2, QTableWidgetItem(r[3]))
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
        if seq:
            self.timeline_widget.segments = []; cm = 0
            for i, t in enumerate(seq):
                seg = self.timeline_widget.add_track(t, start_ms=cm); seg.waveform = self.processor.get_waveform_envelope(t['file_path']); cm += (20000 if i % 2 == 0 else 30000) - 8000 
            self.timeline_widget.update_geometry()
        self.loading_overlay.hide_loading()
    def render_timeline(self):
        if not self.timeline_widget.segments: return
        ss = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms); tb = float(self.tbe.text()) if self.tbe.text() else 124.0
        self.loading_overlay.show_loading("Rendering Mix..."); 
        try:
            out = "timeline_mix.mp3"; rd = [s.to_dict() for s in ss]; self.renderer.render_timeline(rd, out, target_bpm=tb, mutes=self.timeline_widget.mutes, solos=self.timeline_widget.solos); self.loading_overlay.hide_loading(); QMessageBox.information(self, "Success", f"Mix rendered: {out}"); os.startfile(out)
        except Exception as e: self.loading_overlay.hide_loading(); show_error(self, "Render Error", "Failed to render.", e)
    def export_stems(self):
        if not self.timeline_widget.segments: return
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder"); 
        if not folder: return
        tb = float(self.tbe.text()) if self.tbe.text() else 124.0; self.loading_overlay.show_loading("Exporting Stems...")
        try:
            rd = [s.to_dict() for s in self.timeline_widget.segments]; paths = self.renderer.render_stems(rd, folder, target_bpm=tb); self.loading_overlay.hide_loading(); QMessageBox.information(self, "Stems Exported", f"Successfully exported {len(paths)} stems to:\n{folder}"); os.startfile(folder)
        except Exception as e: self.loading_overlay.hide_loading(); show_error(self, "Stem Export Error", "Failed to export stems.", e)
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
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space: self.toggle_playback()
        elif event.key() == Qt.Key.Key_M:
            sel = self.timeline_widget.selected_segment
            if sel: self.timeline_widget.mutes[sel.lane] = not self.timeline_widget.mutes[sel.lane]; self.timeline_widget.update(); self.preview_dirty = True
        elif event.key() == Qt.Key.Key_S:
            sel = self.timeline_widget.selected_segment
            if sel: self.timeline_widget.solos[sel.lane] = not self.timeline_widget.solos[sel.lane]; self.timeline_widget.update(); self.preview_dirty = True
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_Z: self.undo()
            elif event.key() == Qt.Key.Key_Y: self.redo()
        else: super().keyPressEvent(event)

def sqlite3_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description): d[col[0]] = row[idx]
    return d

if __name__ == "__main__":
    app = QApplication(sys.argv); window = AudioSequencerApp(); window.show(); sys.exit(app.exec())
