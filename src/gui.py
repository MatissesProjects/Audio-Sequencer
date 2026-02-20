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
from PyQt6.QtCore import Qt, QSize, QRect, pyqtSignal, QPoint, QMimeData, QThread
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont, QDrag

# Project Imports
from src.database import DataManager
from src.scoring import CompatibilityScorer
from src.processor import AudioProcessor
from src.renderer import FlowRenderer
from src.generator import TransitionGenerator
from src.orchestrator import FullMixOrchestrator
from src.embeddings import EmbeddingEngine

class SearchThread(QThread):
    resultsFound = pyqtSignal(list)
    errorOccurred = pyqtSignal(str)

    def __init__(self, query, dm):
        super().__init__()
        self.query = query
        self.dm = dm

    def run(self):
        try:
            # We initialize engine here to avoid thread safety issues with some torch versions
            engine = EmbeddingEngine()
            text_emb = engine.get_text_embedding(self.query)
            # Query ChromaDB via DataManager
            results = self.dm.search_embeddings(text_emb, n_results=20)
            self.resultsFound.emit(results)
        except Exception as e:
            self.errorOccurred.emit(str(e))

class DetailedErrorDialog(QDialog):
    def __init__(self, title, message, details, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout(self)
        
        # Icon and Message
        msg_layout = QHBoxLayout()
        icon_label = QLabel("❌")
        icon_label.setStyleSheet("font-size: 32px;")
        msg_layout.addWidget(icon_label)
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        msg_layout.addWidget(msg_label, stretch=1)
        layout.addLayout(msg_layout)

        # Technical Details
        layout.addWidget(QLabel("Technical Details:"))
        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setText(details)
        self.details_box.setStyleSheet("background-color: #1a1a1a; color: #ff5555; font-family: Consolas, monospace;")
        layout.addWidget(self.details_box)

        # Buttons
        btn_layout = QHBoxLayout()
        copy_btn = QPushButton("📋 Copy to Clipboard")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        btn_layout.addWidget(copy_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        self.setStyleSheet("""
            QDialog { background-color: #252525; color: white; }
            QLabel { color: white; }
            QPushButton { background-color: #444; color: white; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #555; }
        """)

    def copy_to_clipboard(self):
        QApplication.clipboard().setText(self.details_box.toPlainText())
        QMessageBox.information(self, "Copied", "Error details copied to clipboard.")

def show_error(parent, title, message, exception):
    details = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    dialog = DetailedErrorDialog(title, message, details, parent)
    dialog.exec()

class TrackSegment:
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
        self.lane = lane
        self.is_primary = False
        self.waveform = [] 
        self.fade_in_ms = 2000
        self.fade_out_ms = 2000
        self.pitch_shift = 0 # Semitones
        self.color = QColor(70, 130, 180, 200) # SteelBlue

    def to_dict(self):
        return {
            'id': self.id, 'filename': self.filename, 'file_path': self.file_path,
            'bpm': self.bpm, 'key': self.key, 'start_ms': self.start_ms,
            'duration_ms': self.duration_ms, 'offset_ms': self.offset_ms,
            'volume': self.volume, 'lane': self.lane, 'is_primary': self.is_primary,
            'fade_in_ms': self.fade_in_ms, 'fade_out_ms': self.fade_out_ms,
            'pitch_shift': self.pitch_shift
        }

class UndoManager:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def push_state(self, segments):
        # Save a deep copy of segments state
        state = [json.dumps(s.to_dict()) for s in segments]
        self.undo_stack.append(state)
        self.redo_stack.clear()
        if len(self.undo_stack) > 50: self.undo_stack.pop(0)

    def undo(self, current_segments):
        if not self.undo_stack: return None
        self.redo_stack.append([json.dumps(s.to_dict()) for s in current_segments])
        return self.undo_stack.pop()

    def redo(self, current_segments):
        if not self.redo_stack: return None
        self.undo_stack.append([json.dumps(s.to_dict()) for s in current_segments])
        return self.redo_stack.pop()
class DraggableTable(QTableWidget):
    """A table that allows dragging its rows as track data."""
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if item:
                row = item.row()
                track_id = self.item(row, 0).data(Qt.ItemDataRole.UserRole)
                if track_id is not None:
                    drag = QDrag(self)
                    mime = QMimeData()
                    mime.setText(str(track_id))
                    drag.setMimeData(mime)
                    drag.exec(Qt.DropAction.CopyAction)
        super().mousePressEvent(event)

class TimelineWidget(QWidget):
    segmentSelected = pyqtSignal(object)
    timelineChanged = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.segments = []
        self.setMinimumHeight(550)
        self.setAcceptDrops(True)
        self.pixels_per_ms = 0.05 
        self.selected_segment = None
        self.dragging = False
        self.resizing = False
        self.vol_dragging = False
        self.fade_in_dragging = False
        self.fade_out_dragging = False
        
        self.drag_start_pos = None
        self.drag_start_ms = 0
        self.drag_start_dur = 0
        self.drag_start_vol = 1.0
        self.drag_start_lane = 0
        self.drag_start_fade = 0
        
        self.lane_height = 120
        self.lane_spacing = 10
        self.snap_threshold_ms = 2000 
        self.target_bpm = 124.0
        self.show_modifications = True
        self.cursor_pos_ms = 0
        self.show_waveforms = True
        self.update_geometry()

    def update_geometry(self):
        max_ms = 600000 
        if self.segments:
            last_end = max(s.start_ms + s.duration_ms for s in self.segments)
            max_ms = max(max_ms, last_end + 60000)
        
        self.setMinimumWidth(int(max_ms * self.pixels_per_ms))
        self.update()

    def get_seg_rect(self, seg):
        x = int(seg.start_ms * self.pixels_per_ms)
        w = int(seg.duration_ms * self.pixels_per_ms)
        h = int((self.lane_height - 20) * seg.volume)
        y_center = (seg.lane * (self.lane_height + self.lane_spacing)) + (self.lane_height // 2) + 40
        y = y_center - (h // 2)
        return QRect(x, y, w, h)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(25, 25, 25))
        
        # Draw lane backgrounds
        painter.setPen(QPen(QColor(45, 45, 45), 1))
        for i in range(5): 
            y = i * (self.lane_height + self.lane_spacing) + 40
            painter.fillRect(0, y, self.width(), self.lane_height, QColor(32, 32, 32))
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(5, y + 15, f"LANE {i+1}")

        # Time markings
        painter.setPen(QColor(70, 70, 70))
        for i in range(0, 3600000, 10000): 
            x = int(i * self.pixels_per_ms)
            if x > self.width(): break
            painter.drawLine(x, 0, x, self.height())
            if i % 30000 == 0:
                painter.drawText(x + 5, 25, f"{i//1000}s")

        for seg in self.segments:
            rect = self.get_seg_rect(seg)
            color = QColor(seg.color)
            color.setAlpha(int(120 + 135 * (min(seg.volume, 1.5) / 1.5)))
            
            if seg == self.selected_segment:
                painter.setBrush(QBrush(color.lighter(130)))
                painter.setPen(QPen(Qt.GlobalColor.white, 3))
            elif seg.is_primary:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(255, 215, 0), 3)) 
            else:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(200, 200, 200), 1))
            
            painter.drawRoundedRect(rect, 6, 6)

            if self.show_waveforms and seg.waveform:
                painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
                pts = len(seg.waveform)
                mid_y = rect.center().y()
                max_h = rect.height() // 2
                for i in range(0, rect.width(), 2):
                    idx = int((i / rect.width()) * pts)
                    if idx < pts:
                        val = seg.waveform[idx] * max_h
                        painter.drawLine(rect.left() + i, int(mid_y - val), rect.left() + i, int(mid_y + val))

            # Draw Fades
            fade_in_w = int(seg.fade_in_ms * self.pixels_per_ms)
            fade_out_w = int(seg.fade_out_ms * self.pixels_per_ms)
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1, Qt.PenStyle.DashLine))
            painter.drawLine(rect.left(), rect.bottom(), rect.left() + fade_in_w, rect.top())
            painter.drawLine(rect.right() - fade_out_w, rect.top(), rect.right(), rect.bottom())
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(rect.left() + fade_in_w - 4, rect.top() - 4, 8, 8)
            painter.drawEllipse(rect.right() - fade_out_w - 4, rect.top() - 4, 8, 8)

            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(rect.adjusted(8, 8, -8, -8), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, seg.filename)
            
            badge_y = rect.bottom() - 22
            badge_x = rect.left() + 8
            
            if seg.is_primary:
                painter.setBrush(QBrush(QColor(255, 215, 0))) 
                painter.setPen(Qt.PenStyle.NoPen)
                badge_rect = QRect(badge_x, badge_y, 60, 16)
                painter.drawRoundedRect(badge_rect, 4, 4)
                painter.setPen(Qt.GlobalColor.black)
                painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "PRIMARY")
                badge_x += 65

            if self.show_modifications:
                if abs(seg.bpm - self.target_bpm) > 0.1:
                    painter.setBrush(QBrush(QColor(255, 165, 0))) 
                    painter.setPen(Qt.PenStyle.NoPen)
                    badge_rect = QRect(badge_x, badge_y, 55, 16)
                    painter.drawRoundedRect(badge_rect, 4, 4)
                    painter.setPen(Qt.GlobalColor.black)
                    painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                    painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "STRETCH")
                    badge_x += 60

                if abs(seg.volume - 1.0) > 0.05:
                    painter.setBrush(QBrush(QColor(0, 200, 255))) 
                    painter.setPen(Qt.PenStyle.NoPen)
                    badge_rect = QRect(badge_x, badge_y, 40, 16)
                    painter.drawRoundedRect(badge_rect, 4, 4)
                    painter.setPen(Qt.GlobalColor.black)
                    painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                    painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"{int(seg.volume*100)}%")
                    badge_x += 45

                if seg.pitch_shift != 0:
                    painter.setBrush(QBrush(QColor(200, 100, 255))) # Purple
                    painter.setPen(Qt.PenStyle.NoPen)
                    badge_rect = QRect(badge_x, badge_y, 40, 16)
                    painter.drawRoundedRect(badge_rect, 4, 4)
                    painter.setPen(Qt.GlobalColor.black)
                    painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                    painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"{seg.pitch_shift:+}st")

        cursor_x = int(self.cursor_pos_ms * self.pixels_per_ms)
        painter.setPen(QPen(QColor(255, 50, 50), 2))
        painter.drawLine(cursor_x, 0, cursor_x, self.height())
        painter.setBrush(QBrush(QColor(255, 50, 50)))
        painter.drawPolygon(QPoint(cursor_x - 8, 0), QPoint(cursor_x + 8, 0), QPoint(cursor_x, 12))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            clicked_seg = None
            for seg in reversed(self.segments):
                rect = self.get_seg_rect(seg)
                fi_x = rect.left() + int(seg.fade_in_ms * self.pixels_per_ms)
                fo_x = rect.right() - int(seg.fade_out_ms * self.pixels_per_ms)
                
                if QRect(fi_x-10, rect.top()-10, 20, 20).contains(event.pos()):
                    self.selected_segment = seg
                    self.fade_in_dragging = True
                    self.drag_start_pos = event.pos()
                    self.drag_start_fade = seg.fade_in_ms
                    self.update(); return
                if QRect(fo_x-10, rect.top()-10, 20, 20).contains(event.pos()):
                    self.selected_segment = seg
                    self.fade_out_dragging = True
                    self.drag_start_pos = event.pos()
                    self.drag_start_fade = seg.fade_out_ms
                    self.update(); return

                if rect.contains(event.pos()):
                    clicked_seg = seg
                    break
            
            self.selected_segment = clicked_seg
            self.segmentSelected.emit(clicked_seg)
            
            if self.selected_segment:
                self.drag_start_pos = event.pos()
                self.drag_start_ms = self.selected_segment.start_ms
                self.drag_start_dur = self.selected_segment.duration_ms
                self.drag_start_vol = self.selected_segment.volume
                self.drag_start_lane = self.selected_segment.lane
                rect = self.get_seg_rect(self.selected_segment)
                if event.pos().x() > (rect.right() - 20): self.resizing = True
                elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier: self.vol_dragging = True
                else: self.dragging = True
            else:
                self.cursor_pos_ms = event.pos().x() / self.pixels_per_ms
            self.update()
        
        elif event.button() == Qt.MouseButton.RightButton:
            target_seg = None
            for seg in reversed(self.segments):
                if self.get_seg_rect(seg).contains(event.pos()):
                    target_seg = seg
                    break
            
            menu = QMenu(self)
            if target_seg:
                primary_text = "⭐ Unmark Primary" if target_seg.is_primary else "⭐ Set as Primary"
                primary_action = menu.addAction(primary_text)
                split_action = menu.addAction("✂ Split at Cursor")
                
                pitch_menu = menu.addMenu("🎵 Shift Pitch")
                for i in range(-6, 7):
                    text = f"{i:+} Semitones" if i != 0 else "Original Pitch"
                    pa = pitch_menu.addAction(text)
                    pa.setData(i)
                
                menu.addSeparator()
                del_action = menu.addAction("🗑 Remove Track")
                action = menu.exec(self.mapToGlobal(event.pos()))
                
                if action == primary_action:
                    self.window().push_undo()
                    target_seg.is_primary = not target_seg.is_primary
                elif action == split_action:
                    self.window().push_undo()
                    self.split_segment(target_seg, event.pos().x())
                elif action in pitch_menu.actions():
                    self.window().push_undo()
                    target_seg.pitch_shift = action.data()
                elif action == del_action:
                    self.window().push_undo()
                    self.segments.remove(target_seg)
                    if self.selected_segment == target_seg: self.selected_segment = None
                self.update_geometry(); self.timelineChanged.emit()
            else:
                # Clicked empty space
                bridge_action = menu.addAction("🪄 Find Bridge Track here")
                action = menu.exec(self.mapToGlobal(event.pos()))
                if action == bridge_action:
                    self.window().find_bridge_for_gap(event.pos().x())

    def split_segment(self, seg, x_pos):
        split_ms = x_pos / self.pixels_per_ms
        rel_split = split_ms - seg.start_ms
        if rel_split < 500 or rel_split > (seg.duration_ms - 500): return 
        new_dur = seg.duration_ms - rel_split
        new_offset = seg.offset_ms + rel_split
        seg.duration_ms = rel_split 
        track_data = {'id': seg.id, 'filename': seg.filename, 'file_path': seg.file_path, 'bpm': seg.bpm, 'harmonic_key': seg.key}
        new_seg = TrackSegment(track_data, start_ms=split_ms, duration_ms=new_dur, lane=seg.lane, offset_ms=new_offset)
        new_seg.volume = seg.volume; new_seg.is_primary = seg.is_primary; new_seg.waveform = seg.waveform
        self.segments.append(new_seg); self.update_geometry(); self.timelineChanged.emit()

    def mouseMoveEvent(self, event):
        if not self.selected_segment: return
        dx = event.pos().x() - self.drag_start_pos.x()
        dy = event.pos().y() - self.drag_start_pos.y()
        
        if self.fade_in_dragging:
            self.selected_segment.fade_in_ms = max(0, min(self.selected_segment.duration_ms/2, self.drag_start_fade + dx/self.pixels_per_ms))
        elif self.fade_out_dragging:
            self.selected_segment.fade_out_ms = max(0, min(self.selected_segment.duration_ms/2, self.drag_start_fade - dx/self.pixels_per_ms))
        elif self.resizing:
            self.selected_segment.duration_ms = max(1000, self.drag_start_dur + dx/self.pixels_per_ms)
        elif self.vol_dragging:
            self.selected_segment.volume = max(0.0, min(1.5, self.drag_start_vol - dy/150.0))
        elif self.dragging:
            new_start = max(0, self.drag_start_ms + dx/self.pixels_per_ms)
            for other in self.segments:
                if other == self.selected_segment: continue
                oe = other.start_ms + other.duration_ms
                if abs(new_start - oe) < self.snap_threshold_ms: new_start = oe
                elif abs(new_start - other.start_ms) < self.snap_threshold_ms: new_start = other.start_ms
            self.selected_segment.start_ms = new_start
            new_lane = int((event.pos().y() - 40) // (self.lane_height + self.lane_spacing))
            self.selected_segment.lane = max(0, min(4, new_lane))
        self.update_geometry(); self.timelineChanged.emit()

    def mouseReleaseEvent(self, event):
        self.dragging = self.resizing = self.vol_dragging = self.fade_in_dragging = self.fade_out_dragging = False
        self.update_geometry()

    def add_track(self, track_data, start_ms=None):
        lane = 0
        if start_ms is None:
            if self.segments:
                last_seg = max(self.segments, key=lambda s: s.start_ms + s.duration_ms)
                start_ms = last_seg.start_ms + last_seg.duration_ms - 5000 
                lane = last_seg.lane
            else: start_ms = 0
        new_seg = TrackSegment(track_data, start_ms=start_ms, lane=lane)
        self.segments.append(new_seg); self.update_geometry(); self.timelineChanged.emit()
        return new_seg

    def dragEnterEvent(self, event):
        if event.mimeData().hasText(): event.acceptProposedAction()

    def dropEvent(self, event):
        track_id = event.mimeData().text()
        if track_id:
            pos = event.position()
            lane = max(0, min(4, int((pos.y() - 40) // (self.lane_height + self.lane_spacing))))
            self.window().add_track_by_id(int(track_id), pos.x(), lane=lane)
            event.acceptProposedAction()

class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.message = "Processing Journey..."
        self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 180))
        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)

    def show_loading(self, message="Processing..."):
        self.message = message
        self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()
        QApplication.processEvents()

    def hide_loading(self):
        self.hide()

class AudioSequencerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dm = DataManager()
        self.scorer = CompatibilityScorer()
        self.processor = AudioProcessor()
        self.renderer = FlowRenderer()
        self.generator = TransitionGenerator()
        self.orchestrator = FullMixOrchestrator()
        self.undo_manager = UndoManager()
        self.selected_library_track = None
        self.init_ui()
        self.load_library()
        self.loading_overlay = LoadingOverlay(self.centralWidget())

    def push_undo(self):
        self.undo_manager.push_state(self.timeline_widget.segments)

    def undo(self):
        new_state = self.undo_manager.undo(self.timeline_widget.segments)
        if new_state: self.apply_state(new_state)

    def redo(self):
        new_state = self.undo_manager.redo(self.timeline_widget.segments)
        if new_state: self.apply_state(new_state)

    def apply_state(self, state_list):
        self.timeline_widget.segments = []
        for s_json in state_list:
            s = json.loads(s_json)
            seg = TrackSegment(s, start_ms=s['start_ms'], duration_ms=s['duration_ms'], lane=s['lane'], offset_ms=s['offset_ms'])
            seg.volume = s['volume']; seg.is_primary = s['is_primary']; seg.fade_in_ms = s['fade_in_ms']; seg.fade_out_ms = s['fade_out_ms']
            seg.pitch_shift = s.get('pitch_shift', 0)
            seg.waveform = self.processor.get_waveform_envelope(seg.file_path)
            self.timeline_widget.segments.append(seg)
        self.timeline_widget.update_geometry()
        self.update_status()

    def init_ui(self):
        self.setWindowTitle("AudioSequencer AI - The Pro Flow")
        self.setMinimumSize(QSize(1400, 950))
        cw = QWidget(); self.setCentralWidget(cw); ml = QVBoxLayout(cw)
        tp = QHBoxLayout()
        lp = QFrame(); lp.setFixedWidth(450); ll = QVBoxLayout(lp)
        ll.addWidget(QLabel("<h2>📁 Audio Library</h2>"))
        la = QHBoxLayout()
        self.scan_btn = QPushButton("📂 Scan Folder"); self.scan_btn.clicked.connect(self.scan_folder); la.addWidget(self.scan_btn)
        self.embed_btn = QPushButton("🧠 AI Index"); self.embed_btn.clicked.connect(self.run_embedding); la.addWidget(self.embed_btn)
        ll.addLayout(la)
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("🔍 Semantic Search (Enter for AI vibe search)...")
        self.search_bar.textChanged.connect(self.on_search_text_changed)
        self.search_bar.returnPressed.connect(self.trigger_semantic_search)
        search_layout.addWidget(self.search_bar)
        
        self.reset_search_btn = QPushButton("↺")
        self.reset_search_btn.setFixedWidth(30)
        self.reset_search_btn.setToolTip("Reset search and show full library.")
        self.reset_search_btn.clicked.connect(self.load_library)
        search_layout.addWidget(self.reset_search_btn)
        ll.addLayout(search_layout)
        self.library_table = DraggableTable(0, 3); self.library_table.setHorizontalHeaderLabels(["Track Name", "BPM", "Key"])
        self.library_table.setColumnWidth(0, 250); self.library_table.itemSelectionChanged.connect(self.on_library_track_selected); ll.addWidget(self.library_table)
        tp.addWidget(lp)
        mp = QFrame(); mp.setFixedWidth(250); mlayout = QVBoxLayout(mp)
        ag = QFrame(); ag.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 5px;"); al = QVBoxLayout(ag)
        al.addWidget(QLabel("<h3>📊 Analytics Board</h3>"))
        self.mod_toggle = QPushButton("🔍 Hide Markers"); self.mod_toggle.setCheckable(True); self.mod_toggle.clicked.connect(self.toggle_analytics); al.addWidget(self.mod_toggle)
        
        # Undo/Redo
        ur_layout = QHBoxLayout()
        self.undo_btn = QPushButton("↶ Undo"); self.undo_btn.clicked.connect(self.undo); ur_layout.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("↷ Redo"); self.redo_btn.clicked.connect(self.redo); ur_layout.addWidget(self.redo_btn)
        al.addLayout(ur_layout)
        
        self.stats_label = QLabel("Timeline empty"); self.stats_label.setStyleSheet("color: #aaa; font-size: 11px;"); al.addWidget(self.stats_label)
        save_btn = QPushButton("💾 Save Journey"); save_btn.clicked.connect(self.save_project); al.addWidget(save_btn)
        load_btn = QPushButton("📂 Load Journey"); load_btn.clicked.connect(self.load_project); al.addWidget(load_btn)
        mlayout.addWidget(ag); mlayout.addSpacing(10)
        mlayout.addWidget(QLabel("<h3>⚡ Actions</h3>"))
        self.add_to_timeline_btn = QPushButton("➕ Add to Timeline"); self.add_to_timeline_btn.clicked.connect(self.add_selected_to_timeline); mlayout.addWidget(self.add_to_timeline_btn)
        self.play_btn = QPushButton("▶ Preview"); self.play_btn.clicked.connect(self.play_selected); mlayout.addWidget(self.play_btn)
        mlayout.addStretch()
        help_box = QLabel("<b>💡 Pro Tips:</b><br><br>• <b>Drag circles</b> at top to adjust Fades.<br>• <b>Shift + Drag</b> for Volume.<br>• <b>Right-Click</b> to Split.")
        help_box.setStyleSheet("color: #888; font-size: 11px; background: #1a1a1a; padding: 10px;"); mlayout.addWidget(help_box)
        tp.addWidget(mp)
        rp = QFrame(); rp.setFixedWidth(450); rl = QVBoxLayout(rp); rl.addWidget(QLabel("<h3>✨ Smart Suggestions</h3>"))
        self.rec_list = DraggableTable(0, 2); self.rec_list.setHorizontalHeaderLabels(["Match %", "Track"]); self.rec_list.itemDoubleClicked.connect(self.on_rec_double_clicked); rl.addWidget(self.rec_list)
        tp.addWidget(rp); ml.addLayout(tp, stretch=1)
        th = QHBoxLayout(); th.addWidget(QLabel("<h2>🎞 Timeline Journey</h2>"))
        
        # Zoom Control
        th.addSpacing(20)
        th.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 200) # 0.01 to 0.2 pixels per ms
        self.zoom_slider.setValue(50)
        self.zoom_slider.setFixedWidth(150)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        th.addWidget(self.zoom_slider)
        
        th.addStretch()
        self.auto_gen_btn = QPushButton("🪄 Auto-Generate Path"); self.auto_gen_btn.clicked.connect(self.auto_populate_timeline); th.addWidget(self.auto_gen_btn)
        
        self.clear_btn = QPushButton("🗑 Clear"); self.clear_btn.clicked.connect(self.clear_timeline); th.addWidget(self.clear_btn)
        
        th.addWidget(QLabel("Target BPM:")); self.target_bpm_edit = QLineEdit("124"); self.target_bpm_edit.setFixedWidth(60); self.target_bpm_edit.textChanged.connect(self.on_bpm_changed); th.addWidget(self.target_bpm_edit)
        self.render_btn = QPushButton("🚀 RENDER FINAL MIX"); self.render_btn.setStyleSheet("background-color: #007acc; padding: 12px 25px; color: white; font-weight: bold;"); self.render_btn.clicked.connect(self.render_timeline); th.addWidget(self.render_btn)
        ml.addLayout(th)
        self.timeline_scroll = QScrollArea(); self.timeline_scroll.setWidgetResizable(True)
        self.timeline_widget = TimelineWidget(); self.timeline_scroll.setWidget(self.timeline_widget); ml.addWidget(self.timeline_scroll, stretch=1)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.status_bar.showMessage("Ready.")
        self.timeline_widget.segmentSelected.connect(self.on_segment_selected); self.timeline_widget.timelineChanged.connect(self.update_status)
        self.setStyleSheet("QMainWindow { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI'; } QLabel { color: #ffffff; } QTableWidget { background-color: #1e1e1e; gridline-color: #333; } QPushButton { background-color: #333; color: #fff; padding: 8px; border-radius: 4px; }")

    def on_zoom_changed(self, value):
        # Scale zoom: 50 -> 0.05 px/ms
        self.timeline_widget.pixels_per_ms = value / 1000.0
        self.timeline_widget.update_geometry()

    def clear_timeline(self):
        if QMessageBox.question(self, "Clear", "Clear the entire journey?") == QMessageBox.StandardButton.Yes:
            self.timeline_widget.segments = []
            self.timeline_widget.update_geometry()
            self.update_status()

    def on_search_text_changed(self, text):
        if not text:
            # Show all
            for row in range(self.library_table.rowCount()):
                self.library_table.setRowHidden(row, False)
            self.status_bar.showMessage("Showing all library tracks.")
            return

        query = text.lower()
        
        # If text is long, suggest semantic search on enter
        if len(text) > 3:
            self.status_bar.showMessage(f"Searching for vibe: '{text}'... (Press Enter for AI Search)")
        
        # Local text filter
        for row in range(self.library_table.rowCount()):
            match = query in self.library_table.item(row, 0).text().lower()
            self.library_table.setRowHidden(row, not match)

    def trigger_semantic_search(self):
        query = self.search_bar.text()
        if len(query) < 3: return
        
        self.loading_overlay.show_loading(f"AI: Searching for '{query}'...")
        self.search_thread = SearchThread(query, self.dm)
        self.search_thread.resultsFound.connect(self.on_semantic_results)
        self.search_thread.errorOccurred.connect(self.on_search_error)
        self.search_thread.start()

    def on_semantic_results(self, results):
        self.loading_overlay.hide_loading()
        if not results:
            self.status_bar.showMessage("No semantic matches found.")
            return
            
        # We'll update the library table to show ONLY search results
        self.library_table.setRowCount(0)
        # Results format from DataManager.search_embeddings: list of dicts with track info + distance
        for res in results:
            ri = self.library_table.rowCount()
            self.library_table.insertRow(ri)
            
            # Confidence/Match score based on distance
            dist = res.get('distance', 1.0)
            match_pct = int(max(0, 1.0 - dist) * 100)
            
            ni = QTableWidgetItem(res['filename'])
            ni.setData(Qt.ItemDataRole.UserRole, res['id'])
            if match_pct > 70: ni.setForeground(QBrush(QColor(0, 255, 200))) # High confidence
            
            self.library_table.setItem(ri, 0, ni)
            self.library_table.setItem(ri, 1, QTableWidgetItem(f"{res['bpm']:.1f}"))
            self.library_table.setItem(ri, 2, QTableWidgetItem(res['harmonic_key']))
            
        self.status_bar.showMessage(f"AI found {len(results)} matches for your description.")

    def on_search_error(self, err):
        self.loading_overlay.hide_loading()
        QMessageBox.warning(self, "AI Search Error", f"Could not perform semantic search: {err}\n\nMake sure you have run 'AI Index' first.")

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Journey", "", "JSON Files (*.json)")
        if path:
            data = {'target_bpm': self.timeline_widget.target_bpm, 'segments': [s.to_dict() for s in self.timeline_widget.segments]}
            with open(path, 'w') as f: json.dump(data, f)
            self.status_bar.showMessage(f"Journey saved to {os.path.basename(path)}")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Journey", "", "JSON Files (*.json)")
        if path:
            with open(path, 'r') as f: data = json.load(f)
            self.timeline_widget.segments = []
            self.target_bpm_edit.setText(str(data['target_bpm']))
            for s in data['segments']:
                seg = TrackSegment(s, start_ms=s['start_ms'], duration_ms=s['duration_ms'], lane=s['lane'], offset_ms=s['offset_ms'])
                seg.volume = s['volume']; seg.is_primary = s['is_primary']; seg.fade_in_ms = s['fade_in_ms']; seg.fade_out_ms = s['fade_out_ms']
                seg.waveform = self.processor.get_waveform_envelope(seg.file_path)
                self.timeline_widget.segments.append(seg)
            self.timeline_widget.update_geometry(); self.update_status()

    def find_bridge_for_gap(self, x_pos):
        gap_ms = x_pos / self.timeline_widget.pixels_per_ms
        
        # Find track before and after gap_ms
        prev_seg = None
        next_seg = None
        
        sorted_segs = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        for s in sorted_segs:
            if s.start_ms + s.duration_ms <= gap_ms:
                prev_seg = s
            elif s.start_ms >= gap_ms:
                if next_seg is None: next_seg = s
        
        if not prev_seg or not next_seg:
            self.status_bar.showMessage("Need a track before AND after the gap to find a bridge.")
            return

        self.loading_overlay.show_loading(f"AI: Finding bridge between '{prev_seg.filename[:15]}' and '{next_seg.filename[:15]}'...")
        
        try:
            conn = self.dm.get_conn()
            conn.row_factory = sqlite3_factory
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE id NOT IN (?, ?)", (prev_seg.id, next_seg.id))
            candidates = cursor.fetchall()
            
            p_emb = self.dm.get_embedding(prev_seg.id) # Placeholder, might need real embedding logic
            n_emb = self.dm.get_embedding(next_seg.id)
            
            results = []
            for c in candidates:
                c_dict = dict(c)
                c_emb = self.dm.get_embedding(c_dict['clp_embedding_id']) if c_dict['clp_embedding_id'] else None
                score = self.scorer.calculate_bridge_score(prev_seg.__dict__, next_seg.__dict__, c_dict, c_emb=c_emb)
                results.append((score, c_dict))
            
            results.sort(key=lambda x: x[0], reverse=True)
            
            # Show bridge results in the Recommendation list
            self.rec_list.setRowCount(0)
            for score, o_track in results[:15]:
                ri = self.rec_list.rowCount(); self.rec_list.insertRow(ri)
                si = QTableWidgetItem(f"{score}% (BRIDGE)")
                si.setData(Qt.ItemDataRole.UserRole, o_track['id'])
                si.setToolTip(f"This track has high compatibility with BOTH surrounding segments.")
                self.rec_list.setItem(ri, 0, si)
                self.rec_list.setItem(ri, 1, QTableWidgetItem(o_track['filename']))
            
            self.loading_overlay.hide_loading()
            self.status_bar.showMessage(f"AI found {len(results)} potential bridges.")
            conn.close()
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Bridge Error", "AI Bridge search failed.", e)

    def on_bpm_changed(self, text):
        try: self.timeline_widget.target_bpm = float(text); self.timeline_widget.update(); self.update_status()
        except: pass
    def toggle_analytics(self):
        self.timeline_widget.show_modifications = not self.mod_toggle.isChecked()
        self.mod_toggle.setText("🔍 Show Markers" if self.mod_toggle.isChecked() else "🔍 Hide Markers"); self.timeline_widget.update()
    def update_status(self):
        count = len(self.timeline_widget.segments)
        if count > 0:
            total_dur = max(s.start_ms + s.duration_ms for s in self.timeline_widget.segments)
            avg_bpm = sum(s.bpm for s in self.timeline_widget.segments) / count
            bpm_diff = abs(avg_bpm - self.timeline_widget.target_bpm)
            self.status_bar.showMessage(f"Timeline: {count} tracks | Total Duration: {total_dur/1000:.1f}s")
            
            at = (f"<b>Tracks:</b> {count}<br><b>Duration:</b> {total_dur/1000:.1f}s<br><b>Avg Track BPM:</b> {avg_bpm:.1f}<br><b>BPM Variance:</b> {bpm_diff:.1f}<br>")
            if bpm_diff > 10: at += "<br><span style='color: #ffaa00;'>⚠️ Large BPM stretch active</span>"
            
            # Harmonic Compatibility Check for Selected
            if self.timeline_widget.selected_segment:
                sel = self.timeline_widget.selected_segment
                at += f"<br><b>Selected Key:</b> {sel.key}"
                # Find overlaps or neighbors
                for other in self.timeline_widget.segments:
                    if other == sel: continue
                    # Simple check: if they are in different lanes but overlap
                    overlap = max(sel.start_ms, other.start_ms) < min(sel.start_ms + sel.duration_ms, other.start_ms + other.duration_ms)
                    if overlap:
                        h_score = self.scorer.calculate_harmonic_score(sel.key, other.key)
                        color = "#00ff66" if h_score >= 100 else "#ccff00" if h_score >= 80 else "#ff5555"
                        at += f"<br>Overlap with '{other.filename[:10]}...': <span style='color: {color};'>{h_score}% Match</span>"
            
            self.stats_label.setText(at)
        else: self.status_bar.showMessage("Ready."); self.stats_label.setText("Timeline empty")
    def load_library(self):
        try:
            conn = self.dm.get_conn(); cursor = conn.cursor(); cursor.execute("SELECT id, filename, bpm, harmonic_key FROM tracks")
            rows = cursor.fetchall(); self.library_table.setRowCount(0)
            for row in rows:
                ri = self.library_table.rowCount(); self.library_table.insertRow(ri)
                ni = QTableWidgetItem(row[1]); ni.setData(Qt.ItemDataRole.UserRole, row[0]); self.library_table.setItem(ri, 0, ni)
                self.library_table.setItem(ri, 1, QTableWidgetItem(f"{row[2]:.1f}")); self.library_table.setItem(ri, 2, QTableWidgetItem(row[3]))
            conn.close()
        except Exception as e: show_error(self, "Library Error", "Failed to load library.", e)
    def on_library_track_selected(self):
        si = self.library_table.selectedItems()
        if not si: return
        tid = self.library_table.item(si[0].row(), 0).data(Qt.ItemDataRole.UserRole); self.add_track_by_id(tid, only_update_recs=True)
    def add_track_by_id(self, track_id, x_pos=None, only_update_recs=False, lane=0):
        try:
            conn = self.dm.get_conn(); conn.row_factory = sqlite3_factory; cursor = conn.cursor(); cursor.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
            track = dict(cursor.fetchone()); conn.close()
            if not only_update_recs:
                start_ms = x_pos / self.timeline_widget.pixels_per_ms if x_pos is not None else None
                seg = self.timeline_widget.add_track(track, start_ms=start_ms)
                if x_pos is not None: seg.lane = lane
                seg.waveform = self.processor.get_waveform_envelope(track['file_path']); self.timeline_widget.update()
            self.selected_library_track = track; self.update_recommendations(track_id)
        except Exception as e: show_error(self, "Data Error", "Failed to retrieve track.", e)
    def add_selected_to_timeline(self):
        if self.selected_library_track: self.add_track_by_id(self.selected_library_track['id'])
    def on_rec_double_clicked(self, item):
        tid = self.rec_list.item(item.row(), 0).data(Qt.ItemDataRole.UserRole); self.add_track_by_id(tid)
    def auto_populate_timeline(self):
        if not self.selected_library_track: return
        self.loading_overlay.show_loading()
        sequence = self.orchestrator.find_curated_sequence(max_tracks=6, seed_track=self.selected_library_track)
        if sequence:
            self.timeline_widget.segments = []
            curr_ms = 0
            for i, track in enumerate(sequence):
                duration = 20000 if i % 2 == 0 else 30000
                seg = self.timeline_widget.add_track(track, start_ms=curr_ms)
                seg.waveform = self.processor.get_waveform_envelope(track['file_path'])
                curr_ms += duration - 8000 
            self.timeline_widget.update_geometry()
        self.loading_overlay.hide_loading()
    def on_segment_selected(self, segment):
        if segment: self.status_bar.showMessage(f"Selected: {segment.filename} | Shift+Drag vertically for Volume | Right-Click for Split/Primary")
        else: self.update_status()
    def render_timeline(self):
        if not self.timeline_widget.segments: return
        sorted_segs = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        try: target_bpm = float(self.target_bpm_edit.text())
        except: target_bpm = 124.0
        self.loading_overlay.show_loading("Rendering your journey...")
        try:
            output_file = "timeline_mix.mp3"
            render_data = [{'file_path': s.file_path, 'start_ms': int(s.start_ms), 'duration_ms': int(s.duration_ms), 'bpm': s.bpm, 'volume': s.volume, 'offset_ms': int(s.offset_ms), 'is_primary': s.is_primary, 'fade_in_ms': s.fade_in_ms, 'fade_out_ms': s.fade_out_ms} for s in sorted_segs]
            self.renderer.render_timeline(render_data, output_file, target_bpm=target_bpm)
            self.loading_overlay.hide_loading()
            QMessageBox.information(self, "Success", f"Mix rendered: {output_file}"); os.startfile(output_file)
        except Exception as e: self.loading_overlay.hide_loading(); show_error(self, "Render Error", "Failed to render timeline.", e)
    def scan_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Music Folder")
        if folder:
            self.loading_overlay.show_loading("Scanning audio files...")
            try:
                from src.ingestion import IngestionEngine
                engine = IngestionEngine(db_path=self.dm.db_path); engine.scan_directory(folder); self.load_library(); self.loading_overlay.hide_loading()
            except Exception as e: self.loading_overlay.hide_loading(); show_error(self, "Scan Error", "Failed to scan folder.", e)
    def run_embedding(self):
        self.loading_overlay.show_loading("AI Semantic Indexing... (this takes a moment)")
        try:
            from src.embeddings import EmbeddingEngine
            embed_engine = EmbeddingEngine(); conn = self.dm.get_conn(); cursor = conn.cursor(); cursor.execute("SELECT id, file_path, clp_embedding_id FROM tracks")
            tracks = cursor.fetchall()
            for track_id, file_path, existing_embed in tracks:
                if not existing_embed:
                    embedding = embed_engine.get_embedding(file_path); self.dm.add_embedding(track_id, embedding, metadata={"file_path": file_path})
            conn.close(); self.loading_overlay.hide_loading(); QMessageBox.information(self, "AI Complete", "Semantic indexing finished!")
        except Exception as e: self.loading_overlay.hide_loading(); show_error(self, "AI Error", "Semantic indexing failed.", e)
    def update_recommendations(self, track_id):
        try:
            conn = self.dm.get_conn(); conn.row_factory = sqlite3_factory; cursor = conn.cursor(); cursor.execute("SELECT * FROM tracks WHERE id = ?",(track_id,))
            target = dict(cursor.fetchone()); target_emb = self.dm.get_embedding(target['clp_embedding_id']) if target['clp_embedding_id'] else None
            cursor.execute("SELECT * FROM tracks WHERE id != ?", (track_id,)); others = cursor.fetchall(); results = []
            for other in others:
                other_dict = dict(other); other_emb = self.dm.get_embedding(other_dict['clp_embedding_id']) if other_dict['clp_embedding_id'] else None
                score_data = self.scorer.get_total_score(target, other_dict, target_emb, other_emb)
                results.append((score_data, other_dict))
            
            results.sort(key=lambda x: x[0]['total'], reverse=True); self.rec_list.setRowCount(0)
            
            for score, o_track in results[:15]:
                ri = self.rec_list.rowCount(); self.rec_list.insertRow(ri)
                
                # Score Item with Tooltip
                si = QTableWidgetItem(f"{score['total']}%")
                si.setData(Qt.ItemDataRole.UserRole, o_track['id'])
                tooltip = (f"Breakdown:\n"
                           f"• BPM Match: {score['bpm_score']}%\n"
                           f"• Harmonic Fit: {score['harmonic_score']}%\n"
                           f"• Semantic Vibe: {score['semantic_score']}%")
                si.setToolTip(tooltip)
                self.rec_list.setItem(ri, 0, si)
                
                # Name Item with Harmonic Color
                ni = QTableWidgetItem(o_track['filename'])
                if score['harmonic_score'] >= 100: ni.setForeground(QBrush(QColor(0, 255, 100))) # Perfect Match
                elif score['harmonic_score'] >= 80: ni.setForeground(QBrush(QColor(200, 255, 0))) # Adjacent
                self.rec_list.setItem(ri, 1, ni)
            conn.close()
        except Exception as e: print(f"Rec Engine Error: {e}")
    def play_selected(self):
        if self.selected_library_track:
            try: os.startfile(self.selected_library_track['file_path'])
            except Exception as e: show_error(self, "Playback Error", "Failed to play file.", e)

def sqlite3_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description): d[col[0]] = row[idx]
    return d

if __name__ == "__main__":
    app = QApplication(sys.argv); window = AudioSequencerApp(); window.show(); sys.exit(app.exec())
