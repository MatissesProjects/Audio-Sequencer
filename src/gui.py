import sys
import os
import sqlite3
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTableWidget, QTableWidgetItem, 
                             QLineEdit, QLabel, QPushButton, QFrame, QMessageBox,
                             QScrollArea, QMenu, QDialog, QTextEdit, QStatusBar)
from PyQt6.QtCore import Qt, QSize, QRect, pyqtSignal, QPoint, QMimeData
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont, QDrag

# Project Imports
from src.database import DataManager
from src.scoring import CompatibilityScorer
from src.processor import AudioProcessor
from src.renderer import FlowRenderer
from src.generator import TransitionGenerator
from src.orchestrator import FullMixOrchestrator

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
    def __init__(self, track_data, start_ms=0, duration_ms=20000, lane=0):
        self.id = track_data['id']
        self.filename = track_data['filename']
        self.file_path = track_data['file_path']
        self.bpm = track_data['bpm']
        self.key = track_data['harmonic_key']
        self.start_ms = start_ms
        self.duration_ms = duration_ms
        self.volume = 1.0 
        self.lane = lane
        self.is_primary = False
        self.color = QColor(70, 130, 180, 200) # SteelBlue

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
        self.drag_start_pos = None
        self.drag_start_ms = 0
        self.drag_start_dur = 0
        self.drag_start_vol = 1.0
        self.drag_start_lane = 0
        
        self.lane_height = 120
        self.lane_spacing = 10
        self.snap_threshold_ms = 2000 
        self.target_bpm = 124.0
        self.show_modifications = True
        self.update_geometry()

    def update_geometry(self):
        max_ms = 600000 # 10 mins default
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
        for i in range(0, 3600000, 10000): # Up to 1 hour
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
                painter.setPen(QPen(QColor(255, 215, 0), 3)) # Gold for primary
            else:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(200, 200, 200), 1))
            
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(rect.adjusted(8, 8, -8, -8), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, seg.filename)
            
            # Badges
            badge_y = rect.bottom() - 22
            badge_x = rect.left() + 8
            
            if seg.is_primary:
                painter.setBrush(QBrush(QColor(255, 215, 0))) # Gold
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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            clicked_seg = None
            for seg in reversed(self.segments):
                if self.get_seg_rect(seg).contains(event.pos()):
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
                if event.pos().x() > (rect.right() - 20):
                    self.resizing = True
                elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.vol_dragging = True
                else:
                    self.dragging = True
            
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
                menu.addSeparator()
                del_action = menu.addAction("🗑 Remove Track")
                
                action = menu.exec(self.mapToGlobal(event.pos()))
                if action == primary_action:
                    target_seg.is_primary = not target_seg.is_primary
                elif action == split_action:
                    self.split_segment(target_seg, event.pos().x())
                elif action == del_action:
                    self.segments.remove(target_seg)
                    if self.selected_segment == target_seg: self.selected_segment = None
                
                self.update_geometry()
                self.timelineChanged.emit()

    def split_segment(self, seg, x_pos):
        split_ms = x_pos / self.pixels_per_ms
        relative_split = split_ms - seg.start_ms
        
        if relative_split < 500 or relative_split > (seg.duration_ms - 500):
            return 
            
        new_dur = seg.duration_ms - relative_split
        seg.duration_ms = relative_split 
        
        track_data = {
            'id': seg.id, 'filename': seg.filename, 'file_path': seg.file_path,
            'bpm': seg.bpm, 'harmonic_key': seg.key
        }
        new_seg = TrackSegment(track_data, start_ms=split_ms, duration_ms=new_dur, lane=seg.lane)
        new_seg.volume = seg.volume
        new_seg.is_primary = seg.is_primary
        
        self.segments.append(new_seg)
        self.update_geometry()

    def mouseMoveEvent(self, event):
        if not self.selected_segment: return
            
        delta_x = event.pos().x() - self.drag_start_pos.x()
        delta_y = event.pos().y() - self.drag_start_pos.y()
        
        if self.resizing:
            delta_ms = delta_x / self.pixels_per_ms
            self.selected_segment.duration_ms = max(1000, self.drag_start_dur + delta_ms)
        elif self.vol_dragging:
            delta_vol = -delta_y / 150.0 
            self.selected_segment.volume = max(0.0, min(1.5, self.drag_start_vol + delta_vol))
        elif self.dragging:
            delta_ms = delta_x / self.pixels_per_ms
            new_start = max(0, self.drag_start_ms + delta_ms)
            
            for other in self.segments:
                if other == self.selected_segment: continue
                other_end = other.start_ms + other.duration_ms
                if abs(new_start - other_end) < self.snap_threshold_ms: new_start = other_end
                elif abs(new_start - other.start_ms) < self.snap_threshold_ms: new_start = other.start_ms
            
            self.selected_segment.start_ms = new_start
            new_lane = int((event.pos().y() - 40) // (self.lane_height + self.lane_spacing))
            self.selected_segment.lane = max(0, min(4, new_lane))
            
        self.update_geometry()
        self.timelineChanged.emit()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.resizing = False
        self.vol_dragging = False
        self.update_geometry()

    def add_track(self, track_data, start_ms=None):
        lane = 0
        if start_ms is None:
            if self.segments:
                last_seg = max(self.segments, key=lambda s: s.start_ms + s.duration_ms)
                start_ms = last_seg.start_ms + last_seg.duration_ms - 5000 
                lane = last_seg.lane
            else:
                start_ms = 0
        
        new_seg = TrackSegment(track_data, start_ms=start_ms, lane=lane)
        self.segments.append(new_seg)
        self.selected_segment = new_seg
        self.update_geometry()
        self.timelineChanged.emit()
        return new_seg

    def dragEnterEvent(self, event):
        if event.mimeData().hasText(): event.acceptProposedAction()

    def dropEvent(self, event):
        track_id = event.mimeData().text()
        if track_id:
            pos = event.position()
            lane = int((pos.y() - 40) // (self.lane_height + self.lane_spacing))
            lane = max(0, min(4, lane))
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
        self.selected_library_track = None
        self.init_ui()
        self.load_library()
        self.loading_overlay = LoadingOverlay(self.centralWidget())

    def init_ui(self):
        self.setWindowTitle("AudioSequencer AI - The Pro Flow")
        self.setMinimumSize(QSize(1400, 950))
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_panels = QHBoxLayout()
        
        library_panel = QFrame()
        library_panel.setFixedWidth(450)
        lib_layout = QVBoxLayout(library_panel)
        lib_layout.addWidget(QLabel("<h2>📁 Audio Library</h2>"))
        
        lib_actions = QHBoxLayout()
        self.scan_btn = QPushButton("📂 Scan Folder")
        self.scan_btn.setToolTip("Select a folder to import audio loops into your library.")
        self.scan_btn.clicked.connect(self.scan_folder)
        
        self.embed_btn = QPushButton("🧠 AI Index")
        self.embed_btn.setToolTip("Run AI analysis on new tracks to enable 'vibe-based' recommendations.")
        self.embed_btn.clicked.connect(self.run_embedding)
        
        lib_actions.addWidget(self.scan_btn)
        lib_actions.addWidget(self.embed_btn)
        lib_layout.addLayout(lib_actions)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("🔍 Semantic Search (e.g. 'lofi beat', 'cinematic')")
        self.search_bar.setToolTip("Search your library by 'vibe' using AI semantic matching.")
        lib_layout.addWidget(self.search_bar)
        
        self.library_table = DraggableTable(0, 3)
        self.library_table.setHorizontalHeaderLabels(["Track Name", "BPM", "Key"])
        self.library_table.setColumnWidth(0, 250)
        self.library_table.setColumnWidth(1, 60)
        self.library_table.setColumnWidth(2, 60)
        self.library_table.itemSelectionChanged.connect(self.on_library_track_selected)
        lib_layout.addWidget(self.library_table)
        top_panels.addWidget(library_panel)

        mid_panel = QFrame()
        mid_panel.setFixedWidth(250)
        mid_layout = QVBoxLayout(mid_panel)
        
        analytics_group = QFrame()
        analytics_group.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 5px;")
        analytics_layout = QVBoxLayout(analytics_group)
        analytics_layout.addWidget(QLabel("<h3>📊 Analytics Board</h3>"))
        
        self.mod_toggle = QPushButton("🔍 Hide Markers")
        self.mod_toggle.setCheckable(True)
        self.mod_toggle.setChecked(False)
        self.mod_toggle.clicked.connect(self.toggle_analytics)
        analytics_layout.addWidget(self.mod_toggle)
        
        self.stats_label = QLabel("Timeline empty")
        self.stats_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.stats_label.setWordWrap(True)
        analytics_layout.addWidget(self.stats_label)
        
        mid_layout.addWidget(analytics_group)
        mid_layout.addSpacing(10)

        mid_layout.addWidget(QLabel("<h3>⚡ Actions</h3>"))
        self.add_to_timeline_btn = QPushButton("➕ Add to Timeline")
        self.add_to_timeline_btn.setToolTip("Place the selected track at the end of your timeline.")
        self.add_to_timeline_btn.setStyleSheet("background-color: #2e7d32; color: white;")
        self.add_to_timeline_btn.clicked.connect(self.add_selected_to_timeline)
        mid_layout.addWidget(self.add_to_timeline_btn)
        
        self.play_btn = QPushButton("▶ Preview")
        self.play_btn.setToolTip("Play the raw audio file.")
        self.play_btn.clicked.connect(self.play_selected)
        mid_layout.addWidget(self.play_btn)
        
        mid_layout.addStretch()
        
        help_box = QLabel("<b>💡 Pro Tips:</b><br><br>"
                          "• <b>Drag</b> clips between lanes to overlap.<br>"
                          "• <b>Shift + Drag</b> vertically for volume.<br>"
                          "• <b>Right-Click</b> to Split clips or mark Primary.<br>"
                          "• <b>Horizontal Scroll</b> enabled for long sessions.")
        help_box.setWordWrap(True)
        help_box.setStyleSheet("color: #888; font-size: 11px; background: #1a1a1a; padding: 10px; border-radius: 5px;")
        mid_layout.addWidget(help_box)
        top_panels.addWidget(mid_panel)

        rec_panel = QFrame()
        rec_panel.setFixedWidth(450)
        rec_layout = QVBoxLayout(rec_panel)
        rec_layout.addWidget(QLabel("<h3>✨ Smart Suggestions</h3>"))
        
        self.rec_list = DraggableTable(0, 2)
        self.rec_list.setHorizontalHeaderLabels(["Match %", "Track"])
        self.rec_list.setColumnWidth(0, 80)
        self.rec_list.setColumnWidth(1, 300)
        self.rec_list.itemDoubleClicked.connect(self.on_rec_double_clicked)
        rec_layout.addWidget(self.rec_list)
        top_panels.addWidget(rec_panel)
        
        main_layout.addLayout(top_panels, stretch=1)

        timeline_header = QHBoxLayout()
        timeline_header.addWidget(QLabel("<h2>🎞 Timeline Journey</h2>"))
        
        self.auto_gen_btn = QPushButton("🪄 Auto-Generate Path")
        self.auto_gen_btn.setToolTip("Let AI build a 6-track journey based on your currently selected track.")
        self.auto_gen_btn.clicked.connect(self.auto_populate_timeline)
        timeline_header.addWidget(self.auto_gen_btn)
        
        timeline_header.addWidget(QLabel("Target BPM:"))
        self.target_bpm_edit = QLineEdit("124")
        self.target_bpm_edit.setFixedWidth(60)
        self.target_bpm_edit.textChanged.connect(self.on_bpm_changed)
        timeline_header.addWidget(self.target_bpm_edit)
        
        self.render_btn = QPushButton("🚀 RENDER FINAL MIX")
        self.render_btn.setToolTip("Stitch and synchronize all tracks into a high-quality MP3.")
        self.render_btn.setStyleSheet("background-color: #007acc; padding: 12px 25px; font-size: 16px; color: white; font-weight: bold;")
        self.render_btn.clicked.connect(self.render_timeline)
        timeline_header.addWidget(self.render_btn)
        
        timeline_header.addStretch()
        main_layout.addLayout(timeline_header)

        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_widget = TimelineWidget()
        self.timeline_scroll.setWidget(self.timeline_widget)
        main_layout.addWidget(self.timeline_scroll, stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

        self.timeline_widget.segmentSelected.connect(self.on_segment_selected)
        self.timeline_widget.timelineChanged.connect(self.update_status)

        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            QLabel { color: #ffffff; }
            QTableWidget { background-color: #1e1e1e; color: #e0e0e0; gridline-color: #333333; selection-background-color: #333333; border: 1px solid #333; }
            QLineEdit { background-color: #252525; color: #ffffff; border: 1px solid #444; padding: 8px; border-radius: 4px; }
            QPushButton { background-color: #333333; color: #ffffff; padding: 10px; border-radius: 6px; font-weight: bold; border: 1px solid #444; }
            QPushButton:hover { background-color: #444444; border: 1px solid #666; }
            QScrollArea { border: 1px solid #333; background-color: #121212; border-radius: 8px; }
            QStatusBar { background-color: #1e1e1e; color: #aaa; }
        """)

    def on_bpm_changed(self, text):
        try:
            self.timeline_widget.target_bpm = float(text)
            self.timeline_widget.update()
            self.update_status()
        except: pass

    def toggle_analytics(self):
        self.timeline_widget.show_modifications = not self.mod_toggle.isChecked()
        self.mod_toggle.setText("🔍 Show Markers" if self.mod_toggle.isChecked() else "🔍 Hide Markers")
        self.timeline_widget.update()

    def update_status(self):
        count = len(self.timeline_widget.segments)
        if count > 0:
            total_dur = max(s.start_ms + s.duration_ms for s in self.timeline_widget.segments)
            avg_bpm = sum(s.bpm for s in self.timeline_widget.segments) / count
            bpm_diff = abs(avg_bpm - self.timeline_widget.target_bpm)
            
            self.status_bar.showMessage(f"Timeline: {count} tracks | Total Duration: {total_dur/1000:.1f}s")
            
            analytics_text = (f"<b>Tracks:</b> {count}<br>"
                              f"<b>Duration:</b> {total_dur/1000:.1f}s<br>"
                              f"<b>Avg Track BPM:</b> {avg_bpm:.1f}<br>"
                              f"<b>BPM Variance:</b> {bpm_diff:.1f}<br>")
            
            if bpm_diff > 10: analytics_text += "<br><span style='color: #ffaa00;'>⚠️ Large BPM stretch active</span>"
            self.stats_label.setText(analytics_text)
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
            for row in rows:
                row_idx = self.library_table.rowCount()
                self.library_table.insertRow(row_idx)
                name_item = QTableWidgetItem(row[1])
                name_item.setData(Qt.ItemDataRole.UserRole, row[0])
                self.library_table.setItem(row_idx, 0, name_item)
                self.library_table.setItem(row_idx, 1, QTableWidgetItem(f"{row[2]:.1f}"))
                self.library_table.setItem(row_idx, 2, QTableWidgetItem(row[3]))
            conn.close()
        except Exception as e: show_error(self, "Library Error", "Failed to load library.", e)

    def on_library_track_selected(self):
        selected_items = self.library_table.selectedItems()
        if not selected_items: return
        row = selected_items[0].row()
        track_id = self.library_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.add_track_by_id(track_id, only_update_recs=True)

    def add_track_by_id(self, track_id, x_pos=None, only_update_recs=False, lane=0):
        try:
            conn = self.dm.get_conn()
            conn.row_factory = sqlite3_factory
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
            track = dict(cursor.fetchone())
            conn.close()
            if not only_update_recs:
                # Calculate correct offset from scroll area if needed
                start_ms = x_pos / self.timeline_widget.pixels_per_ms if x_pos is not None else None
                seg = self.timeline_widget.add_track(track, start_ms=start_ms)
                if x_pos is not None: seg.lane = lane
            self.selected_library_track = track
            self.update_recommendations(track_id)
        except Exception as e: show_error(self, "Data Error", "Failed to retrieve track.", e)

    def add_selected_to_timeline(self):
        if self.selected_library_track: self.timeline_widget.add_track(self.selected_library_track)

    def on_rec_double_clicked(self, item):
        row = item.row()
        track_id = self.rec_list.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.add_track_by_id(track_id)

    def auto_populate_timeline(self):
        if not self.selected_library_track: return
        self.loading_overlay.show_loading()
        sequence = self.orchestrator.find_curated_sequence(max_tracks=6, seed_track=self.selected_library_track)
        if sequence:
            self.timeline_widget.segments = []
            curr_ms = 0
            for i, track in enumerate(sequence):
                duration = 20000 if i % 2 == 0 else 30000
                self.timeline_widget.add_track(track, start_ms=curr_ms)
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
            render_data = [{'file_path': s.file_path, 'start_ms': int(s.start_ms), 'duration_ms': int(s.duration_ms), 'bpm': s.bpm, 'volume': s.volume} for s in sorted_segs]
            self.renderer.render_timeline(render_data, output_file, target_bpm=target_bpm)
            self.loading_overlay.hide_loading()
            QMessageBox.information(self, "Success", f"Mix rendered: {output_file}")
            os.startfile(output_file)
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Render Error", "Failed to render timeline.", e)

    def scan_folder(self):
        from PyQt6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "Select Music Folder")
        if folder:
            self.loading_overlay.show_loading("Scanning audio files...")
            try:
                from src.ingestion import IngestionEngine
                engine = IngestionEngine(db_path=self.dm.db_path)
                engine.scan_directory(folder)
                self.load_library()
                self.loading_overlay.hide_loading()
            except Exception as e:
                self.loading_overlay.hide_loading()
                show_error(self, "Scan Error", "Failed to scan folder.", e)

    def run_embedding(self):
        self.loading_overlay.show_loading("AI Semantic Indexing... (this takes a moment)")
        try:
            from src.embeddings import EmbeddingEngine
            embed_engine = EmbeddingEngine()
            conn = self.dm.get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id, file_path, clp_embedding_id FROM tracks")
            tracks = cursor.fetchall()
            for track_id, file_path, existing_embed in tracks:
                if not existing_embed:
                    embedding = embed_engine.get_embedding(file_path)
                    self.dm.add_embedding(track_id, embedding, metadata={"file_path": file_path})
            conn.close()
            self.loading_overlay.hide_loading()
            QMessageBox.information(self, "AI Complete", "Semantic indexing finished!")
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "AI Error", "Semantic indexing failed.", e)

    def update_recommendations(self, track_id):
        try:
            conn = self.dm.get_conn()
            conn.row_factory = sqlite3_factory
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
            target = dict(cursor.fetchone())
            target_emb = self.dm.get_embedding(target['clp_embedding_id']) if target['clp_embedding_id'] else None
            cursor.execute("SELECT * FROM tracks WHERE id != ?", (track_id,))
            others = cursor.fetchall()
            results = []
            for other in others:
                other_dict = dict(other)
                other_emb = self.dm.get_embedding(other_dict['clp_embedding_id']) if other_dict['clp_embedding_id'] else None
                score = self.scorer.get_total_score(target, other_dict, target_emb, other_emb)
                results.append((score['total'], other_dict['filename'], other_dict['id']))
            results.sort(key=lambda x: x[0], reverse=True)
            self.rec_list.setRowCount(0)
            for score, name, tid in results[:10]:
                row_idx = self.rec_list.rowCount()
                self.rec_list.insertRow(row_idx)
                score_item = QTableWidgetItem(f"{score}%")
                score_item.setData(Qt.ItemDataRole.UserRole, tid)
                self.rec_list.setItem(row_idx, 0, score_item)
                self.rec_list.setItem(row_idx, 1, QTableWidgetItem(name))
            conn.close()
        except Exception as e: print(f"Rec Engine Error: {e}")

    def play_selected(self):
        if self.selected_library_track:
            try: os.startfile(self.selected_library_track['file_path'])
            except Exception as e: show_error(self, "Playback Error", "Failed to play file.", e)

def sqlite3_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioSequencerApp()
    window.show()
    sys.exit(app.exec())
