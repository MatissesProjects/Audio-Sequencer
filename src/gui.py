import sys
import os
import sqlite3
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTableWidget, QTableWidgetItem, 
                             QLineEdit, QLabel, QPushButton, QFrame, QMessageBox,
                             QScrollArea, QMenu, QDialog, QTextEdit)
from PyQt6.QtCore import Qt, QSize, QRect, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont

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
    def __init__(self, track_data, start_ms=0, duration_ms=20000):
        self.id = track_data['id']
        self.filename = track_data['filename']
        self.file_path = track_data['file_path']
        self.bpm = track_data['bpm']
        self.key = track_data['harmonic_key']
        self.start_ms = start_ms
        self.duration_ms = duration_ms
        self.volume = 1.0 # 0.0 to 1.0
        self.color = QColor(70, 130, 180, 200) # SteelBlue

class TimelineWidget(QWidget):
    segmentSelected = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.segments = []
        self.setMinimumHeight(300)
        self.setAcceptDrops(True)
        self.pixels_per_ms = 0.05 # 1 second = 50 pixels
        self.selected_segment = None
        self.dragging = False
        self.resizing = False
        self.vol_dragging = False
        self.drag_start_pos = None
        self.drag_start_ms = 0
        self.drag_start_dur = 0
        self.drag_start_vol = 1.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Draw time markings
        painter.setPen(QColor(60, 60, 60))
        for i in range(0, 1200000, 10000): # Up to 20 mins
            x = int(i * self.pixels_per_ms)
            painter.drawLine(x, 0, x, self.height())
            if i % 30000 == 0:
                painter.drawText(x + 5, 15, f"{i//1000}s")

        # Draw segments
        for seg in self.segments:
            x = int(seg.start_ms * self.pixels_per_ms)
            w = int(seg.duration_ms * self.pixels_per_ms)
            h = int(120 * seg.volume) # Volume affects height
            y = (self.height() - h) // 2
            
            rect = QRect(x, y, w, h)
            
            # Opacity based on volume
            color = QColor(seg.color)
            color.setAlpha(int(100 + 155 * seg.volume))
            
            if seg == self.selected_segment:
                painter.setBrush(QBrush(color.lighter(130)))
                painter.setPen(QPen(Qt.GlobalColor.white, 2))
            else:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(200, 200, 200), 1))
            
            painter.drawRoundedRect(rect, 5, 5)
            
            # Duration handles (subtle bars at edges)
            painter.setPen(QColor(255, 255, 255, 100))
            painter.drawLine(x + w - 5, y + 5, x + w - 5, y + h - 5)
            
            # Text info
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            painter.drawText(rect.adjusted(10, 10, -10, -10), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, seg.filename)
            painter.setFont(QFont("Arial", 8))
            painter.drawText(rect.adjusted(10, 30, -10, -10), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, f"{seg.bpm} BPM | {seg.key}\nVol: {int(seg.volume*100)}%")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            clicked_seg = None
            edge_click = False
            vol_click = False
            
            for seg in reversed(self.segments):
                x = int(seg.start_ms * self.pixels_per_ms)
                w = int(seg.duration_ms * self.pixels_per_ms)
                h = int(120 * seg.volume)
                y = (self.height() - h) // 2
                rect = QRect(x, y, w, h)
                
                if rect.contains(event.pos()):
                    clicked_seg = seg
                    # Check if near right edge for resize
                    if event.pos().x() > (x + w - 15):
                        edge_click = True
                    # Check if in middle for volume drag (Shift key or just vertical)
                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        vol_click = True
                    break
            
            self.selected_segment = clicked_seg
            self.segmentSelected.emit(clicked_seg)
            
            if self.selected_segment:
                self.drag_start_pos = event.pos()
                if edge_click:
                    self.resizing = True
                    self.drag_start_dur = self.selected_segment.duration_ms
                elif vol_click:
                    self.vol_dragging = True
                    self.drag_start_vol = self.selected_segment.volume
                else:
                    self.dragging = True
                    self.drag_start_ms = self.selected_segment.start_ms
            
            self.update()
        
        elif event.button() == Qt.MouseButton.RightButton:
            for seg in reversed(self.segments):
                x = int(seg.start_ms * self.pixels_per_ms)
                w = int(seg.duration_ms * self.pixels_per_ms)
                h = int(120 * seg.volume)
                y = (self.height() - h) // 2
                if QRect(x, y, w, h).contains(event.pos()):
                    menu = QMenu(self)
                    del_action = menu.addAction("Remove Track")
                    action = menu.exec(self.mapToGlobal(event.pos()))
                    if action == del_action:
                        self.segments.remove(seg)
                        if self.selected_segment == seg:
                            self.selected_segment = None
                        self.update()
                    break

    def mouseMoveEvent(self, event):
        if not self.selected_segment:
            return
            
        delta_x = event.pos().x() - self.drag_start_pos.x()
        delta_y = event.pos().y() - self.drag_start_pos.y()
        
        if self.resizing:
            delta_ms = delta_x / self.pixels_per_ms
            self.selected_segment.duration_ms = max(1000, self.drag_start_dur + delta_ms)
        elif self.vol_dragging:
            delta_vol = -delta_y / 100.0 # Up is louder
            self.selected_segment.volume = max(0.1, min(1.2, self.drag_start_vol + delta_vol))
        elif self.dragging:
            delta_ms = delta_x / self.pixels_per_ms
            self.selected_segment.start_ms = max(0, self.drag_start_ms + delta_ms)
            
        self.update()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.resizing = False
        self.vol_dragging = False

    def add_track(self, track_data, start_ms=None):
        if start_ms is None:
            # Place at the end of the last track
            if self.segments:
                last_seg = max(self.segments, key=lambda s: s.start_ms + s.duration_ms)
                start_ms = last_seg.start_ms + last_seg.duration_ms - 5000 # 5s default overlap
            else:
                start_ms = 0
        
        new_seg = TrackSegment(track_data, start_ms=start_ms)
        self.segments.append(new_seg)
        self.selected_segment = new_seg
        self.update()
        return new_seg

    def dragEnterEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        # We'll handle this from the main app's logic if needed, 
        # but for now let's just accept if it's from our library
        event.acceptProposedAction()

class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 150))
        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Processing... Please Wait")

    def show_loading(self):
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
        self.setMinimumSize(QSize(1300, 900))
        
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top Section: Library and Recommendations
        top_panels = QHBoxLayout()
        
        # 1. Left Sidebar: Library
        library_panel = QFrame()
        library_panel.setFixedWidth(400)
        lib_layout = QVBoxLayout(library_panel)
        lib_layout.addWidget(QLabel("<h2>Audio Library</h2>"))
        
        lib_actions = QHBoxLayout()
        self.scan_btn = QPushButton("📂 Scan Folder")
        self.scan_btn.clicked.connect(self.scan_folder)
        self.embed_btn = QPushButton("🧠 Run AI Embed")
        self.embed_btn.clicked.connect(self.run_embedding)
        lib_actions.addWidget(self.scan_btn)
        lib_actions.addWidget(self.embed_btn)
        lib_layout.addLayout(lib_actions)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Semantic Search (e.g. 'ambient', 'heavy')")
        lib_layout.addWidget(self.search_bar)
        
        self.library_table = QTableWidget(0, 3)
        self.library_table.setHorizontalHeaderLabels(["Track Name", "BPM", "Key"])
        self.library_table.itemSelectionChanged.connect(self.on_library_track_selected)
        lib_layout.addWidget(self.library_table)
        
        top_panels.addWidget(library_panel)

        # 2. Middle: Selected Track Actions
        actions_panel = QFrame()
        actions_panel.setFixedWidth(250)
        actions_layout = QVBoxLayout(actions_panel)
        actions_layout.addWidget(QLabel("<h3>Track Actions</h3>"))
        
        self.add_to_timeline_btn = QPushButton("➕ Add to Timeline")
        self.add_to_timeline_btn.setStyleSheet("background-color: #2e7d32; color: white;")
        self.add_to_timeline_btn.clicked.connect(self.add_selected_to_timeline)
        actions_layout.addWidget(self.add_to_timeline_btn)
        
        self.play_btn = QPushButton("▶ Preview Audio")
        self.play_btn.clicked.connect(self.play_selected)
        actions_layout.addWidget(self.play_btn)
        
        actions_layout.addStretch()
        top_panels.addWidget(actions_panel)

        # 3. Right Sidebar: Recommendations
        rec_panel = QFrame()
        rec_panel.setFixedWidth(350)
        rec_layout = QVBoxLayout(rec_panel)
        rec_layout.addWidget(QLabel("<h3>AI Recommendations</h3>"))
        
        self.rec_list = QTableWidget(0, 2)
        self.rec_list.setHorizontalHeaderLabels(["Match %", "Track"])
        self.rec_list.itemDoubleClicked.connect(self.on_rec_double_clicked)
        rec_layout.addWidget(self.rec_list)
        
        top_panels.addWidget(rec_panel)
        
        main_layout.addLayout(top_panels, stretch=1)

        # Bottom Section: Timeline
        timeline_header = QHBoxLayout()
        timeline_header.addWidget(QLabel("<h2>Timeline Editor</h2>"))
        
        self.auto_gen_btn = QPushButton("🪄 Auto-Generate Path (6 clips)")
        self.auto_gen_btn.clicked.connect(self.auto_populate_timeline)
        timeline_header.addWidget(self.auto_gen_btn)
        
        timeline_header.addWidget(QLabel("Target BPM:"))
        self.target_bpm_edit = QLineEdit("124")
        self.target_bpm_edit.setFixedWidth(50)
        timeline_header.addWidget(self.target_bpm_edit)
        
        self.render_btn = QPushButton("🚀 RENDER FINAL MIX")
        self.render_btn.setStyleSheet("background-color: #007acc; padding: 10px 20px; font-size: 16px; color: white;")
        self.render_btn.clicked.connect(self.render_timeline)
        timeline_header.addWidget(self.render_btn)
        
        timeline_header.addStretch()
        main_layout.addLayout(timeline_header)

        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_widget = TimelineWidget()
        self.timeline_scroll.setWidget(self.timeline_widget)
        main_layout.addWidget(self.timeline_scroll, stretch=1)

        # Segment Properties Panel
        self.prop_panel = QFrame()
        self.prop_panel.setFixedHeight(100)
        self.prop_panel.setStyleSheet("background-color: #252525; border-top: 1px solid #444;")
        prop_layout = QHBoxLayout(self.prop_panel)
        self.seg_label = QLabel("Select a segment to edit properties")
        prop_layout.addWidget(self.seg_label)
        
        # Duration edit
        prop_layout.addWidget(QLabel("Duration (ms):"))
        self.dur_edit = QLineEdit()
        self.dur_edit.setFixedWidth(80)
        self.dur_edit.editingFinished.connect(self.update_segment_duration)
        prop_layout.addWidget(self.dur_edit)
        
        prop_layout.addStretch()
        main_layout.addWidget(self.prop_panel)

        self.timeline_widget.segmentSelected.connect(self.on_segment_selected)

        # Apply basic dark styling
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #ffffff; }
            QTableWidget { background-color: #2d2d2d; color: #ffffff; gridline-color: #3d3d3d; selection-background-color: #444; }
            QLineEdit { background-color: #3d3d3d; color: #ffffff; border: 1px solid #555; padding: 5px; }
            QPushButton { background-color: #444; color: #ffffff; padding: 8px; border-radius: 5px; font-weight: bold; border: 1px solid #555; }
            QPushButton:hover { background-color: #555; }
            QScrollArea { border: none; background-color: #1e1e1e; }
        """)

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
                self.library_table.setItem(row_idx, 1, QTableWidgetItem(str(row[2])))
                self.library_table.setItem(row_idx, 2, QTableWidgetItem(row[3]))
            conn.close()
        except Exception as e:
            show_error(self, "Library Error", "Failed to load audio library.", e)

    def on_library_track_selected(self):
        selected_items = self.library_table.selectedItems()
        if not selected_items:
            return
            
        row = selected_items[0].row()
        track_id = self.library_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        
        conn = self.dm.get_conn()
        conn.row_factory = sqlite3_factory
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
        self.selected_library_track = dict(cursor.fetchone())
        conn.close()
        
        self.update_recommendations(track_id)

    def add_selected_to_timeline(self):
        if not self.selected_library_track:
            QMessageBox.warning(self, "No Track", "Select a track from the library first.")
            return
        self.timeline_widget.add_track(self.selected_library_track)

    def on_rec_double_clicked(self, item):
        row = item.row()
        name = self.rec_list.item(row, 1).text()
        conn = self.dm.get_conn()
        conn.row_factory = sqlite3_factory
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks WHERE filename = ?", (name,))
        track = dict(cursor.fetchone())
        conn.close()
        self.timeline_widget.add_track(track)

    def auto_populate_timeline(self):
        if not self.selected_library_track:
            QMessageBox.warning(self, "No Seed", "Select a starting track in the library first.")
            return
            
        sequence = self.orchestrator.find_curated_sequence(max_tracks=6, seed_track=self.selected_library_track)
        if sequence:
            self.timeline_widget.segments = []
            curr_ms = 0
            for i, track in enumerate(sequence):
                duration = 20000 if i % 2 == 0 else 30000
                self.timeline_widget.add_track(track, start_ms=curr_ms)
                curr_ms += duration - 8000 # 8s overlap
            self.timeline_widget.update()

    def on_segment_selected(self, segment):
        if segment:
            self.seg_label.setText(f"Editing: {segment.filename}")
            self.dur_edit.setText(str(segment.duration_ms))
        else:
            self.seg_label.setText("Select a segment to edit properties")
            self.dur_edit.clear()

    def update_segment_duration(self):
        if self.timeline_widget.selected_segment:
            try:
                new_dur = int(self.dur_edit.text())
                self.timeline_widget.selected_segment.duration_ms = new_dur
                self.timeline_widget.update()
            except ValueError:
                pass

    def render_timeline(self):
        if not self.timeline_widget.segments:
            QMessageBox.warning(self, "Empty", "Add some tracks to the timeline first.")
            return
            
        sorted_segs = sorted(self.timeline_widget.segments, key=lambda s: s.start_ms)
        
        try:
            target_bpm = float(self.target_bpm_edit.text())
        except ValueError:
            target_bpm = 124.0

        self.loading_overlay.show_loading()
        
        try:
            output_file = "timeline_mix.mp3"
            
            # Prepare data for renderer
            render_data = []
            for s in sorted_segs:
                render_data.append({
                    'file_path': s.file_path,
                    'start_ms': int(s.start_ms),
                    'duration_ms': int(s.duration_ms),
                    'bpm': s.bpm,
                    'volume': s.volume
                })
            
            self.renderer.render_timeline(render_data, output_file, target_bpm=target_bpm)
            self.loading_overlay.hide_loading()
            QMessageBox.information(self, "Success", f"Mix rendered to {output_file}")
            os.startfile(output_file)
            
        except Exception as e:
            self.loading_overlay.hide_loading()
            show_error(self, "Render Error", "An error occurred while rendering the timeline.", e)

    def scan_folder(self):
        from PyQt6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "Select Music Folder")
        if folder:
            self.loading_overlay.show_loading()
            try:
                from src.ingestion import IngestionEngine
                engine = IngestionEngine(db_path=self.dm.db_path)
                engine.scan_directory(folder)
                self.load_library()
                self.loading_overlay.hide_loading()
                QMessageBox.information(self, "Scan Complete", f"Library updated from {folder}")
            except Exception as e:
                self.loading_overlay.hide_loading()
                show_error(self, "Scan Error", "Failed to scan directory for audio files.", e)

    def run_embedding(self):
        self.loading_overlay.show_loading()
        from src.embeddings import EmbeddingEngine
        try:
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
            QMessageBox.information(self, "AI Complete", "All tracks have been semantic-indexed!")
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
                results.append((score['total'], other_dict['filename']))
                
            results.sort(key=lambda x: x[0], reverse=True)
            
            self.rec_list.setRowCount(0)
            for score, name in results[:10]:
                row_idx = self.rec_list.rowCount()
                self.rec_list.insertRow(row_idx)
                self.rec_list.setItem(row_idx, 0, QTableWidgetItem(f"{score}%"))
                self.rec_list.setItem(row_idx, 1, QTableWidgetItem(name))
                
            conn.close()
        except Exception as e:
            # We don't want a popup every time selection changes, but logging details is good
            print(f"Rec Engine Error: {e}")

    def play_selected(self):
        if not self.selected_library_track:
            return
        try:
            os.startfile(self.selected_library_track['file_path'])
        except Exception as e:
            show_error(self, "Playback Error", "Failed to play the selected file.", e)

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
