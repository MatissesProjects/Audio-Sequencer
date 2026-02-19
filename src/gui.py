import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTableWidget, QTableWidgetItem, 
                             QLineEdit, QLabel, QPushButton, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, QSize
from src.database import DataManager
from src.scoring import CompatibilityScorer
from src.processor import AudioProcessor
from src.renderer import FlowRenderer

class AudioSequencerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dm = DataManager()
        self.scorer = CompatibilityScorer()
        self.processor = AudioProcessor()
        self.renderer = FlowRenderer()
        self.selected_track = None
        self.init_ui()
        self.load_library()

    def init_ui(self):
        self.setWindowTitle("AudioSequencer AI - The Flow")
        self.setMinimumSize(QSize(1100, 750))
        
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 1. Left Sidebar: Library
        library_panel = QFrame()
        library_panel.setFixedWidth(400)
        lib_layout = QVBoxLayout(library_panel)
        
        lib_layout.addWidget(QLabel("<h2>Audio Library</h2>"))
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Semantic Search (e.g. 'ambient', 'heavy')")
        lib_layout.addWidget(self.search_bar)
        
        self.library_table = QTableWidget(0, 3)
        self.library_table.setHorizontalHeaderLabels(["Track Name", "BPM", "Key"])
        self.library_table.itemSelectionChanged.connect(self.on_track_selected)
        lib_layout.addWidget(self.library_table)
        
        main_layout.addWidget(library_panel)

        # 2. Center Canvas: The Flow
        canvas_panel = QFrame()
        canvas_panel.setFrameShape(QFrame.Shape.StyledPanel)
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.addWidget(QLabel("<h2>The Flow</h2>"), alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.active_track_label = QLabel("Drop a track here to start the flow")
        self.active_track_label.setStyleSheet("border: 2px dashed #666; padding: 40px; background-color: #222;")
        canvas_layout.addWidget(self.active_track_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Playback Controls
        controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Preview Selected")
        self.play_btn.clicked.connect(self.play_selected)
        self.mix_btn = QPushButton("🔀 Mix with Top Recommendation")
        self.mix_btn.clicked.connect(self.mix_with_top)
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.mix_btn)
        canvas_layout.addLayout(controls_layout)
        
        main_layout.addWidget(canvas_panel, stretch=2)

        # 3. Right Sidebar: Recommendations
        rec_panel = QFrame()
        rec_panel.setFixedWidth(300)
        rec_layout = QVBoxLayout(rec_panel)
        rec_layout.addWidget(QLabel("<h3>AI Recommendations</h3>"))
        
        self.rec_list = QTableWidget(0, 2)
        self.rec_list.setHorizontalHeaderLabels(["Match %", "Track"])
        rec_layout.addWidget(self.rec_list)
        
        main_layout.addWidget(rec_panel)

        # Apply basic dark styling
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #ffffff; }
            QTableWidget { background-color: #2d2d2d; color: #ffffff; gridline-color: #3d3d3d; }
            QLineEdit { background-color: #3d3d3d; color: #ffffff; border: 1px solid #555; padding: 5px; }
            QPushButton { background-color: #007acc; color: #ffffff; padding: 10px; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #005fa3; }
        """)

    def load_library(self):
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

    def on_track_selected(self):
        selected_items = self.library_table.selectedItems()
        if not selected_items:
            return
            
        row = selected_items[0].row()
        track_id = self.library_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        
        # Get full track info
        conn = self.dm.get_conn()
        conn.row_factory = sqlite3_factory
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
        self.selected_track = dict(cursor.fetchone())
        conn.close()
        
        self.active_track_label.setText(f"Selected: {self.selected_track['filename']}\n{self.selected_track['bpm']} BPM | Key: {self.selected_track['harmonic_key']}")
        self.update_recommendations(track_id)

    def play_selected(self):
        if not self.selected_track:
            return
        os.startfile(self.selected_track['file_path']) # Simple OS-level playback

    def mix_with_top(self):
        if not self.selected_track or self.rec_list.rowCount() == 0:
            QMessageBox.warning(self, "No Match", "Select a track with recommendations first.")
            return
        
        # Get the top recommendation filename
        top_name = self.rec_list.item(0, 1).text()
        
        conn = self.dm.get_conn()
        conn.row_factory = sqlite3_factory
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks WHERE filename = ?", (top_name,))
        t2 = dict(cursor.fetchone())
        conn.close()
        
        t1 = self.selected_track
        
        QMessageBox.information(self, "Mixing", f"Synchronizing {t2['filename']} to {t1['bpm']} BPM...")
        
        try:
            stretched_path = "gui_temp_stretched.wav"
            self.processor.stretch_to_bpm(t2['file_path'], t2['bpm'], t1['bpm'], stretched_path)
            
            final_mix = "gui_final_mix.wav"
            self.renderer.mix_tracks(t1['file_path'], stretched_path, final_mix)
            
            if os.path.exists(stretched_path):
                os.remove(stretched_path)
                
            QMessageBox.information(self, "Success", f"Mix ready: {final_mix}\nOpening file...")
            os.startfile(final_mix)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Mixing failed: {e}")

    def update_recommendations(self, track_id):
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
