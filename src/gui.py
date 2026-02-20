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
from src.generator import TransitionGenerator
from src.orchestrator import FullMixOrchestrator

class AudioSequencerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dm = DataManager()
        self.scorer = CompatibilityScorer()
        self.processor = AudioProcessor()
        self.renderer = FlowRenderer()
        self.generator = TransitionGenerator()
        self.orchestrator = FullMixOrchestrator()
        self.selected_track = None
        self.init_ui()
        self.load_library()

    def init_ui(self):
        self.setWindowTitle("AudioSequencer AI - The Pro Flow")
        self.setMinimumSize(QSize(1200, 800))
        
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 1. Left Sidebar: Library
        library_panel = QFrame()
        library_panel.setFixedWidth(400)
        lib_layout = QVBoxLayout(library_panel)
        
        lib_layout.addWidget(QLabel("<h2>Audio Library</h2>"))
        
        # Action Buttons for Library
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
        self.library_table.itemSelectionChanged.connect(self.on_track_selected)
        lib_layout.addWidget(self.library_table)
        
        main_layout.addWidget(library_panel)

        # 2. Center Canvas: The Flow & Pro Orchestration
        canvas_panel = QFrame()
        canvas_panel.setFrameShape(QFrame.Shape.StyledPanel)
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.addWidget(QLabel("<h2>Master Orchestrator</h2>"), alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.active_track_label = QLabel("Select a 'Seed' track to begin")
        self.active_track_label.setStyleSheet("border: 2px dashed #666; padding: 40px; background-color: #222; font-size: 14px;")
        self.active_track_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_layout.addWidget(self.active_track_label)
        
        # Power Actions
        pro_group = QFrame()
        pro_group.setStyleSheet("background-color: #252525; border-radius: 10px; padding: 10px;")
        pro_layout = QVBoxLayout(pro_group)
        pro_layout.addWidget(QLabel("<b>Pro Orchestration Modes</b>"))

        self.full_mix_btn = QPushButton("🚀 Generate Curated Journey (6-Clip Path)")
        self.full_mix_btn.setStyleSheet("background-color: #007acc; padding: 15px; color: white;")
        self.full_mix_btn.clicked.connect(self.run_curated_journey)
        
        self.layered_btn = QPushButton("🌊 Create 2-Min Layered Foundation")
        self.layered_btn.setStyleSheet("background-color: #2e7d32; padding: 15px; color: white;")
        self.layered_btn.clicked.connect(self.run_layered_journey)
        
        pro_layout.addWidget(self.full_mix_btn)
        pro_layout.addWidget(self.layered_btn)
        canvas_layout.addWidget(pro_group)

        # Basic Playback Controls
        controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Preview Seed")
        self.play_btn.clicked.connect(self.play_selected)
        self.mix_btn = QPushButton("🔀 Smart Mix (Seed + Top Rec)")
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
            QTableWidget { background-color: #2d2d2d; color: #ffffff; gridline-color: #3d3d3d; selection-background-color: #444; }
            QLineEdit { background-color: #3d3d3d; color: #ffffff; border: 1px solid #555; padding: 5px; }
            QPushButton { background-color: #444; color: #ffffff; padding: 10px; border-radius: 5px; font-weight: bold; border: 1px solid #555; }
            QPushButton:hover { background-color: #555; }
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

    def scan_folder(self):
        from PyQt6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "Select Music Folder")
        if folder:
            from src.ingestion import IngestionEngine
            engine = IngestionEngine(db_path=self.dm.db_path)
            engine.scan_directory(folder)
            self.load_library()
            QMessageBox.information(self, "Scan Complete", f"Library updated from {folder}")

    def run_embedding(self):
        QMessageBox.information(self, "AI Processing", "Starting CLAP embedding engine. This may take a moment...")
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
            QMessageBox.information(self, "AI Complete", "All tracks have been semantic-indexed!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"AI Embedding failed: {e}")

    def on_track_selected(self):
        selected_items = self.library_table.selectedItems()
        if not selected_items:
            return
            
        row = selected_items[0].row()
        track_id = self.library_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        
        conn = self.dm.get_conn()
        conn.row_factory = sqlite3_factory
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
        self.selected_track = dict(cursor.fetchone())
        conn.close()
        
        self.active_track_label.setText(f"SEED TRACK: {self.selected_track['filename']}\n{self.selected_track['bpm']} BPM | Key: {self.selected_track['harmonic_key']}")
        self.update_recommendations(track_id)

    def run_curated_journey(self):
        if not self.selected_track:
            QMessageBox.warning(self, "No Seed", "Select a starting track first.")
            return
        
        QMessageBox.information(self, "Orchestrating", "Calculating high-compatibility path for 6 tracks...")
        try:
            out_file = "gui_curated_journey.mp3"
            self.orchestrator.generate_full_mix(output_path=out_file, target_bpm=124, seed_track=self.selected_track)
            QMessageBox.information(self, "Success", f"Journey Created: {out_file}")
            os.startfile(out_file)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Journey failed: {e}")

    def run_layered_journey(self):
        if not self.selected_track:
            QMessageBox.warning(self, "No Seed", "Select a foundation track first.")
            return
            
        QMessageBox.information(self, "Orchestrating", "Building 2-minute layered foundation...")
        try:
            out_file = "gui_layered_journey.mp3"
            self.orchestrator.generate_layered_journey(output_path=out_file, target_bpm=124, seed_track=self.selected_track)
            QMessageBox.information(self, "Success", f"Layered journey created: {out_file}")
            os.startfile(out_file)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Layering failed: {e}")

    def play_selected(self):
        if not self.selected_track:
            return
        os.startfile(self.selected_track['file_path'])

    def mix_with_top(self):
        if not self.selected_track or self.rec_list.rowCount() == 0:
            return
        
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
            final_mix = "gui_quick_mix.mp3"
            self.renderer.dj_stitch([t1['file_path'], t2['file_path']], final_mix)
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
