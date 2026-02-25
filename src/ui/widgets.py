from PyQt6.QtWidgets import QWidget, QTableWidget, QFrame, QLabel, QVBoxLayout, QMenu, QApplication, QProgressBar
from PyQt6.QtCore import Qt, QRect, pyqtSignal, QPoint, QMimeData
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont, QDrag

from src.scoring import CompatibilityScorer
from src.core.models import TrackSegment

class DraggableTable(QTableWidget):
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if item:
                row = item.row()
                tid = self.item(row, 0).data(Qt.ItemDataRole.UserRole)
                if tid is not None:
                    drag = QDrag(self)
                    mime = QMimeData()
                    mime.setText(str(tid))
                    drag.setMimeData(mime)
                    
                    # Visual representation of the clip being dragged
                    pixmap = self.viewport().grab(self.visualItemRect(item))
                    drag.setPixmap(pixmap)
                    drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
                    
                    drag.exec(Qt.DropAction.CopyAction)
        super().mousePressEvent(event)

class LibraryWaveformPreview(QWidget):
    selectionChanged = pyqtSignal(float, float) # start_pct, end_pct
    dragStarted = pyqtSignal(float, float) # start_pct, end_pct

    def __init__(self):
        super().__init__()
        self.waveform = []
        self.setFixedHeight(100)
        self.selection_start = None
        self.selection_end = None
        self.is_selecting = False
        self.setMouseTracking(True)
        
    def set_waveform(self, w):
        self.waveform = w
        self.selection_start = None
        self.selection_end = None
        self.update()
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.selection_start = None
            self.selection_end = None
            self.selectionChanged.emit(0.0, 1.0)
            self.update()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            x_pct = event.position().x() / self.width()
            # If we click inside an existing selection, start a drag
            if self.selection_start is not None and self.selection_end is not None:
                if self.selection_start <= x_pct <= self.selection_end:
                    self.start_drag()
                    return

            self.selection_start = x_pct
            self.selection_end = self.selection_start
            self.is_selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.selection_end = event.position().x() / self.width()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = False
            s = min(self.selection_start, self.selection_end)
            e = max(self.selection_start, self.selection_end)
            if e - s < 0.01: # Single click reset
                self.selection_start = None
                self.selection_end = None
                self.selectionChanged.emit(0.0, 1.0)
            else:
                self.selection_start = s
                self.selection_end = e
                self.selectionChanged.emit(s, e)
            self.update()
            
    def start_drag(self):
        if self.selection_start is not None and self.selection_end is not None:
            self.dragStarted.emit(self.selection_start, self.selection_end)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(25, 25, 25))
        if not self.waveform:
            return
            
        # Draw Selection
        if self.selection_start is not None and self.selection_end is not None:
            s = int(min(self.selection_start, self.selection_end) * self.width())
            e = int(max(self.selection_start, self.selection_end) * self.width())
            p.fillRect(s, 0, e - s, self.height(), QColor(0, 200, 255, 60))
            p.setPen(QPen(QColor(0, 200, 255, 150), 1))
            p.drawLine(s, 0, s, self.height())
            p.drawLine(e, 0, e, self.height())

        p.setPen(QPen(QColor(0, 255, 200, 180), 1))
        pts = len(self.waveform)
        mid = self.height() // 2
        mh = self.height() // 2 - 5
        
        for i in range(0, self.width(), 2):
            idx = int((i / self.width()) * pts)
            if idx < pts:
                v = self.waveform[idx] * mh
                p.drawLine(i, int(mid - v), i, int(mid + v))

class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.message_label = QLabel("Processing Journey...")
        self.message_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.message_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(400)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 2px solid #444; border-radius: 5px; text-align: center; color: white; background: #111; }
            QProgressBar::chunk { background-color: #007acc; width: 20px; }
        """)
        self.layout.addWidget(self.progress_bar)
        self.hide()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 200))
        
    def show_loading(self, m="Processing...", total=0):
        self.message_label.setText(m)
        self.progress_bar.setValue(0)
        if total > 0:
            self.progress_bar.setRange(0, total)
        else:
            self.progress_bar.setRange(0, 0) # Indeterminate
        
        self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()
        QApplication.processEvents()
        
    def set_progress(self, value):
        self.progress_bar.setValue(value)
        QApplication.processEvents()
        
    def hide_loading(self):
        self.hide()

class TimelineWidget(QWidget):
    segmentSelected = pyqtSignal(object)
    timelineChanged = pyqtSignal()
    undoRequested = pyqtSignal()
    cursorJumped = pyqtSignal(float)
    bridgeRequested = pyqtSignal(float)
    aiTransitionRequested = pyqtSignal(float, str) # x_pos, prompt_type
    duplicateRequested = pyqtSignal(object) # TrackSegment
    captureRequested = pyqtSignal(object) # TrackSegment
    stemsRequested = pyqtSignal(object) # TrackSegment
    sidechainRequested = pyqtSignal(object, int) # segment, source_lane
    zoomChanged = pyqtSignal(int)
    trackDropped = pyqtSignal(object, int, int) # tid_str, x, y
    fillRangeRequested = pyqtSignal(float, float) # start_ms, end_ms

    def __init__(self):
        super().__init__()
        self.segments = []
        self.setMinimumHeight(550)
        self.setAcceptDrops(True)
        self.setMouseTracking(True) # Enabled for dynamic cursor updates
        self.pixels_per_ms = 0.05
        self.selected_segment = None
        self.dragging = self.resizing = self.resizing_left = self.vol_dragging = self.fade_in_dragging = self.fade_out_dragging = self.slipping = self.setting_loop = self.resizing_timeline = False
        self.keyframe_dragging = False
        self.selected_keyframe_idx = -1
        self.selected_keyframe_param = None
        self.active_automation_param = "volume"
        self.drag_start_pos = None
        self.drag_start_ms = self.drag_start_dur = self.drag_start_fade = self.drag_start_offset = 0
        self.drag_start_vol = 1.0
        self.drag_start_lane = 0
        self.lane_height = 120
        self.lane_spacing = 10
        self.snap_threshold_ms = 2000
        self.target_bpm = 124.0
        self.show_modifications = True
        self.cursor_pos_ms = 0
        self.show_waveforms = True
        self.snap_to_grid = True
        self.lane_count = 8
        self.mutes = [False] * self.lane_count
        self.solos = [False] * self.lane_count
        self.loop_start_ms = 0
        self.loop_end_ms = 30000
        self.loop_enabled = False
        self.scorer = CompatibilityScorer()
        self.silence_regions = []
        self.update_geometry()

    def add_lane(self):
        self.lane_count += 1
        self.mutes.append(False)
        self.solos.append(False)
        self.update_geometry()
        self.timelineChanged.emit()

    def remove_lane(self):
        if self.lane_count > 1:
            self.lane_count -= 1
            self.mutes.pop()
            self.solos.pop()
            # Move any segments in the removed lane up one
            for s in self.segments:
                if s.lane >= self.lane_count:
                    s.lane = self.lane_count - 1
            self.update_geometry()
            self.timelineChanged.emit()

    def find_silence_regions(self):
        """Analyzes timeline energy to find gaps where volume is below threshold."""
        if not self.segments: return []
        
        # Determine total length to scan
        total_len = max(s.start_ms + s.duration_ms for s in self.segments)
        
        gaps = []
        step_ms = 500 # Scan every 0.5s
        threshold = 0.15 # 15% combined volume is "quiet"
        
        in_gap = False
        gap_start = 0
        
        import math
        
        for t in range(0, int(total_len), step_ms):
            combined_vol = 0.0
            for s in self.segments:
                if s.start_ms <= t <= (s.start_ms + s.duration_ms):
                    # Calculate segment volume at time t
                    seg_v = s.volume
                    rel_t = t - s.start_ms
                    
                    # Apply Fades (Linear approximation for detector)
                    if rel_t < s.fade_in_ms and s.fade_in_ms > 0:
                        seg_v *= (rel_t / s.fade_in_ms)
                    elif rel_t > (s.duration_ms - s.fade_out_ms) and s.fade_out_ms > 0:
                        seg_v *= ((s.duration_ms - rel_t) / s.fade_out_ms)
                    
                    combined_vol += seg_v
            
            if combined_vol < threshold:
                if not in_gap:
                    in_gap = True
                    gap_start = t
            else:
                if in_gap:
                    in_gap = False
                    if t - gap_start > 500: # Only mark gaps longer than 0.5s
                        gaps.append((gap_start, t))
        
        # Close final gap if it exists
        if in_gap:
            gaps.append((gap_start, total_len))
            
        self.silence_regions = gaps
        return gaps

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        try:
            tid_str = event.mimeData().text()
            pos = event.position()
            self.trackDropped.emit(tid_str, int(pos.x()), int(pos.y()))
            event.acceptProposedAction()
        except:
            pass

    def update_geometry(self):
        max_ms = 600000
        if self.segments:
            max_ms = max(max_ms, max(s.start_ms + s.duration_ms for s in self.segments) + 60000)
        self.setMinimumWidth(int(max_ms * self.pixels_per_ms))
        
        total_h = self.lane_count * (self.lane_height + self.lane_spacing) + 100
        self.setMinimumHeight(total_h)
        self.update()

    def get_ms_per_beat(self):
        return (60.0 / self.target_bpm) * 1000.0

    def get_seg_rect(self, seg):
        x = int(seg.start_ms * self.pixels_per_ms)
        w = int(seg.duration_ms * self.pixels_per_ms)
        h = self.lane_height - 20
        y_center = (seg.lane * (self.lane_height + self.lane_spacing)) + (self.lane_height // 2) + 40
        return QRect(x, y_center - (h // 2), w, h)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(25, 25, 25))
        
        # Draw Silence Guard Warnings
        for start, end in self.silence_regions:
            sx = int(start * self.pixels_per_ms)
            sw = int((end - start) * self.pixels_per_ms)
            painter.fillRect(sx, 0, sw, 40, QColor(255, 50, 50, 80))
            painter.setPen(QPen(QColor(255, 50, 50, 150), 1))
            painter.drawText(sx + 2, 38, "⚠ GAP")

        if self.loop_enabled:
            lx = int(self.loop_start_ms * self.pixels_per_ms)
            lw = int((self.loop_end_ms - self.loop_start_ms) * self.pixels_per_ms)
            painter.fillRect(lx, 0, lw, 40, QColor(0, 200, 255, 60))
            painter.setPen(QPen(QColor(0, 200, 255, 150), 2))
            painter.drawLine(lx, 0, lx, 40)
            painter.drawLine(lx + lw, 0, lx + lw, 40)
            
        any_solo = any(self.solos)
        for i in range(self.lane_count): 
            y = i * (self.lane_height + self.lane_spacing) + 40
            bg = QColor(32, 32, 32)
            if self.solos[i]:
                bg = QColor(45, 45, 32)
            elif self.mutes[i] or (any_solo and not self.solos[i]):
                bg = QColor(20, 20, 20)
                
            painter.fillRect(0, y, self.width(), self.lane_height, bg)
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(5, y + 15, f"LANE {i+1}")
            
            mr = QRect(5, y + 25, 20, 20)
            painter.setBrush(QBrush(QColor(255, 50, 50) if self.mutes[i] else QColor(60, 60, 60)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(mr, 3, 3)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(mr, Qt.AlignmentFlag.AlignCenter, "M")
            
            sr = QRect(30, y + 25, 20, 20)
            painter.setBrush(QBrush(QColor(255, 200, 0) if self.solos[i] else QColor(60, 60, 60)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(sr, 3, 3)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(sr, Qt.AlignmentFlag.AlignCenter, "S")
            
        mpb = self.get_ms_per_beat()
        mpbar = mpb * 4
        for i in range(0, 3600000, int(mpb)):
            x = int(i * self.pixels_per_ms)
            if x > self.width():
                break
            if (i % int(mpbar)) < 10:
                painter.setPen(QPen(QColor(80, 80, 80), 1))
                painter.drawLine(x, 0, x, self.height())
                painter.setPen(QColor(150, 150, 150))
                painter.drawText(x + 5, 15, f"BAR {int(i // mpbar) + 1}")
            else:
                painter.setPen(QPen(QColor(50, 50, 50), 1, Qt.PenStyle.DotLine))
                painter.drawLine(x, 40, x, self.height())
                
        # Time Tickers (Every 10 seconds)
        painter.setPen(QPen(QColor(0, 200, 255, 100), 1))
        for s in range(0, 3600, 10):
            ms = s * 1000
            x = int(ms * self.pixels_per_ms)
            if x > self.width(): break
            painter.drawLine(x, 25, x, 40)
            if s % 30 == 0:
                mins = s // 60
                secs = s % 60
                painter.setPen(QColor(0, 200, 255, 180))
                painter.drawText(x + 5, 35, f"{mins}:{secs:02d}")
                painter.setPen(QPen(QColor(0, 200, 255, 100), 1))
                
        for seg in self.segments:
            rect = self.get_seg_rect(seg)
            color = QColor(seg.color)
            is_ducked = False
            if not seg.is_primary:
                for o in self.segments:
                    if o != seg and o.is_primary and max(seg.start_ms, o.start_ms) < min(seg.get_end_ms(), o.get_end_ms()):
                        is_ducked = True
                        break
            hc = False
            for o in self.segments:
                if o != seg and seg.overlaps_with(o):
                    if self.scorer.calculate_harmonic_score(seg.key, o.key) < 60:
                        hc = True
                        break
                        
            dv = seg.volume * 0.63 if is_ducked else seg.volume
            color.setAlpha(int(120 + 135 * (min(dv, 1.5) / 1.5)))
            
            if seg == self.selected_segment:
                painter.setBrush(QBrush(color.lighter(130)))
                painter.setPen(QPen(Qt.GlobalColor.white, 3))
            elif seg.is_primary:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(255, 215, 0), 3))
            elif hc:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(255, 50, 50), 3))
            else:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(200, 200, 200), 1))
                
            painter.drawRoundedRect(rect, 6, 6)
            
            if self.show_waveforms:
                mid_y = rect.center().y()
                max_h = rect.height() // 2
                
                # --- Multi-Stem Visualizer ---
                if hasattr(seg, 'stem_waveforms') and seg.stem_waveforms:
                    stems = [
                        ("vocals", QColor(255, 204, 0, 180)), # Yellow
                        ("drums", QColor(255, 51, 102, 180)),  # Pink
                        ("bass", QColor(0, 204, 255, 180)),   # Blue
                        ("other", QColor(153, 51, 255, 180))   # Purple
                    ]
                    
                    # Split the vertical space into 4 lanes
                    stem_h = rect.height() // 4
                    for idx, (stype, scolor) in enumerate(stems):
                        if stype in seg.stem_waveforms:
                            sw = seg.stem_waveforms[stype]
                            painter.setPen(QPen(scolor, 1))
                            pts = len(sw)
                            s_top = rect.top() + (idx * stem_h)
                            s_mid = s_top + (stem_h // 2)
                            s_max_h = stem_h // 2 - 2
                            
                            for i in range(0, rect.width(), 2):
                                ri = (i / rect.width()) * (seg.duration_ms / 30000.0)
                                s_idx = int((ri + (seg.offset_ms / 30000.0)) * pts) % pts
                                val = sw[s_idx] * s_max_h
                                painter.drawLine(rect.left() + i, int(s_mid - val), rect.left() + i, int(s_mid + val))
                
                # --- Single Waveform Fallback ---
                elif seg.waveform:
                    painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
                    pts = len(seg.waveform)
                    for i in range(0, rect.width(), 2):
                        ri = (i / rect.width()) * (seg.duration_ms / 30000.0)
                        idx = int((ri + (seg.offset_ms / 30000.0)) * pts) % pts
                        val = seg.waveform[idx] * max_h
                        painter.drawLine(rect.left() + i, int(mid_y - val), rect.left() + i, int(mid_y + val))
                    
            painter.setPen(QPen(QColor(255, 255, 255, 180), 2))
            vy = rect.bottom() - int(rect.height() * (dv / 1.5))
            painter.drawLine(rect.left(), vy, rect.right(), vy)
            
            if seg.onsets:
                painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
                s_f = self.target_bpm / seg.bpm
                for o_ms in seg.onsets:
                    adj = (o_ms - seg.offset_ms) * s_f
                    if 0 <= adj <= seg.duration_ms:
                        tx = rect.left() + int(adj * self.pixels_per_ms)
                        painter.drawLine(tx, rect.top() + 5, tx, rect.bottom() - 5)

            # --- Visual Section Markers (Remote MIR) ---
            if hasattr(seg, 'sections') and seg.sections:
                s_f = self.target_bpm / seg.bpm
                for sec in seg.sections:
                    s_ms = sec['start'] * 1000.0
                    adj = (s_ms - seg.offset_ms) * s_f
                    if 0 <= adj <= seg.duration_ms:
                        tx = rect.left() + int(adj * self.pixels_per_ms)
                        label = sec['label'].upper()
                        
                        # Style based on section type
                        if label == "DROP": s_color = QColor(255, 50, 50, 180)
                        elif label == "BUILD": s_color = QColor(255, 200, 0, 180)
                        else: s_color = QColor(255, 255, 255, 100)
                        
                        painter.setPen(QPen(s_color, 1, Qt.PenStyle.DashLine))
                        painter.drawLine(tx, rect.top(), tx, rect.bottom())
                        
                        painter.setPen(s_color)
                        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                        painter.drawText(tx + 3, rect.bottom() - 5, label)
                        
            fi_w = int(seg.fade_in_ms * self.pixels_per_ms)
            fo_w = int(seg.fade_out_ms * self.pixels_per_ms)
            
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1, Qt.PenStyle.DashLine))
            painter.drawLine(rect.left(), rect.bottom(), rect.left() + fi_w, rect.top())
            painter.drawLine(rect.right() - fo_w, rect.top(), rect.right(), rect.bottom())
            
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(rect.left() + fi_w - 4, rect.top() - 4, 8, 8)
            painter.drawEllipse(rect.right() - fo_w - 4, rect.top() - 4, 8, 8)
            
            # --- Visual FX Indicators ---
            if hasattr(seg, 'reverb') and seg.reverb > 0:
                painter.setBrush(QBrush(QColor(0, 200, 255, int(255 * seg.reverb))))
                painter.drawEllipse(rect.right() - 25, rect.bottom() - 25, 12, 12)
            if hasattr(seg, 'harmonics') and seg.harmonics > 0:
                painter.setBrush(QBrush(QColor(255, 150, 0, int(255 * seg.harmonics))))
                painter.drawEllipse(rect.right() - 45, rect.bottom() - 25, 12, 12)

            # --- Keyframe Rendering ---
            if hasattr(seg, 'keyframes'):
                # Only draw for the active parameter to avoid clutter
                if self.active_automation_param in seg.keyframes:
                    points = seg.keyframes[self.active_automation_param]
                    if points:
                        # Color coding based on parameter type
                        if self.active_automation_param == "volume": k_color = QColor(255, 200, 0, 200) # Yellow
                        elif self.active_automation_param == "pan": k_color = QColor(0, 200, 255, 200) # Blue
                        elif "cut" in self.active_automation_param: k_color = QColor(0, 255, 100, 200) # Green
                        else: k_color = QColor(255, 100, 255, 200) # Purple
                        
                        painter.setPen(QPen(k_color, 2))
                        painter.setBrush(QBrush(k_color))
                        
                        # Sort points just in case
                        sorted_pts = sorted(points, key=lambda x: x[0])
                        prev_x = rect.left()
                        prev_y = rect.bottom() - int(rect.height() * sorted_pts[0][1])
                        
                        for ms, val in sorted_pts:
                            # Draw point
                            x = rect.left() + int(ms * self.pixels_per_ms)
                            y = rect.bottom() - int(rect.height() * max(0.0, min(1.0, val)))
                            
                            painter.drawLine(prev_x, prev_y, x, y)
                            painter.drawEllipse(x - 3, y - 3, 6, 6)
                            
                            prev_x = x
                            prev_y = y
                            
                        # Line to end?
                        if prev_x < rect.right():
                            painter.drawLine(prev_x, prev_y, rect.right(), prev_y)

            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(rect.adjusted(8, 8, -8, -8), Qt.AlignmentFlag.AlignTop, seg.filename)

        # Draw Playback Cursor
        cx = int(self.cursor_pos_ms * self.pixels_per_ms)
        painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
        painter.drawLine(cx, 0, cx, self.height())
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawPolygon(QPoint(cx-6, 0), QPoint(cx+6, 0), QPoint(cx, 10))

    def mousePressEvent(self, event):
        if event.pos().y() > self.height() - 15:
            self.resizing_timeline = True
            self.drag_start_pos = event.pos()
            self.drag_start_h = self.height()
            return

        for i in range(self.lane_count):
            y = i * (self.lane_height + self.lane_spacing) + 40
            m_r = QRect(5, y + 25, 20, 20)
            s_r = QRect(30, y + 25, 20, 20)
            if m_r.contains(event.pos()):
                self.mutes[i] = not self.mutes[i]
                self.update()
                self.timelineChanged.emit()
                return
            if s_r.contains(event.pos()):
                self.solos[i] = not self.solos[i]
                self.update()
                self.timelineChanged.emit()
                return

        if event.pos().y() < 40:
            self.setting_loop = True
            self.loop_start_ms = event.pos().x() / self.pixels_per_ms
            self.loop_end_ms = self.loop_start_ms
            self.loop_enabled = True
            self.update()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Check for Keyframe Selection/Dragging
            for seg in reversed(self.segments):
                rect = self.get_seg_rect(seg)
                if rect.contains(event.pos()) and hasattr(seg, 'keyframes'):
                    for param, points in seg.keyframes.items():
                        for idx, (ms, val) in enumerate(points):
                            kx = rect.left() + int(ms * self.pixels_per_ms)
                            ky = rect.bottom() - int(rect.height() * val)
                            if QRect(kx - 6, ky - 6, 12, 12).contains(event.pos()):
                                self.selected_segment = seg
                                self.selected_keyframe_param = param
                                self.selected_keyframe_idx = idx
                                self.keyframe_dragging = True
                                self.drag_start_pos = event.pos()
                                self.update()
                                return

            # Check for Keyframe Addition (Ctrl + Click)
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                for seg in self.segments:
                    r = self.get_seg_rect(seg)
                    if r.contains(event.pos()):
                        rel_ms = (event.pos().x() - r.left()) / self.pixels_per_ms
                        # Invert Y to get 0.0 - 1.0 value
                        val = 1.0 - ((event.pos().y() - r.top()) / r.height())
                        
                        # Use selected parameter
                        seg.add_keyframe(self.active_automation_param, rel_ms, val)
                        self.update()
                        self.timelineChanged.emit()
                        return

            cs = None
            for seg in reversed(self.segments):
                r = self.get_seg_rect(seg)
                fi = r.left() + int(seg.fade_in_ms * self.pixels_per_ms)
                fo = r.right() - int(seg.fade_out_ms * self.pixels_per_ms)
                
                if QRect(fi-10, r.top()-10, 20, 20).contains(event.pos()):
                    self.selected_segment = seg
                    self.fade_in_dragging = True
                    self.drag_start_pos = event.pos()
                    self.drag_start_fade = seg.fade_in_ms
                    self.update()
                    return
                    
                if QRect(fo-10, r.top()-10, 20, 20).contains(event.pos()):
                    self.selected_segment = seg
                    self.fade_out_dragging = True
                    self.drag_start_pos = event.pos()
                    self.drag_start_fade = seg.fade_out_ms
                    self.update()
                    return
                    
                if r.contains(event.pos()):
                    cs = seg
                    break
                    
            self.selected_segment = cs
            self.segmentSelected.emit(cs)
            
            if self.selected_segment:
                self.undoRequested.emit()
                self.drag_start_pos = event.pos()
                self.drag_start_ms = self.selected_segment.start_ms
                self.drag_start_dur = self.selected_segment.duration_ms
                self.drag_start_vol = self.selected_segment.volume
                self.drag_start_lane = self.selected_segment.lane
                self.drag_start_offset = self.selected_segment.offset_ms
                r = self.get_seg_rect(self.selected_segment)
                
                if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                    self.slipping = True
                elif event.position().x() < (r.left() + 20): # Left Edge
                    self.resizing_left = True
                elif event.position().x() > (r.right() - 20): # Right Edge
                    self.resizing = True
                elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.vol_dragging = True
                else:
                    self.dragging = True
            else:
                self.cursor_pos_ms = event.pos().x() / self.pixels_per_ms
                self.cursorJumped.emit(self.cursor_pos_ms)
            self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            ts = None
            for seg in reversed(self.segments):
                if self.get_seg_rect(seg).contains(event.pos()):
                    ts = seg
                    break
                    
            m = QMenu(self)
            if ts:
                pa = m.addAction("⭐ Unmark Primary" if ts.is_primary else "⭐ Set as Primary")
                sa = m.addAction("✂ Split at Cursor")
                qa = m.addAction("🪄 Quantize to Grid")
                da_dup = m.addAction("👯 Duplicate Track")
                m.addSeparator()
                sa_stems = m.addAction("🔪 Separate Stems (Remote AI)")
                m.addSeparator()
                pm = m.addMenu("🎵 Shift Pitch")
                for i in range(-6, 7):
                    a = pm.addAction(f"{i:+} st")
                    a.setData(i)
                sl = m.addAction("💾 Capture as New Loop")
                da_rem = m.addAction("🗑 Remove Track")
                m.addSeparator()
                ra_keys = m.addAction("🧹 Remove Keyframes")
                m.addSeparator()
                scm = m.addMenu("🔗 Auto-Sidechain to Lane")
                for i in range(self.lane_count):
                    la = scm.addAction(f"Lane {i+1}")
                    la.setData(i)
                
                act = m.exec(self.mapToGlobal(event.pos()))
                
                if act == pa:
                    self.undoRequested.emit()
                    ts.is_primary = not ts.is_primary
                elif act == sa:
                    self.undoRequested.emit()
                    self.split_segment(ts, event.pos().x())
                elif act == qa:
                    self.undoRequested.emit()
                    self.quantize_segment(ts)
                elif act == da_dup:
                    self.undoRequested.emit()
                    self.duplicateRequested.emit(ts)
                elif act == sa_stems:
                    self.stemsRequested.emit(ts)
                elif act == sl:
                    self.captureRequested.emit(ts)
                elif act in pm.actions():
                    self.undoRequested.emit()
                    ts.pitch_shift = act.data()
                elif act == da_rem:
                    self.undoRequested.emit()
                    self.segments.remove(ts)
                elif act == ra_keys:
                    self.undoRequested.emit()
                    ts.keyframes = {}
                elif act in scm.actions():
                    self.sidechainRequested.emit(ts, act.data())
                    
                if self.selected_segment == ts:
                    self.selected_segment = None
                self.update_geometry()
                self.timelineChanged.emit()
            else:
                ba = m.addAction("🪄 Find Bridge Track here")
                
                ai_menu = m.addMenu("✨ Generate AI Asset")
                ar = ai_menu.addAction("📈 Cinematic Riser")
                ad = ai_menu.addAction("📉 Bass Drop")
                ap = ai_menu.addAction("🌌 Ambient Pad")
                ab = ai_menu.addAction("🥁 Percussion Build")
                
                fa = fs = fe = None
                if self.loop_enabled and (self.loop_end_ms - self.loop_start_ms) > 1000:
                    m.addSeparator()
                    fa = m.addAction("🩹 AI: Fill Selected Range")
                else:
                    m.addSeparator()
                    fs = m.addAction("🩹 AI: Fill from Start to Here")
                    fe = m.addAction("🩹 AI: Fill from Here to End")

                act = m.exec(self.mapToGlobal(event.pos()))
                if act == ba:
                    self.bridgeRequested.emit(event.pos().x())
                elif act == ar:
                    self.aiTransitionRequested.emit(event.pos().x(), "riser")
                elif act == ad:
                    self.aiTransitionRequested.emit(event.pos().x(), "drop")
                elif act == ap:
                    self.aiTransitionRequested.emit(event.pos().x(), "pad")
                elif act == ab:
                    self.aiTransitionRequested.emit(event.pos().x(), "percussion")
                elif act == fa:
                    self.fillRangeRequested.emit(self.loop_start_ms, self.loop_end_ms)
                elif act == fs:
                    self.fillRangeRequested.emit(0.0, event.pos().x() / self.pixels_per_ms)
                elif act == fe:
                    total_dur = max([s.get_end_ms() for s in self.segments]) if self.segments else 30000
                    self.fillRangeRequested.emit(event.pos().x() / self.pixels_per_ms, total_dur)

    def mouseMoveEvent(self, event):
        # Update Cursor based on hover position (if not dragging)
        if not any([self.dragging, self.resizing, self.resizing_left, self.vol_dragging, self.fade_in_dragging, self.fade_out_dragging, self.slipping]):
            over_edge = False
            for seg in self.segments:
                r = self.get_seg_rect(seg)
                if r.contains(event.pos()):
                    # Tooltip for vocals
                    if hasattr(seg, 'vocal_lyrics') and (seg.vocal_lyrics or seg.vocal_gender):
                        tip = ""
                        if seg.vocal_gender: tip += f"[{seg.vocal_gender}] "
                        if seg.vocal_lyrics: tip += f'"{seg.vocal_lyrics}"'
                        from PyQt6.QtWidgets import QToolTip
                        QToolTip.showText(event.globalPosition().toPoint(), tip, self)

                    if event.position().x() < (r.left() + 20) or event.position().x() > (r.right() - 20):
                        self.setCursor(Qt.CursorShape.SizeHorCursor)
                        over_edge = True
                    else:
                        self.setCursor(Qt.CursorShape.PointingHandCursor)
                        over_edge = True
                    break
            if not over_edge:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        if self.resizing_timeline:
            self.setMinimumHeight(max(400, self.drag_start_h + (event.pos().y() - self.drag_start_pos.y())))
            self.update_geometry()
            return

        if self.setting_loop:
            self.loop_end_ms = max(self.loop_start_ms, event.pos().x() / self.pixels_per_ms)
            self.update()
            return

        if self.keyframe_dragging and self.selected_segment:
            rect = self.get_seg_rect(self.selected_segment)
            rel_ms = (event.pos().x() - rect.left()) / self.pixels_per_ms
            val = 1.0 - ((event.pos().y() - rect.top()) / rect.height())
            
            # Constrain to segment bounds
            rel_ms = max(0.0, min(self.selected_segment.duration_ms, rel_ms))
            val = max(0.0, min(1.0, val))
            
            # Update keyframe
            pts = self.selected_segment.keyframes[self.selected_keyframe_param]
            pts[self.selected_keyframe_idx] = (rel_ms, val)
            pts.sort(key=lambda x: x[0])
            
            # Re-find index after sort to maintain selection
            for i, p in enumerate(pts):
                if p[0] == rel_ms:
                    self.selected_keyframe_idx = i
                    break
                    
            self.update()
            return

        if not self.selected_segment:
            return

        dx = event.pos().x() - self.drag_start_pos.x()
        dy = event.pos().y() - self.drag_start_pos.y()
        mpb = self.get_ms_per_beat()

        if self.slipping:
            self.selected_segment.offset_ms = max(0, self.drag_start_offset - dx/self.pixels_per_ms)
        elif self.resizing_left:
            # Resize from start: adjust both start_ms and offset_ms
            delta_ms = dx / self.pixels_per_ms
            if self.snap_to_grid:
                new_start = round((self.drag_start_ms + delta_ms) / mpb) * mpb
                actual_delta = new_start - self.drag_start_ms
            else:
                actual_delta = delta_ms
            
            # Constraints: don't let duration go below 500ms
            if self.drag_start_dur - actual_delta > 500:
                self.selected_segment.start_ms = self.drag_start_ms + actual_delta
                self.selected_segment.offset_ms = max(0, self.drag_start_offset + actual_delta)
                self.selected_segment.duration_ms = self.drag_start_dur - actual_delta
        elif self.fade_in_dragging:
            rf = self.drag_start_fade + dx/self.pixels_per_ms
            if self.snap_to_grid:
                rf = round(rf / mpb) * mpb
            self.selected_segment.fade_in_ms = max(0, min(self.selected_segment.duration_ms/2, rf))
        elif self.fade_out_dragging:
            rf = self.drag_start_fade - dx/self.pixels_per_ms
            if self.snap_to_grid:
                rf = round(rf / mpb) * mpb
            self.selected_segment.fade_out_ms = max(0, min(self.selected_segment.duration_ms/2, rf))
        elif self.resizing:
            rd = self.drag_start_dur + dx/self.pixels_per_ms
            if self.snap_to_grid:
                rd = round(rd / mpb) * mpb
            self.selected_segment.duration_ms = max(1000, rd)
        elif self.vol_dragging:
            self.selected_segment.volume = max(0.0, min(1.5, self.drag_start_vol - dy/150.0))
        elif self.dragging:
            ns = max(0, self.drag_start_ms + dx/self.pixels_per_ms)
            if self.snap_to_grid:
                ns = round(ns / mpb) * mpb
            for o in self.segments:
                if o == self.selected_segment:
                    continue
                oe = o.get_end_ms()
                if abs(ns - oe) < self.snap_threshold_ms:
                    ns = oe
                elif abs(ns - o.start_ms) < self.snap_threshold_ms:
                    ns = o.start_ms
            self.selected_segment.start_ms = ns
            nl = max(0, min(7, int((event.pos().y() - 40) // (self.lane_height + self.lane_spacing))))
            self.selected_segment.lane = nl
            
        self.update_geometry()
        self.timelineChanged.emit()

    def mouseReleaseEvent(self, event):
        self.dragging = self.resizing = self.resizing_left = self.vol_dragging = self.fade_in_dragging = self.fade_out_dragging = self.slipping = self.setting_loop = self.resizing_timeline = self.keyframe_dragging = False
        self.update_geometry()
        self.timelineChanged.emit()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            nv = max(10, min(200, int(self.pixels_per_ms * 1000) + (event.angleDelta().y() // 120 * 10)))
            self.pixels_per_ms = nv / 1000.0
            self.update_geometry()
            self.zoomChanged.emit(nv)
        else:
            super().wheelEvent(event)
    
    def split_segment(self, seg, x_pos):
        sm = x_pos / self.pixels_per_ms
        rs = sm - seg.start_ms
        if rs < 500 or rs > (seg.duration_ms - 500):
            return 
        nd = seg.duration_ms - rs
        no = seg.offset_ms + rs
        seg.duration_ms = rs 
        td = {
            'id': seg.id, 'filename': seg.filename, 'file_path': seg.file_path, 
            'bpm': seg.bpm, 'harmonic_key': seg.key, 
            'onsets_json': ",".join([str(x/1000.0) for x in seg.onsets])
        }
        ns = TrackSegment(td, start_ms=sm, duration_ms=nd, lane=seg.lane, offset_ms=no)
        ns.volume = seg.volume
        ns.is_primary = seg.is_primary
        ns.waveform = seg.waveform
        ns.pitch_shift = seg.pitch_shift
        self.segments.append(ns)
        self.update_geometry()
        self.timelineChanged.emit()

    def quantize_segment(self, seg):
        if not seg.onsets:
            return
        mpb = self.get_ms_per_beat()
        stretch = self.target_bpm / seg.bpm
        foc = (seg.onsets[0] - seg.offset_ms) * stretch
        cwp = seg.start_ms + foc
        twp = round(cwp / mpb) * mpb
        seg.start_ms += (twp - cwp)
        self.update()
        self.timelineChanged.emit()

    def add_track(self, td, start_ms=None, lane=0):
        if start_ms is None:
            if self.segments:
                l_s = max(self.segments, key=lambda s: s.get_end_ms())
                start_ms = l_s.get_end_ms() - 5000
                lane = l_s.lane
            else:
                start_ms = 0
        ns = TrackSegment(td, start_ms=start_ms, lane=lane)
        self.segments.append(ns)
        self.update_geometry()
        self.timelineChanged.emit()
        return ns
