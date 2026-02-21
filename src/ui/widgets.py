from PyQt6.QtWidgets import QWidget, QTableWidget, QFrame, QLabel, QVBoxLayout, QMenu, QApplication
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
    def __init__(self):
        super().__init__()
        self.waveform = []
        self.setFixedHeight(60)
        
    def set_waveform(self, w):
        self.waveform = w
        self.update()
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(25, 25, 25))
        if not self.waveform:
            return
            
        p.setPen(QPen(QColor(0, 255, 200, 180), 1))
        pts = len(self.waveform)
        mid = self.height() // 2
        mh = self.height() // 2 - 5
        
        for i in range(0, self.width(), 2):
            idx = int((i / self.width()) * pts)
            if idx < pts:
                v = self.waveform[idx] * mh
                p.drawLine(i, int(mid - v), i, int(mid + v))

from PyQt6.QtWidgets import QWidget, QTableWidget, QFrame, QLabel, QVBoxLayout, QMenu, QApplication, QProgressBar
from PyQt6.QtCore import Qt, QRect, pyqtSignal, QPoint, QMimeData
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont, QDrag

from src.scoring import CompatibilityScorer
from src.core.models import TrackSegment

class DraggableTable(QTableWidget):
    # ... (remains same)

class LibraryWaveformPreview(QWidget):
    # ... (remains same)

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
    aiTransitionRequested = pyqtSignal(float)
    duplicateRequested = pyqtSignal(object) # TrackSegment
    captureRequested = pyqtSignal(object) # TrackSegment
    zoomChanged = pyqtSignal(int)
    trackDropped = pyqtSignal(int, int, int) # tid, x, y

    def __init__(self):
        super().__init__()
        self.segments = []
        self.setMinimumHeight(550)
        self.setAcceptDrops(True)
        self.setMouseTracking(True) # Enabled for dynamic cursor updates
        self.pixels_per_ms = 0.05
        self.selected_segment = None
        self.dragging = self.resizing = self.resizing_left = self.vol_dragging = self.fade_in_dragging = self.fade_out_dragging = self.slipping = self.setting_loop = self.resizing_timeline = False
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
        self.mutes = [False] * 5
        self.solos = [False] * 5
        self.loop_start_ms = 0
        self.loop_end_ms = 30000
        self.loop_enabled = False
        self.scorer = CompatibilityScorer()
        self.update_geometry()

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        try:
            tid = int(event.mimeData().text())
            pos = event.position()
            self.trackDropped.emit(tid, int(pos.x()), int(pos.y()))
            event.acceptProposedAction()
        except:
            pass

    def update_geometry(self):
        max_ms = 600000
        if self.segments:
            max_ms = max(max_ms, max(s.start_ms + s.duration_ms for s in self.segments) + 60000)
        self.setMinimumWidth(int(max_ms * self.pixels_per_ms))
        
        total_h = 5 * (self.lane_height + self.lane_spacing) + 100
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
        
        if self.loop_enabled:
            lx = int(self.loop_start_ms * self.pixels_per_ms)
            lw = int((self.loop_end_ms - self.loop_start_ms) * self.pixels_per_ms)
            painter.fillRect(lx, 0, lw, 40, QColor(0, 200, 255, 60))
            painter.setPen(QPen(QColor(0, 200, 255, 150), 2))
            painter.drawLine(lx, 0, lx, 40)
            painter.drawLine(lx + lw, 0, lx + lw, 40)
            
        any_solo = any(self.solos)
        for i in range(5): 
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
                    if o != seg and o.is_primary and max(seg.start_ms, o.start_ms) < min(seg.start_ms + seg.duration_ms, o.start_ms + o.duration_ms):
                        is_ducked = True
                        break
            hc = False
            for o in self.segments:
                if o != seg and max(seg.start_ms, o.start_ms) < min(seg.start_ms + seg.duration_ms, o.start_ms + o.duration_ms):
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
            
            if self.show_waveforms and seg.waveform:
                painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
                pts = len(seg.waveform)
                mid_y = rect.center().y()
                max_h = rect.height() // 2
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
                        
            fi_w = int(seg.fade_in_ms * self.pixels_per_ms)
            fo_w = int(seg.fade_out_ms * self.pixels_per_ms)
            
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1, Qt.PenStyle.DashLine))
            painter.drawLine(rect.left(), rect.bottom(), rect.left() + fi_w, rect.top())
            painter.drawLine(rect.right() - fo_w, rect.top(), rect.right(), rect.bottom())
            
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(rect.left() + fi_w - 4, rect.top() - 4, 8, 8)
            painter.drawEllipse(rect.right() - fo_w - 4, rect.top() - 4, 8, 8)
            
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(rect.adjusted(8, 8, -8, -8), Qt.AlignmentFlag.AlignTop, seg.filename)
            
            by = rect.bottom() - 22
            bx = rect.left() + 8
            if seg.is_primary:
                painter.setBrush(QBrush(QColor(255, 215, 0)))
                painter.setPen(Qt.PenStyle.NoPen)
                br = QRect(bx, by, 60, 16)
                painter.drawRoundedRect(br, 4, 4)
                painter.setPen(Qt.GlobalColor.black)
                painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                painter.drawText(br, Qt.AlignmentFlag.AlignCenter, "PRIMARY")
                bx += 65
                
            if self.show_modifications:
                if abs(seg.bpm - self.target_bpm) > 0.1:
                    painter.setBrush(QBrush(QColor(255, 165, 0)))
                    br = QRect(bx, by, 55, 16)
                    painter.drawRoundedRect(br, 4, 4)
                    painter.setPen(Qt.GlobalColor.black)
                    painter.drawText(br, Qt.AlignmentFlag.AlignCenter, "STRETCH")
                    bx += 60
                if abs(seg.volume - 1.0) > 0.05:
                    painter.setBrush(QBrush(QColor(0, 200, 255)))
                    br = QRect(bx, by, 40, 16)
                    painter.drawRoundedRect(br, 4, 4)
                    painter.setPen(Qt.GlobalColor.black)
                    painter.drawText(br, Qt.AlignmentFlag.AlignCenter, f"{int(seg.volume*100)}%")
                    bx += 45
                if seg.pitch_shift != 0:
                    painter.setBrush(QBrush(QColor(200, 100, 255)))
                    br = QRect(bx, by, 40, 16)
                    painter.drawRoundedRect(br, 4, 4)
                    painter.setPen(Qt.GlobalColor.black)
                    painter.drawText(br, Qt.AlignmentFlag.AlignCenter, f"{seg.pitch_shift:+}st")
                    
        cx = int(self.cursor_pos_ms * self.pixels_per_ms)
        painter.setPen(QPen(QColor(255, 50, 50), 2))
        painter.drawLine(cx, 0, cx, self.height())
        painter.setBrush(QBrush(QColor(255, 50, 50)))
        painter.drawPolygon(QPoint(cx - 8, 0), QPoint(cx + 8, 0), QPoint(cx, 12))
        
        painter.fillRect(0, self.height() - 15, self.width(), 15, QColor(40, 40, 40))
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawLine(0, self.height() - 15, self.width(), self.height() - 15)
        painter.setPen(QColor(120, 120, 120))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(self.rect().adjusted(0, 0, 0, -2), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, "↕ DRAG TO RESIZE TIMELINE VERTICALLY")

    def mousePressEvent(self, event):
        if event.pos().y() > self.height() - 15:
            self.resizing_timeline = True
            self.drag_start_pos = event.pos()
            self.drag_start_h = self.height()
            return

        for i in range(5):
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
                pm = m.addMenu("🎵 Shift Pitch")
                for i in range(-6, 7):
                    t = f"{i:+} st" if i != 0 else "Original"
                    p_act = pm.addAction(t)
                    p_act.setData(i)
                m.addSeparator()
                sl = m.addAction("💾 Capture as New Loop")
                da_rem = m.addAction("🗑 Remove Track")
                
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
                elif act == sl:
                    self.captureRequested.emit(ts)
                elif act in pm.actions():
                    self.undoRequested.emit()
                    ts.pitch_shift = act.data()
                elif act == da_rem:
                    self.undoRequested.emit()
                    self.segments.remove(ts)
                    
                if self.selected_segment == ts:
                    self.selected_segment = None
                self.update_geometry()
                self.timelineChanged.emit()
            else:
                ba = m.addAction("🪄 Find Bridge Track here")
                ta = m.addAction("✨ Generate AI Transition")
                act = m.exec(self.mapToGlobal(event.pos()))
                if act == ba:
                    self.bridgeRequested.emit(event.pos().x())
                elif act == ta:
                    self.aiTransitionRequested.emit(event.pos().x())

    def mouseMoveEvent(self, event):
        # Update Cursor based on hover position (if not dragging)
        if not any([self.dragging, self.resizing, self.resizing_left, self.vol_dragging, self.fade_in_dragging, self.fade_out_dragging, self.slipping]):
            over_edge = False
            for seg in self.segments:
                r = self.get_seg_rect(seg)
                if r.contains(event.pos()):
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
                oe = o.start_ms + o.duration_ms
                if abs(ns - oe) < self.snap_threshold_ms:
                    ns = oe
                elif abs(ns - o.start_ms) < self.snap_threshold_ms:
                    ns = o.start_ms
            self.selected_segment.start_ms = ns
            nl = max(0, min(4, int((event.pos().y() - 40) // (self.lane_height + self.lane_spacing))))
            self.selected_segment.lane = nl
            
        self.update_geometry()
        self.timelineChanged.emit()

    def mouseReleaseEvent(self, event):
        self.dragging = self.resizing = self.resizing_left = self.vol_dragging = self.fade_in_dragging = self.fade_out_dragging = self.slipping = self.setting_loop = self.resizing_timeline = False
        self.update_geometry()

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
                l_s = max(self.segments, key=lambda s: s.start_ms + s.duration_ms)
                start_ms = l_s.start_ms + l_s.duration_ms - 5000
                lane = l_s.lane
            else:
                start_ms = 0
        ns = TrackSegment(td, start_ms=start_ms, lane=lane)
        self.segments.append(ns)
        self.update_geometry()
        self.timelineChanged.emit()
        return ns
