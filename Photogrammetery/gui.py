"""
gui_main.py
-----------
PyQt5 GUI that shows:
  • Live camera feed (with optional underwater simulation + ROI overlay)
  • Real-time IMU data received from imu_sensor_mock.py over UDP
  • Image quality pipeline controls (start/stop capture)
  • Mini orientation cube visualisation (ASCII-style roll/pitch/yaw gauges)

Dependencies:
    pip install PyQt5 opencv-python-headless numpy pyiqa torch torchvision

Run order:
    1.  python imu_sensor_mock.py   (in one terminal)
    2.  python gui_main.py          (in another terminal)
"""

import sys, json, socket, threading, time, math, os, queue, base64
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QFrame, QSizePolicy, QGroupBox, QProgressBar,
    QSlider, QCheckBox, QSplitter,
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QRectF, QPointF,
)
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QFontDatabase,
    QLinearGradient, QBrush, QPainterPath, QRadialGradient,
)

# ── import your existing pipeline ────────────────────────────────────────────
# Wrap in try so the GUI opens even without the full ML stack installed.
try:
    from video_pipeline import FramePipeline, FrameWorker, ROI, ImageEnhancer
    PIPELINE_AVAILABLE = True
except Exception as e:
    PIPELINE_AVAILABLE = False
    print(f"[WARN] Full pipeline not loaded: {e}")
    # ── Minimal stubs so force-capture still works without pyiqa/torch ──────
    import threading, queue as _queue, os as _os
    import numpy as _np

    class ImageEnhancer:
        def orthogonal_sharpener(self, image, threshold=130, alpha=0.3):
            import cv2 as _cv2
            lab = _cv2.cvtColor(image, _cv2.COLOR_BGR2LAB)
            l, a, b = _cv2.split(lab)
            l = l.astype(_np.float32)
            gx = _cv2.Sobel(l, _cv2.CV_32F, 1, 0, ksize=3)
            gy = _cv2.Sobel(l, _cv2.CV_32F, 0, 1, ksize=3)
            mag = _np.sqrt(gx**2 + gy**2)
            angle = _np.degrees(_np.arctan2(gy, gx))
            mask = (((_np.abs(angle) < 15) | (_np.abs(_np.abs(angle) - 90) < 15))
                    & (mag > threshold))
            kernel = _np.ones((3, 3), _np.uint8)
            mask = _cv2.dilate(mask.astype(_np.uint8), kernel,
                               iterations=1).astype(bool)
            l[mask] += alpha * mag[mask]
            l = _np.clip(l, 0, 255).astype(_np.uint8)
            return _cv2.cvtColor(_cv2.merge((l, a, b)), _cv2.COLOR_LAB2BGR)

    class _ForceCapture:
        def __init__(self, frame): self.frame = frame

    class FrameWorker(threading.Thread):
        def __init__(self, pipeline=None, queue_size=4,
                     on_persistent_reject=None, streak_threshold=3, **kw):
            super().__init__(daemon=True)
            self.pipeline               = pipeline
            self.queue                  = _queue.Queue(queue_size)
            self.stop_event             = threading.Event()
            self.counter                = 0
            self.forced_counter         = 0
            self._rej_streak            = 0
            self.streak_threshold       = streak_threshold
            self._on_persistent_reject  = on_persistent_reject
            self.enhancer               = ImageEnhancer()

        def run(self):
            while not self.stop_event.is_set():
                try:
                    item = self.queue.get(timeout=0.1)
                except _queue.Empty:
                    continue
                if isinstance(item, _ForceCapture):
                    self._handle_forced(item.frame)
                # normal frames silently discarded when pipeline unavailable
                self.queue.task_done()

        def _handle_forced(self, frame):
            import cv2 as _cv2
            sharpened = self.enhancer.orthogonal_sharpener(frame)
            _os.makedirs("modified", exist_ok=True)
            _os.makedirs("original", exist_ok=True)
            _cv2.imwrite(f"original/forced_{self.forced_counter}_raw.jpg", frame)
            _cv2.imwrite(f"modified/forced_{self.forced_counter}.jpg", sharpened)
            print(f"[FORCE] forced_{self.forced_counter}.jpg saved to modified/ (no pipeline)")
            self.forced_counter += 1

        def load(self, frame):
            pass   # no-op without pipeline

        def force_save(self, frame):
            try:
                self.queue.put(_ForceCapture(frame.copy()), block=True, timeout=1.0)
            except _queue.Full:
                print("[WARN] forced frame dropped – queue full")

        def stop(self):
            self.stop_event.set()

    class ROI:
        def __init__(self, frame_w, frame_h, w=320, h=320, step=10):
            self.w, self.h = w, h
            self.frame_w, self.frame_h = frame_w, frame_h
            self.x = frame_w // 2 - w // 2
            self.y = frame_h // 2 - h // 2
            self.dragging = False
            self.offset_x = self.offset_y = 0

        def resize(self, dw=0, dh=0):
            self.w = max(50, min(self.w + dw, self.frame_w - 20))
            self.h = max(50, min(self.h + dh, self.frame_h - 20))
            self.clamp()

        def mouse_event(self, event, mx, my, flags, param):
            import cv2 as _cv2
            if event == _cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                self.offset_x, self.offset_y = mx - self.x, my - self.y
            elif event == _cv2.EVENT_MOUSEMOVE and self.dragging:
                self.x, self.y = mx - self.offset_x, my - self.offset_y
                self.clamp()
            elif event == _cv2.EVENT_LBUTTONUP:
                self.dragging = False
            elif event == _cv2.EVENT_MOUSEWHEEL:
                d = 20 if flags > 0 else -20
                self.x -= d // 2; self.y -= d // 2
                self.w += d;      self.h += d
                self.clamp()

        def clamp(self):
            self.x = max(0, min(self.x, self.frame_w - self.w))
            self.y = max(0, min(self.y, self.frame_h - self.h))

        def crop(self, frame):
            return frame[self.y:self.y+self.h, self.x:self.x+self.w]

        def draw(self, frame):
            import cv2 as _cv2
            _cv2.rectangle(frame,
                           (self.x - 10, self.y - 10),
                           (self.x + self.w + 10, self.y + self.h + 10),
                           (0, 255, 0), 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (deep-sea / bioluminescent terminal theme)
# ═══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":        "#050d14",
    "panel":     "#08161f",
    "border":    "#0d3348",
    "accent":    "#00e5ff",
    "accent2":   "#00ff9d",
    "warn":      "#ff6b35",
    "danger":    "#ff2d55",
    "text":      "#c8e8f0",
    "text_dim":  "#3d6b7a",
    "grid":      "#071820",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  UDP LISTENER THREAD  –  receives combined Pi packets (frame + IMU)
# ═══════════════════════════════════════════════════════════════════════════════
class IMUListener(QThread):
    data_received = pyqtSignal(dict)   # IMU fields only — no numpy, safe for Qt

    def __init__(self, host="127.0.0.1", port=5005):
        super().__init__()
        self.host = host
        self.port = port
        self._running  = True
        self.frame_q   = queue.Queue(maxsize=2)   # latest decoded frame, main thread drains

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(0.3)
        while self._running:
            try:
                data, _ = sock.recvfrom(65536)
                payload = json.loads(data.decode("utf-8"))

                # ── Decode embedded frame if present ─────────────────────────
                frame_b64 = payload.pop("frame", None)
                if frame_b64:
                    jpg_bytes = base64.b64decode(frame_b64)
                    arr       = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        # Drop oldest if full so we always have the latest frame
                        if self.frame_q.full():
                            try: self.frame_q.get_nowait()
                            except queue.Empty: pass
                        self.frame_q.put_nowait(frame)

                # Emit IMU dict via Qt signal (safe — plain Python dict)
                self.data_received.emit(payload)

            except socket.timeout:
                pass
            except Exception as e:
                print(f"[Listener] error: {e}")
        sock.close()

    def stop(self):
        self._running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════════
#  ORIENTATION GAUGE WIDGET  (roll / pitch / yaw arcs)
# ═══════════════════════════════════════════════════════════════════════════════
class OrientationGauge(QWidget):
    def __init__(self, label="ROLL", color="#00e5ff", range_=(-180, 180)):
        super().__init__()
        self.label = label
        self.color = QColor(color)
        self.range_ = range_
        self.value = 0.0
        self.setMinimumSize(90, 90)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_value(self, v):
        self.value = v
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2 - 8

        # Background circle
        p.setPen(Qt.NoPen)
        bg = QRadialGradient(cx, cy, r)
        bg.setColorAt(0, QColor("#0a1f2e"))
        bg.setColorAt(1, QColor("#050d14"))
        p.setBrush(QBrush(bg))
        p.drawEllipse(QPointF(cx, cy), r, r)

        # Track arc
        pen = QPen(QColor(C["border"]), 5)
        pen.setCapStyle(Qt.FlatCap)
        p.setPen(pen)
        p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2), 0, 360 * 16)

        # Value arc
        lo, hi = self.range_
        span = hi - lo
        norm = max(0.0, min(1.0, (self.value - lo) / span))
        arc_pen = QPen(self.color, 5)
        arc_pen.setCapStyle(Qt.RoundCap)
        p.setPen(arc_pen)
        start_angle = 90 * 16          # 12 o'clock
        sweep = int(-norm * 360 * 16)
        p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2), start_angle, sweep)

        # Center text
        p.setPen(QPen(self.color))
        p.setFont(QFont("Courier New", 9, QFont.Bold))
        p.drawText(QRectF(cx - r, cy - 10, r * 2, 20),
                   Qt.AlignCenter, f"{self.value:.1f}°")

        # Label
        p.setPen(QPen(QColor(C["text_dim"])))
        p.setFont(QFont("Courier New", 7))
        p.drawText(QRectF(cx - r, cy + r - 14, r * 2, 16),
                   Qt.AlignCenter, self.label)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADING RADAR  –  hot/cold game
# ═══════════════════════════════════════════════════════════════════════════════
def _heading_color(heat: float) -> QColor:
    """
    heat: 0.0 (ice cold) → 1.0 (white-hot fire)
    0.0–0.4  : deep blue → cyan
    0.4–0.65 : cyan → yellow
    0.65–0.85: yellow → orange-red
    0.85–1.0 : red → white-hot
    """
    h = max(0.0, min(1.0, heat))
    if h < 0.4:
        t = h / 0.4
        return QColor(int(20 + t * 0),   int(80 + t * 180), int(200 + t * 55))   # blue→cyan
    elif h < 0.65:
        t = (h - 0.4) / 0.25
        return QColor(int(t * 255),       int(255 - t * 55), int(255 - t * 255))  # cyan→yellow
    elif h < 0.85:
        t = (h - 0.65) / 0.2
        return QColor(255,                int(200 - t * 170), 0)                  # yellow→red
    else:
        t = (h - 0.85) / 0.15
        return QColor(255,                int(30 + t * 225),  int(t * 225))       # red→white


_HOT_THRESHOLD  = 20.0   # degrees — "HOT"
_WARM_THRESHOLD = 60.0   # degrees — "WARM"


class HeadingRadarWidget(QWidget):
    """
    Compass rose showing:
      • Cardinal tick marks and labels
      • Rejected-frame target headings as glowing dots
      • Current heading needle coloured by proximity heat
      • A heat-bar below the compass
      • Status text: FREEZING / COLD / WARM / HOT / ON TARGET
    """

    def __init__(self):
        super().__init__()
        self.current_yaw    = 0.0          # live heading from IMU (degrees, 0–360)
        self.target_headings: list[float] = []   # headings of rejected frames

        self.setMinimumSize(260, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ── public API ────────────────────────────────────────────────────────────
    def set_heading(self, yaw: float):
        self.current_yaw = yaw % 360
        self.update()

    def add_target(self, heading: float):
        self.target_headings.append(heading % 360)
        self.update()

    def clear_targets(self):
        self.target_headings.clear()
        self.update()

    # ── heat computation ──────────────────────────────────────────────────────
    @staticmethod
    def _angular_delta(a: float, b: float) -> float:
        """Shortest angular distance between two headings, result in [0, 180]."""
        diff = abs(a - b) % 360
        return diff if diff <= 180 else 360 - diff

    def _closest_delta(self) -> float | None:
        """Smallest angular distance to any target (0-180). None if no targets."""
        if not self.target_headings:
            return None
        return min(self._angular_delta(self.current_yaw, t)
                   for t in self.target_headings)

    def _closest_target_index(self) -> int | None:
        if not self.target_headings:
            return None
        return min(range(len(self.target_headings)),
                   key=lambda i: self._angular_delta(self.current_yaw,
                                                      self.target_headings[i]))

    def remove_closest_target(self):
        """Pop the nearest target dot after a successful force-capture."""
        idx = self._closest_target_index()
        if idx is not None:
            self.target_headings.pop(idx)
            self.update()

    def is_hot(self) -> bool:
        """True when needle is within HOT_THRESHOLD degrees of any target."""
        d = self._closest_delta()
        return d is not None and d < _HOT_THRESHOLD

    def _heat(self) -> float:
        d = self._closest_delta()
        if d is None:
            return 0.0
        return max(0.0, 1.0 - d / 90.0)

    def _status_text(self) -> str:
        d = self._closest_delta()
        if d is None:
            return "NO TARGETS"
        if d < 5:                return "ON TARGET"
        if d < _HOT_THRESHOLD:   return "FIRE  —  CAPTURE READY"
        if d < _WARM_THRESHOLD:  return "WARM"
        if d < 110:              return "COLD"
        return "FREEZING"

    # ── paint ─────────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        heat      = self._heat()
        needle_c  = _heading_color(heat)
        status    = self._status_text()

        # layout: compass occupies top 80 %, heat bar + text bottom 20 %
        compass_h = int(H * 0.78)
        bar_y     = compass_h + 8
        cx, cy    = W / 2, compass_h / 2
        R         = min(cx, cy) - 14

        # ── background ───────────────────────────────────────────────────────
        p.fillRect(0, 0, W, H, QColor(C["bg"]))

        # subtle radial glow behind compass
        glow = QRadialGradient(cx, cy, R)
        gc = QColor(needle_c)
        gc.setAlpha(18)
        glow.setColorAt(0, gc)
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), R, R)

        # ── compass ring ─────────────────────────────────────────────────────
        ring_pen = QPen(QColor(C["border"]), 1.5)
        p.setPen(ring_pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), R, R)

        # ── cardinal ticks + labels ───────────────────────────────────────────
        cardinals = {0: "N", 90: "E", 180: "S", 270: "W"}
        for deg in range(0, 360, 10):
            rad    = math.radians(deg - 90)
            is_big = (deg % 90 == 0)
            is_mid = (deg % 30 == 0)
            tick_len = 10 if is_big else (6 if is_mid else 3)
            inner  = R - tick_len
            outer  = R
            ix = cx + inner * math.cos(rad)
            iy = cy + inner * math.sin(rad)
            ox_ = cx + outer * math.cos(rad)
            oy_ = cy + outer * math.sin(rad)
            pen_w = 1.5 if is_big else 0.8
            p.setPen(QPen(QColor(C["text_dim"] if not is_big else C["text"]), pen_w))
            p.drawLine(QPointF(ix, iy), QPointF(ox_, oy_))

            if is_big:
                lx = cx + (R + 13) * math.cos(rad)
                ly = cy + (R + 13) * math.sin(rad)
                p.setPen(QPen(QColor(C["accent"])))
                p.setFont(QFont("Courier New", 8, QFont.Bold))
                p.drawText(QRectF(lx - 10, ly - 7, 20, 14),
                           Qt.AlignCenter, cardinals[deg])

        # ── target heading dots ───────────────────────────────────────────────
        for t in self.target_headings:
            trad = math.radians(t - 90)
            dot_r = R - 18
            tx = cx + dot_r * math.cos(trad)
            ty = cy + dot_r * math.sin(trad)

            # glow halo — intensity by proximity
            delta = self._angular_delta(self.current_yaw, t)
            proximity = max(0.0, 1.0 - delta / 90.0)
            halo_c = _heading_color(proximity)
            halo_c.setAlpha(60 + int(proximity * 120))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(halo_c))
            p.drawEllipse(QPointF(tx, ty), 8, 8)

            # solid core
            core_c = QColor(C["danger"])
            p.setBrush(QBrush(core_c))
            p.setPen(QPen(QColor("#ffffff"), 0.8))
            p.drawEllipse(QPointF(tx, ty), 4, 4)

            # degree label
            p.setPen(QPen(QColor(C["text_dim"])))
            p.setFont(QFont("Courier New", 6))
            p.drawText(QRectF(tx - 15, ty + 6, 30, 10),
                       Qt.AlignCenter, f"{t:.0f}°")

        # ── needle ────────────────────────────────────────────────────────────
        nrad    = math.radians(self.current_yaw - 90)
        needle_len = R - 22
        tail_len   = 18
        nx  = cx + needle_len * math.cos(nrad)
        ny  = cy + needle_len * math.sin(nrad)
        ntx = cx - tail_len  * math.cos(nrad)
        nty = cy - tail_len  * math.sin(nrad)

        # glowing shadow
        shadow_pen = QPen(needle_c, 6)
        shadow_pen.setCapStyle(Qt.RoundCap)
        sc2 = QColor(needle_c); sc2.setAlpha(50)
        shadow_pen.setColor(sc2)
        p.setPen(shadow_pen)
        p.drawLine(QPointF(ntx, nty), QPointF(nx, ny))

        # main needle line
        needle_pen = QPen(needle_c, 2.5)
        needle_pen.setCapStyle(Qt.RoundCap)
        p.setPen(needle_pen)
        p.drawLine(QPointF(ntx, nty), QPointF(nx, ny))

        # arrowhead
        arrow_size = 8
        angle_left  = nrad + math.radians(145)
        angle_right = nrad - math.radians(145)
        alx = nx + arrow_size * math.cos(angle_left)
        aly = ny + arrow_size * math.sin(angle_left)
        arx = nx + arrow_size * math.cos(angle_right)
        ary = ny + arrow_size * math.sin(angle_right)
        arrow = QPainterPath()
        arrow.moveTo(nx, ny)
        arrow.lineTo(alx, aly)
        arrow.lineTo(arx, ary)
        arrow.closeSubpath()
        p.fillPath(arrow, QBrush(needle_c))

        # centre hub
        p.setBrush(QBrush(QColor(C["panel"])))
        p.setPen(QPen(needle_c, 1.5))
        p.drawEllipse(QPointF(cx, cy), 5, 5)

        # current heading text
        p.setPen(QPen(needle_c))
        p.setFont(QFont("Courier New", 9, QFont.Bold))
        p.drawText(QRectF(cx - 30, cy - 28, 60, 16),
                   Qt.AlignCenter, f"{self.current_yaw:.1f}°")

        # ── heat bar ─────────────────────────────────────────────────────────
        bar_margin = 20
        bar_w      = W - bar_margin * 2
        bar_h      = 8
        bx         = bar_margin

        # track
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(C["grid"])))
        p.drawRoundedRect(bx, bar_y, bar_w, bar_h, 4, 4)

        # fill gradient (always full cold→hot spectrum)
        bar_grad = QLinearGradient(bx, 0, bx + bar_w, 0)
        bar_grad.setColorAt(0.0,  QColor("#1455c8"))
        bar_grad.setColorAt(0.35, QColor("#00e5ff"))
        bar_grad.setColorAt(0.6,  QColor("#ffe600"))
        bar_grad.setColorAt(0.8,  QColor("#ff6b35"))
        bar_grad.setColorAt(1.0,  QColor("#ffffff"))
        p.setBrush(QBrush(bar_grad))
        fill_w = max(6, int(bar_w * heat))
        p.drawRoundedRect(bx, bar_y, fill_w, bar_h, 4, 4)

        # marker tick on bar
        tick_x = bx + int(bar_w * heat)
        p.setPen(QPen(QColor("#ffffff"), 1.5))
        p.drawLine(tick_x, bar_y - 2, tick_x, bar_y + bar_h + 2)

        # ── status label ─────────────────────────────────────────────────────
        p.setPen(QPen(needle_c))
        p.setFont(QFont("Courier New", 10, QFont.Bold))
        p.drawText(QRectF(0, bar_y + bar_h + 6, W, 22),
                   Qt.AlignCenter, status)

        # target count
        p.setPen(QPen(QColor(C["text_dim"])))
        p.setFont(QFont("Courier New", 7))
        p.drawText(QRectF(0, bar_y + bar_h + 26, W, 14),
                   Qt.AlignCenter,
                   f"{len(self.target_headings)} target(s) logged")


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA WIDGET  –  proper coordinate-mapped ROI interaction
# ═══════════════════════════════════════════════════════════════════════════════
class CameraWidget(QWidget):
    """
    Displays a scaled camera frame and forwards mouse events to the ROI object,
    translating Qt widget coordinates → original camera frame coordinates.

    Coordinate mapping
    ------------------
    The frame is drawn centred with KeepAspectRatio scaling.
    We track the rendered rect (self._img_rect) so every mouse event can be
    mapped:
        cam_x = (widget_x - img_rect.left) / scale
        cam_y = (widget_y - img_rect.top)  / scale
    """

    def __init__(self):
        super().__init__()
        self._pixmap        = None
        self._img_rect      = None   # (ox, oy, dw, dh, scale) of rendered image
        self._frame_w       = 1
        self._frame_h       = 1
        self.roi            = None   # set by MainWindow after camera opens
        self._resizing      = False
        self._resize_origin = (0, 0)

        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background:{C['bg']}; border:1px solid {C['border']};")

        # Accept mouse events
        self.setMouseTracking(True)

    # ── frame ingestion ──────────────────────────────────────────────────────
    def set_frame(self, bgr_frame):
        h, w, ch = bgr_frame.shape
        self._frame_w, self._frame_h = w, h
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        qi   = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qi)
        self.update()

    # ── painting ─────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(C["bg"]))

        if self._pixmap is None:
            p.setPen(QPen(QColor(C["text_dim"])))
            p.setFont(QFont("Courier New", 11))
            p.drawText(self.rect(), Qt.AlignCenter, "NO CAMERA SIGNAL")
            self._img_rect = None
            return

        # Scale keeping aspect ratio, centred
        pw, ph   = self._pixmap.width(), self._pixmap.height()
        ww, wh   = self.width(), self.height()
        scale    = min(ww / pw, wh / ph)
        dw, dh   = pw * scale, ph * scale
        ox       = (ww - dw) / 2
        oy       = (wh - dh) / 2
        self._img_rect = (ox, oy, dw, dh, scale)   # x, y, w, h, scale

        p.drawPixmap(int(ox), int(oy), int(dw), int(dh), self._pixmap)

        # Draw resize handle hint at bottom-right corner of ROI (in widget coords)
        if self.roi is not None:
            ox2, oy2, _, _, sc = self._img_rect
            rx = int(ox2 + (self.roi.x + self.roi.w) * sc)
            ry = int(oy2 + (self.roi.y + self.roi.h) * sc)
            hs = 10   # handle size px
            handle_pen = QPen(QColor(C["warn"]), 2)
            p.setPen(handle_pen)
            p.drawLine(rx - hs, ry, rx, ry)
            p.drawLine(rx, ry - hs, rx, ry)
            # hint text
            p.setPen(QPen(QColor(C["text_dim"])))
            p.setFont(QFont("Courier New", 7))
            p.drawText(int(ox2) + 4, int(oy2 + dh) - 4,
                       "LMB drag: move ROI  |  RMB drag: resize  |  Scroll: zoom")

    # ── coordinate helper ────────────────────────────────────────────────────
    def _to_cam(self, qx, qy):
        """Map widget pixel → camera pixel. Returns (cx, cy) or None."""
        if self._img_rect is None:
            return None
        ox, oy, dw, dh, scale = self._img_rect
        cx = (qx - ox) / scale
        cy = (qy - oy) / scale
        # clamp to frame bounds
        cx = max(0, min(self._frame_w - 1, cx))
        cy = max(0, min(self._frame_h - 1, cy))
        return int(cx), int(cy)

    # ── mouse events → ROI ───────────────────────────────────────────────────
    def mousePressEvent(self, e):
        if self.roi is None:
            return
        pt = self._to_cam(e.x(), e.y())
        if not pt:
            return
        if e.button() == Qt.LeftButton:
            self.roi.mouse_event(cv2.EVENT_LBUTTONDOWN, pt[0], pt[1], 0, None)
        elif e.button() == Qt.RightButton:
            # Right-click starts a resize from the bottom-right corner
            self._resizing   = True
            self._resize_origin = pt

    def mouseMoveEvent(self, e):
        if self.roi is None:
            return
        pt = self._to_cam(e.x(), e.y())
        if not pt:
            return
        if e.buttons() & Qt.LeftButton:
            self.roi.mouse_event(cv2.EVENT_MOUSEMOVE, pt[0], pt[1], 1, None)
        elif (e.buttons() & Qt.RightButton) and self._resizing:
            dx = pt[0] - self._resize_origin[0]
            dy = pt[1] - self._resize_origin[1]
            self._resize_origin = pt
            self.roi.resize(dx, dy)
        else:
            self.roi.mouse_event(cv2.EVENT_MOUSEMOVE, pt[0], pt[1], 0, None)
            # Change cursor when hovering inside ROI to hint draggability
            rx, ry, rw, rh = self.roi.x, self.roi.y, self.roi.w, self.roi.h
            if rx <= pt[0] <= rx + rw and ry <= pt[1] <= ry + rh:
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, e):
        if self.roi is None:
            return
        pt = self._to_cam(e.x(), e.y())
        if e.button() == Qt.LeftButton and pt:
            self.roi.mouse_event(cv2.EVENT_LBUTTONUP, pt[0], pt[1], 0, None)
        elif e.button() == Qt.RightButton:
            self._resizing = False

    def wheelEvent(self, e):
        if self.roi is None:
            return
        delta = e.angleDelta().y()
        flags = 1 if delta > 0 else -1
        pt = self._to_cam(e.x(), e.y())
        if pt:
            self.roi.mouse_event(cv2.EVENT_MOUSEWHEEL, pt[0], pt[1], flags, None)


# ═══════════════════════════════════════════════════════════════════════════════
#  NUMERIC VALUE TILE
# ═══════════════════════════════════════════════════════════════════════════════
def make_value_label(title="", color=C["accent"]):
    frame = QFrame()
    frame.setStyleSheet(
        f"background:{C['panel']}; border:1px solid {C['border']}; border-radius:4px;"
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(6, 4, 6, 4)
    layout.setSpacing(1)

    lbl_title = QLabel(title)
    lbl_title.setFont(QFont("Courier New", 7))
    lbl_title.setStyleSheet(f"color:{C['text_dim']}; border:none;")

    lbl_val = QLabel("–")
    lbl_val.setFont(QFont("Courier New", 11, QFont.Bold))
    lbl_val.setStyleSheet(f"color:{color}; border:none;")

    layout.addWidget(lbl_title)
    layout.addWidget(lbl_val)
    return frame, lbl_val


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSISTENT REJECT THUMBNAIL
# ═══════════════════════════════════════════════════════════════════════════════
class RejectedFrameWidget(QWidget):
    """
    Shows the last frame that was rejected 3 times in a row.
    Displays a badge showing the current streak count toward the next threshold.
    """
    STREAK_THRESHOLD = 3

    def __init__(self):
        super().__init__()
        self._pixmap      = None
        self._streak      = 0          # current streak (0–2), reset after display
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(
            f"background:{C['panel']}; border:1px solid {C['border']}; border-radius:4px;"
        )

    def set_frame(self, bgr_frame: np.ndarray):
        """Called when a persistent reject fires — display this frame and reset streak badge."""
        h, w, ch = bgr_frame.shape
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        qi  = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qi)
        self._streak = 0   # reset badge after display
        self.update()

    def set_streak(self, n: int):
        """Update the in-progress streak count (0 to threshold-1)."""
        self._streak = n % self.STREAK_THRESHOLD
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        p.fillRect(0, 0, W, H, QColor(C["panel"]))

        if self._pixmap is None:
            p.setPen(QPen(QColor(C["text_dim"])))
            p.setFont(QFont("Courier New", 8))
            p.drawText(self.rect(), Qt.AlignCenter, "NO PERSISTENT REJECT YET")
            return

        # Draw thumbnail centred
        thumb_w = W - 8
        thumb_h = H - 28
        scaled  = self._pixmap.scaled(thumb_w, thumb_h,
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
        tx = (W - scaled.width())  // 2
        ty = 4
        p.drawPixmap(tx, ty, scaled)

        # ── streak badge (pips) ───────────────────────────────────────────────
        pip_r   = 5
        pip_gap = 14
        total_w = self.STREAK_THRESHOLD * pip_gap - (pip_gap - pip_r * 2)
        bx      = (W - total_w) // 2
        by      = H - 16

        for i in range(self.STREAK_THRESHOLD):
            filled = i < self._streak
            cx_    = bx + i * pip_gap + pip_r
            color  = QColor(C["danger"]) if filled else QColor(C["border"])
            p.setPen(QPen(QColor(C["text_dim"]), 1))
            p.setBrush(QBrush(color))
            p.drawEllipse(QPointF(cx_, by), pip_r, pip_r)

        # label
        p.setPen(QPen(QColor(C["text_dim"])))
        p.setFont(QFont("Courier New", 6))
        p.drawText(QRectF(0, H - 10, W, 10), Qt.AlignCenter,
                   f"streak  {self._streak}/{self.STREAK_THRESHOLD}")



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AUR — Camera + IMU Monitor")
        self.resize(1280, 800)
        self.setStyleSheet(f"background:{C['bg']}; color:{C['text']};")

        # ── camera state ─────────────────────────────────────────────────────
        self.cap = None
        self.capture_active = False
        self.underwater_sim = True
        self.roi = None
        self.pipeline = FramePipeline() if PIPELINE_AVAILABLE else None
        self.enhancer = ImageEnhancer() if PIPELINE_AVAILABLE else None
        self.last_capture = 0.0
        self.capture_interval = 1.0   # seconds between saves
        self.frame_count = 0
        self.accepted_count = 0

        self._current_yaw   = 0.0   # latest yaw from IMU
        self._mag_heading   = 0.0   # latest magnetic heading derived from mx/my
        self.worker         = None
        self.forced_count   = 0
        self._last_rej_count = 0
        self._worker_owned_by_force = False
        self._persist_rej_queue = queue.Queue(maxsize=4)  # thread-safe frame pipe
        self._pi_frame      = None   # latest frame received from Pi over UDP

        self._build_ui()
        self._init_camera()
        self._init_imu_listener()

        # ── timers ────────────────────────────────────────────────────────────
        self.cam_timer = QTimer()
        self.cam_timer.timeout.connect(self._update_camera)
        self.cam_timer.start(33)   # ~30 fps

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # ── LEFT: camera + controls ──────────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Header — AUR logo
        header = QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aur_logo.png")
        if os.path.exists(logo_path):
            logo_pix = QPixmap(logo_path)
            scaled   = logo_pix.scaledToHeight(60, Qt.SmoothTransformation)
            header.setPixmap(scaled)
        else:
            header.setText("AUR")
            header.setFont(QFont("Courier New", 14, QFont.Bold))
            header.setStyleSheet(f"color:{C['accent']}; letter-spacing:4px;")
        header.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(header)

        # Camera
        self.cam_widget = CameraWidget()
        left_layout.addWidget(self.cam_widget, 1)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        self.btn_start = self._make_btn("▶  START CAPTURE", C["accent2"])
        self.btn_start.clicked.connect(self._toggle_capture)
        ctrl.addWidget(self.btn_start)

        self.btn_force = self._make_btn("⊕  FORCE CAPTURE", C["warn"])
        self.btn_force.clicked.connect(self._force_capture)
        ctrl.addWidget(self.btn_force)

        self.chk_sim = QCheckBox("Underwater Sim")
        self.chk_sim.setChecked(True)
        self.chk_sim.setStyleSheet(f"color:{C['text']}; font-family:Courier New;")
        self.chk_sim.stateChanged.connect(lambda s: setattr(self, "underwater_sim", bool(s)))
        ctrl.addWidget(self.chk_sim)

        left_layout.addLayout(ctrl)

        # Stats row
        stats = QHBoxLayout()
        stats.setSpacing(4)
        f, self.lbl_frames   = make_value_label("FRAMES",   C["accent"])
        a, self.lbl_accepted = make_value_label("ACCEPTED", C["accent2"])
        r, self.lbl_rejected = make_value_label("REJECTED", C["warn"])
        fc, self.lbl_forced  = make_value_label("FORCED",   "#c084fc")
        q, self.lbl_quality  = make_value_label("PIPELINE", C["text"])
        for w in [f, a, r, fc, q]:
            stats.addWidget(w)
        left_layout.addLayout(stats)

        self.lbl_quality.setText("READY" if PIPELINE_AVAILABLE else "UNAVAIL")
        self.lbl_quality.setStyleSheet(
            f"color:{'#00ff9d' if PIPELINE_AVAILABLE else C['warn']}; border:none; font-family:Courier New; font-size:11px; font-weight:bold;"
        )

        # ── Persistent reject thumbnail ───────────────────────────────────────
        rej_hdr = QLabel("PERSISTENT REJECT  ( 3× consecutive )")
        rej_hdr.setFont(QFont("Courier New", 7))
        rej_hdr.setStyleSheet(f"color:{C['text_dim']};")
        left_layout.addWidget(rej_hdr)

        self.rej_thumb = RejectedFrameWidget()
        left_layout.addWidget(self.rej_thumb)

        # ── RIGHT: IMU dashboard ─────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(370)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # IMU header
        imu_hdr = QLabel("◈  IMU TELEMETRY  ◈")
        imu_hdr.setFont(QFont("Courier New", 12, QFont.Bold))
        imu_hdr.setAlignment(Qt.AlignCenter)
        imu_hdr.setStyleSheet(f"color:{C['accent2']}; letter-spacing:3px;")
        right_layout.addWidget(imu_hdr)

        # Connection status
        self.lbl_conn = QLabel("● WAITING FOR SENSOR…")
        self.lbl_conn.setFont(QFont("Courier New", 8))
        self.lbl_conn.setAlignment(Qt.AlignCenter)
        self.lbl_conn.setStyleSheet(f"color:{C['warn']};")
        right_layout.addWidget(self.lbl_conn)

        # ── Orientation gauges ───────────────────────────────────────────────
        g_box = QGroupBox("ORIENTATION")
        g_box.setFont(QFont("Courier New", 8))
        g_box.setStyleSheet(
            f"QGroupBox{{color:{C['text_dim']}; border:1px solid {C['border']};"
            f"border-radius:4px; margin-top:8px; padding:6px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin; left:8px; padding:0 4px;}}"
        )
        g_row = QHBoxLayout(g_box)
        g_row.setSpacing(6)
        self.gauge_roll  = OrientationGauge("ROLL",  C["accent"],  (-180, 180))
        self.gauge_pitch = OrientationGauge("PITCH", C["accent2"], (-90, 90))
        self.gauge_yaw   = OrientationGauge("YAW",   C["warn"],    (0, 360))
        for g in [self.gauge_roll, self.gauge_pitch, self.gauge_yaw]:
            g_row.addWidget(g)
        right_layout.addWidget(g_box)

        # ── Numeric tiles ────────────────────────────────────────────────────
        tiles = QGridLayout()
        tiles.setSpacing(4)

        def tile_row(row, names, color):
            widgets = []
            for col, n in enumerate(names):
                f, lbl = make_value_label(n, color)
                tiles.addWidget(f, row, col)
                widgets.append(lbl)
            return widgets

        self.accel_lbls = tile_row(0, ["ACCEL X (g)", "ACCEL Y (g)", "ACCEL Z (g)"], C["accent"])
        self.gyro_lbls  = tile_row(1, ["GYRO X °/s", "GYRO Y °/s", "GYRO Z °/s"],   C["accent2"])
        self.mag_lbls   = tile_row(2, ["MAG X µT",   "MAG Y µT",   "MAG Z µT"],     "#c084fc")

        right_layout.addLayout(tiles)

        # Temperature
        temp_row = QHBoxLayout()
        tf, self.lbl_temp = make_value_label("TEMPERATURE (°C)", "#ff9f43")
        tsf, self.lbl_ts  = make_value_label("TIMESTAMP (s)",    C["text_dim"])
        temp_row.addWidget(tf)
        temp_row.addWidget(tsf)
        right_layout.addLayout(temp_row)

        # ── Heading radar (hot/cold game) ────────────────────────────────────
        radar_box = QGroupBox("HEADING RADAR  —  HOT / COLD")
        radar_box.setFont(QFont("Courier New", 8))
        radar_box.setStyleSheet(
            f"QGroupBox{{color:{C['text_dim']}; border:1px solid {C['border']};"
            f"border-radius:4px; margin-top:8px; padding:4px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin; left:8px; padding:0 4px;}}"
        )
        radar_layout = QVBoxLayout(radar_box)
        radar_layout.setContentsMargins(4, 12, 4, 4)
        self.heading_radar = HeadingRadarWidget()
        radar_layout.addWidget(self.heading_radar)

        right_layout.addWidget(radar_box, 1)

        # ── assemble ─────────────────────────────────────────────────────────
        root_layout.addWidget(left, 1)
        root_layout.addWidget(right)

    def target_headings_exist(self) -> bool:
        return len(self.heading_radar.target_headings) > 0

    # ─────────────────────────────────────────────────────────────────────────
    def _make_btn(self, text, color):
        btn = QPushButton(text)
        btn.setFont(QFont("Courier New", 9, QFont.Bold))
        btn.setStyleSheet(
            f"QPushButton{{background:{C['panel']}; color:{color}; border:1px solid {color};"
            f"border-radius:3px; padding:5px 12px; letter-spacing:1px;}}"
            f"QPushButton:hover{{background:{color}22;}}"
            f"QPushButton:pressed{{background:{color}44;}}"
        )
        return btn

    # ─────────────────────────────────────────────────────────────────────────
    def _init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            return
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]
            self.roi = ROI(w, h, 320, 320)   # always — ROI has no pipeline dependency
            self.cam_widget.roi = self.roi

    # ─────────────────────────────────────────────────────────────────────────
    def _init_imu_listener(self):
        self.imu_thread = IMUListener()
        self.imu_thread.data_received.connect(self._on_imu_data)
        self.imu_thread.start()

    # ─────────────────────────────────────────────────────────────────────────
    def _toggle_capture(self):
        self.capture_active = not self.capture_active
        if self.capture_active:
            import os, shutil
            for folder in ["original", "modified", "rejected"]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                os.makedirs(folder)
            self.heading_radar.clear_targets()
            self.frame_count    = 0
            self.accepted_count = 0
            self.forced_count   = 0
            self._last_rej_count = 0
            # drain any leftover frames from previous session
            while not self._persist_rej_queue.empty():
                try: self._persist_rej_queue.get_nowait()
                except queue.Empty: break
            self.rej_thumb.set_streak(0)
            self.lbl_frames.setText("0")
            self.lbl_accepted.setText("0")
            self.lbl_rejected.setText("0")
            self.lbl_forced.setText("0")

            # Always start the worker — force_save works even without the pipeline
            self.worker = FrameWorker(
                self.pipeline,
                on_persistent_reject=self._on_persistent_reject_cb,
            )
            self.worker.start()

            self.btn_start.setText("■  STOP CAPTURE")
            self.btn_start.setStyleSheet(
                self.btn_start.styleSheet().replace(C["accent2"], C["danger"])
            )
        else:
            self.btn_start.setText("▶  START CAPTURE")
            self.btn_start.setStyleSheet(
                self.btn_start.styleSheet().replace(C["danger"], C["accent2"])
            )
            # Drain and stop worker
            if self.worker:
                self.worker.queue.join()
                self.worker.stop()
                self.worker = None

    # ─────────────────────────────────────────────────────────────────────────
    def _force_capture(self):
        """
        Force-save the current ROI without quality evaluation.
        Works at any time — does NOT require START CAPTURE to be active.
        Only blocked when not in the HOT zone (no targets yet = always allowed).
        A worker is created on-demand if the session hasn't started.
        On success, the matched target dot is removed from the radar.
        """
        if self.roi is None:
            return
        if self._pi_frame is None and self.cap is None:
            return   # no frame source at all

        # ── HOT-zone gate: blocked only when targets exist but needle is cold ─
        if self.target_headings_exist() and not self.heading_radar.is_hot():
            orig_style = self.btn_force.styleSheet()
            deny_style = orig_style.replace(C["warn"], C["danger"])
            self.btn_force.setStyleSheet(deny_style)
            QTimer.singleShot(300, lambda: self.btn_force.setStyleSheet(orig_style))
            return

        # ── Ensure output folders exist ───────────────────────────────────────
        os.makedirs("modified", exist_ok=True)
        os.makedirs("original", exist_ok=True)

        # ── Spin up a worker on demand if capture session isn't running ───────
        if self.worker is None:
            self.worker = FrameWorker(
                self.pipeline,
                on_persistent_reject=self._on_persistent_reject_cb,
            )
            self.worker.start()
            self._worker_owned_by_force = True
        else:
            self._worker_owned_by_force = False

        # ── Grab freshest frame ───────────────────────────────────────────────
        if self._pi_frame is not None:
            frame = self._pi_frame.copy()
        elif self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                return
        else:
            return
        display = frame.copy()
        if self.underwater_sim:
            b, g, r = cv2.split(display)
            r = cv2.multiply(r, 0.5)
            b = cv2.multiply(b, 1.2)
            g = cv2.multiply(g, 1.1)
            display = cv2.merge([b, g, r])

        roi_img = self.roi.crop(display)
        self.worker.force_save(roi_img)

        # ── Remove matched target from radar ──────────────────────────────────
        self.heading_radar.remove_closest_target()

        # ── Update forced counter ─────────────────────────────────────────────
        self.forced_count += 1
        self.lbl_forced.setText(str(self.forced_count))

        # ── Flash white to confirm ────────────────────────────────────────────
        orig_style = self.btn_force.styleSheet()
        flash = orig_style.replace(C["warn"], "#ffffff")
        self.btn_force.setStyleSheet(flash)
        QTimer.singleShot(120, lambda: self.btn_force.setStyleSheet(orig_style))

    def _on_persistent_reject_cb(self, frame: np.ndarray):
        """
        Called from the worker thread when streak_threshold consecutive rejections
        have occurred. Puts the frame into a thread-safe queue; the camera timer
        drains it on the main thread to update the thumbnail safely.
        """
        try:
            self._persist_rej_queue.put_nowait(frame)
        except queue.Full:
            pass   # drop if GUI is lagging — next persistent reject will arrive soon

    # ─────────────────────────────────────────────────────────────────────────
    def _update_camera(self):
        # ── Pull latest frame: Pi queue first, then local VideoCapture ────────
        frame = None
        try:
            while True:   # drain so we always get the newest
                frame = self.imu_thread.frame_q.get_nowait()
        except queue.Empty:
            pass

        if frame is not None:
            self._pi_frame = frame
            # First Pi frame ever — create ROI from real dimensions
            if self.roi is None:
                h, w = frame.shape[:2]
                self.roi = ROI(w, h, 320, 320)
                self.cam_widget.roi = self.roi
        elif self._pi_frame is not None:
            frame = self._pi_frame          # reuse last received Pi frame
        elif self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                return
        else:
            return

        display = frame.copy()

        # Underwater colour sim
        if self.underwater_sim:
            b, g, r = cv2.split(display)
            r = cv2.multiply(r, 0.5)
            b = cv2.multiply(b, 1.2)
            g = cv2.multiply(g, 1.1)
            display = cv2.merge([b, g, r])

        # Draw ROI
        if self.roi is not None:
            self.roi.draw(display)

        self.cam_widget.set_frame(display)

        # ── Drain persistent-reject pipe (worker thread → main thread) ────────
        try:
            while True:
                rej_frame = self._persist_rej_queue.get_nowait()
                self.rej_thumb.set_frame(rej_frame)
                # Also log this heading to the radar each time a persistent reject fires
                self.heading_radar.add_target(self._mag_heading)
        except queue.Empty:
            pass

        # ── Update streak badge live so pilot sees progress ───────────────────
        if self.worker is not None:
            self.rej_thumb.set_streak(self.worker._rej_streak)

        # Capture logic
        if self.capture_active and self.worker and self.roi:
            now = time.monotonic()
            if now - self.last_capture > self.capture_interval:
                self.last_capture = now
                roi_img = self.roi.crop(display)
                self.worker.load(roi_img)
                # We can't know accepted/rejected synchronously now (worker is async),
                # so update totals by peeking at worker counters each tick
                self.frame_count    = self.worker.counter
                self.accepted_count = self.worker.counter - (
                    len([f for f in os.listdir("rejected") if f.endswith(".jpg")])
                    if os.path.exists("rejected") else 0
                )
                rej = self.frame_count - self.accepted_count
                self.lbl_frames.setText(str(self.frame_count))
                self.lbl_accepted.setText(str(max(0, self.accepted_count)))
                self.lbl_rejected.setText(str(max(0, rej)))

    # ─────────────────────────────────────────────────────────────────────────
    def _on_imu_data(self, data: dict):
        # Connection badge
        self.lbl_conn.setText("● SENSOR ONLINE")
        self.lbl_conn.setStyleSheet(f"color:{C['accent2']}; font-family:Courier New; font-size:8pt;")

        # Orientation gauges (roll/pitch/yaw still shown for reference)
        orient = data.get("orientation", {})
        self.gauge_roll.set_value(orient.get("roll",  0))
        self.gauge_pitch.set_value(orient.get("pitch", 0))
        self.gauge_yaw.set_value(orient.get("yaw",    0))

        # Magnetometer → single consistent heading used for BOTH needle and targets
        mag = data.get("mag", {})
        for lbl, key in zip(self.mag_lbls, ["x", "y", "z"]):
            lbl.setText(f"{mag.get(key, 0):+.1f}")
        mx = mag.get("x", 0)
        my = mag.get("y", 0)
        # Standard compass heading: 0=N, 90=E, clockwise
        self._mag_heading = (math.degrees(math.atan2(my, mx)) + 360) % 360

        # Heading radar needle always tracks the MAGNETIC heading
        self.heading_radar.set_heading(self._mag_heading)

        # Accel
        acc = data.get("accel", {})
        for lbl, key in zip(self.accel_lbls, ["x", "y", "z"]):
            lbl.setText(f"{acc.get(key, 0):+.4f}")

        # Gyro
        gyro = data.get("gyro", {})
        for lbl, key in zip(self.gyro_lbls, ["x", "y", "z"]):
            lbl.setText(f"{gyro.get(key, 0):+.2f}")

        # Temp + timestamp
        self.lbl_temp.setText(f"{data.get('temp', 0):.2f}")
        self.lbl_ts.setText(f"{data.get('ts', 0):.1f}")

    # ─────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.cam_timer.stop()
        if self.cap:
            self.cap.release()
        if self.worker:
            self.worker.queue.join()
            self.worker.stop()
        self.imu_thread.stop()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())