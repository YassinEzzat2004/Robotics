import cv2
from time import monotonic
import numpy as np
import pyiqa
import torch
import torchvision.transforms as T
import threading
import queue
import os
import shutil

from time import sleep
# ============================================================
# ---------------------- ENHANCER -----------------------------
# ============================================================

class ImageEnhancer:
    def enhance_exposure(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def enhance_sharpness(self, image):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def orthogonal_sharpener(self, image, threshold=130, alpha=0.3):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = l.astype(np.float32)
        gx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.degrees(np.arctan2(gy, gx))
        mask_v = (np.abs(angle) < 15) & (magnitude > threshold)
        mask_h = (np.abs(np.abs(angle) - 90) < 15) & (magnitude > threshold)
        mask = mask_v | mask_h
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        l[mask] += alpha * magnitude[mask]
        l = np.clip(l, 0, 255).astype(np.uint8)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return image

    def enhance_features(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def denoise(self, image):
        return cv2.fastNlMeansDenoisingColored(
            image, None, h=7, hColor=7,
            templateWindowSize=7, searchWindowSize=21
        )

    def color_correction(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        a += (128 - np.mean(a)) * 0.3
        b += (128 - np.mean(b)) * 0.3
        a = np.clip(a, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply(self, image, failures):
        enhanced = image.copy()
        if {"mean_L", "low_clip", "high_clip"} & set(failures):
            enhanced = self.enhance_exposure(enhanced)
        if "blur" in failures:
            enhanced = self.enhance_sharpness(enhanced)
        if "flat" in failures:
            enhanced = self.enhance_features(enhanced)
        if "color_cast" in failures:
            enhanced = self.color_correction(enhanced)
        if "quality" in failures:
            enhanced = self.denoise(enhanced)
        return enhanced


# ============================================================
# ---------------------- EVALUATOR ----------------------------
# ============================================================

class ImageEvaluator:
    def __init__(self, maxfeatures=800):
        self.maxfeatures = maxfeatures
        self.orb = cv2.ORB_create(nfeatures=maxfeatures)
        self.model = pyiqa.create_metric('brisque', device='cpu')
        self.model.eval()

    def evaluate_exposure(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        return np.mean(L), np.mean(L < 10) * 100, np.mean(L > 245) * 100

    def evaluate_blur(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def evaluate_feature_richness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints = self.orb.detect(gray, None)
        return len(keypoints) / self.maxfeatures

    def color_cast_score(self, image):
        b, g, r = cv2.split(image)
        return (np.mean(b) + np.mean(g)) / (2 * (np.mean(r) + 1e-6))

    @torch.no_grad()
    def evaluate_brisque(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = T.ToTensor()(image).unsqueeze(0)
        return self.model(tensor).item()

    def evaluate(self, image):
        failures = []
        mean_L, low_clip, high_clip = self.evaluate_exposure(image)
        blur_score = self.evaluate_blur(image)
        feature_sat = self.evaluate_feature_richness(image)
        cast = self.color_cast_score(image)
        if not (60 < mean_L < 190):
            failures.append("mean_L")
        if low_clip >= 5:
            failures.append("low_clip")
        if high_clip >= 7:
            failures.append("high_clip")
        if blur_score <= 50:
            failures.append("blur")
        if feature_sat <= 0.2:
            failures.append("flat")
        if cast > 3:
            failures.append("color_cast")
        if not failures:
            if self.evaluate_brisque(image) > 50:
                failures.append("quality")
        return failures, len(failures) == 0


# ============================================================
# ---------------------- PIPELINE -----------------------------
# ============================================================

class FramePipeline:
    def __init__(self):
        self.evaluator = ImageEvaluator()
        self.enhancer  = ImageEnhancer()
        self.counter   = 0

    def process(self, image):
        failures, passed = self.evaluator.evaluate(image)
        if passed:
            return image, True
        print(f"Frame {self.counter} BEFORE: {failures}")
        enhanced = self.enhancer.apply(image, failures)
        failures_after, passed = self.evaluator.evaluate(enhanced)
        if failures_after:
            print(f"Frame {self.counter} AFTER: {failures_after}")
        self.counter += 1
        return enhanced, passed


# ============================================================
# ---------------------- ROI --------------------------------
# ============================================================

class ROI:
    def __init__(self, frame_w, frame_h, w=320, h=320, step=10):
        self.w = w
        self.h = h
        self.step = step
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.x = frame_w // 2 - w // 2
        self.y = frame_h // 2 - h // 2
        self.corner_size = 15
        self.dragging = False
        self.resizing = False
        self.offset_x = 0
        self.offset_y = 0

    def move(self, key):
        if key == ord('a'):
            self.x -= self.step
        elif key == ord('d'):
            self.x += self.step
        elif key == ord('w'):
            self.y -= self.step
        elif key == ord('s'):
            self.y += self.step
        self.clamp()

    def resize(self, dw=0, dh=0):
        self.w += dw
        self.h += dh
        self.w = max(50, min(self.w, self.frame_w - 20))
        self.h = max(50, min(self.h, self.frame_h - 20))
        self.clamp()

    def resize_by_key(self, key):
        if key == ord('e'):
            self.resize(20, 0)
        elif key == ord('r'):
            self.resize(-20, 0)
        elif key == ord('f'):
            self.resize(0, 20)
        elif key == ord('g'):
            self.resize(0, -20)

    def mouse_event(self, event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.offset_x = mx - self.x
            self.offset_y = my - self.y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.x = mx - self.offset_x
                self.y = my - self.offset_y
                self.clamp()
            elif self.resizing:
                self.w = mx - self.x
                self.h = my - self.y
                self.clamp()
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.resizing = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 20 if flags > 0 else -20
            self.x -= delta // 2
            self.y -= delta // 2
            self.w += delta
            self.h += delta
        self.clamp()

    def clamp(self):
        self.x = max(0, min(self.x, self.frame_w - self.w))
        self.y = max(0, min(self.y, self.frame_h - self.h))

    def crop(self, frame):
        return frame[self.y:self.y+self.h, self.x:self.x+self.w]

    def draw(self, frame):
        cv2.rectangle(frame,
                      (self.x - 10, self.y - 10),
                      (self.x + self.w + 10, self.y + self.h + 10),
                      (0, 255, 0), 2)


# ============================================================
# ---------------------- WORKER ------------------------------
# ============================================================

class _ForceCapture:
    """
    Sentinel wrapper placed in the FrameWorker queue to request a
    forced save that bypasses FramePipeline evaluation entirely.
    Only orthogonal_sharpener is applied before writing to forced/.
    """
    def __init__(self, frame: np.ndarray):
        self.frame = frame


class FrameWorker(threading.Thread):
    def __init__(self, pipeline, queue_size=4, save_size=(256, 256),
                 on_persistent_reject=None, streak_threshold=3):
        super().__init__(daemon=True)
        self.pipeline       = pipeline
        self.queue          = queue.Queue(queue_size)
        self.stop_event     = threading.Event()
        self.counter        = 0
        self.forced_counter = 0
        self.save_size      = save_size
        self.enhancer       = ImageEnhancer()

        # ── consecutive-rejection streak tracking ─────────────────────────────
        self.streak_threshold    = streak_threshold   # how many in a row needed
        self._rej_streak         = 0                  # current consecutive count
        self._on_persistent_reject = on_persistent_reject  # callback(frame) or None

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if isinstance(item, _ForceCapture):
                self._handle_forced(item.frame)
            else:
                self._handle_normal(item)

            self.queue.task_done()

    # ── normal path ───────────────────────────────────────────────────────────
    def _handle_normal(self, frame: np.ndarray):
        """
        Sharpener → evaluation → modified|rejected.
        Tracks consecutive rejections; fires on_persistent_reject callback
        every time the streak hits streak_threshold, then resets the counter.
        An accepted frame resets the streak to zero.
        """
        frame = self.enhancer.orthogonal_sharpener(frame)
        if self.pipeline is not None:
            enhanced, accepted = self.pipeline.process(frame)
        else:
            enhanced, accepted = frame, True

        folder = "modified" if accepted else "rejected"
        cv2.imwrite(f"original/frame_{self.counter}_raw.jpg", frame)
        cv2.imwrite(f"{folder}/frame_{self.counter}.jpg", enhanced)
        self.counter += 1

        if accepted:
            # Any acceptance resets the streak
            self._rej_streak = 0
        else:
            self._rej_streak += 1
            if self._rej_streak >= self.streak_threshold:
                # Fire callback with the enhanced rejected frame
                if self._on_persistent_reject is not None:
                    try:
                        self._on_persistent_reject(enhanced.copy())
                    except Exception as e:
                        print(f"[WARN] on_persistent_reject callback error: {e}")
                # Reset so the next threshold is another 3 consecutive rejections
                self._rej_streak = 0

    # ── forced path ───────────────────────────────────────────────────────────
    def _handle_forced(self, frame: np.ndarray):
        """
        Bypass FramePipeline entirely.
        Applies only orthogonal_sharpener (non-destructive edge sharpening),
        then writes directly to forced/ with no quality gating.
        The raw frame is also preserved in original/ for reference.
        """
        sharpened = self.enhancer.orthogonal_sharpener(frame)
        os.makedirs("modified", exist_ok=True)
        os.makedirs("original", exist_ok=True)
        cv2.imwrite(f"original/forced_{self.forced_counter}_raw.jpg", frame)
        cv2.imwrite(f"modified/forced_{self.forced_counter}.jpg", sharpened)
        print(f"[FORCE] forced_{self.forced_counter}.jpg saved to modified/ (evaluation skipped)")
        self.forced_counter += 1

    # ── public loaders ────────────────────────────────────────────────────────
    def load(self, frame: np.ndarray):
        """Queue a normal (evaluated) frame. Drops silently if queue full."""
        try:
            self.queue.put(frame.copy(), block=False)
        except queue.Full:
            print("[WARN] Worker queue full – normal frame dropped")

    def force_save(self, frame: np.ndarray):
        """
        Queue a forced capture that bypasses all evaluation.
        Blocks up to 1 s so the pilot's explicit capture is never
        silently dropped even under heavy load.
        """
        try:
            self.queue.put(_ForceCapture(frame.copy()), block=True, timeout=1.0)
        except queue.Full:
            print("[WARN] Worker queue full – forced frame dropped")

    def stop(self):
        self.stop_event.set()


# ============================================================
# ---------------------- VIDEO APP ---------------------------
# ============================================================

class Video:
    def __init__(self, roi_w=320, roi_h=320, path=0, filter=True):
        self.path      = path
        self.filter    = filter
        self.pipeline  = FramePipeline()
        self.last_time = monotonic()
        self.running   = False
        self.roi_h     = roi_h
        self.roi_w     = roi_w

        if isinstance(path, int):
            self.camera = cv2.VideoCapture(path)
            ret, frame  = self.camera.read()
            if not ret:
                raise RuntimeError("Camera not available")
            h, w = frame.shape[:2]
            self.roi = ROI(w, h, self.roi_w, self.roi_h)
        else:
            self.camera = None
            self.roi    = None

    def _init_output_folders(self):
        for folder in ["original", "modified", "rejected"]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)

    def make_underwater(self, frame):
        if not self.filter:
            return frame
        b, g, r = cv2.split(frame)
        r = cv2.multiply(r, 0.5)
        b = cv2.multiply(b, 1.2)
        g = cv2.multiply(g, 1.1)
        return cv2.merge([b, g, r])

    def run_from_camera(self):
        self._init_output_folders()
        worker = FrameWorker(self.pipeline)
        worker.start()
        cv2.namedWindow("Camera")
        cv2.setMouseCallback("Camera", self.roi.mouse_event)

        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            sim     = self.make_underwater(frame)
            self.roi.draw(sim)
            roi_img = self.roi.crop(sim)

            cv2.imshow("Camera", sim)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord('t'):
                self.running = True
                print("Processing STARTED")
            elif key == ord('y'):
                self.running = False
                print("Processing STOPPED")
            elif key == ord('c'):
                # Force-capture hotkey – bypasses evaluation
                worker.force_save(roi_img)
            else:
                self.roi.move(key)

            if self.running and monotonic() - self.last_time > 1:
                worker.load(roi_img)
                self.last_time = monotonic()

        worker.queue.join()
        worker.stop()
        worker.join()
        self.camera.release()
        cv2.destroyAllWindows()

    def run_from_file(self):
        self._init_output_folders()
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.roi = ROI(width, height, self.roi_w, self.roi_h)

        worker = FrameWorker(self.pipeline)
        worker.start()
        cv2.namedWindow("Video")
        cv2.setMouseCallback("Video", self.roi.mouse_event)
        self.last_time = monotonic()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            sleep(0.033)

            sim     = self.make_underwater(frame)
            self.roi.draw(sim)
            roi_img = self.roi.crop(sim)

            cv2.imshow("Video", sim)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord('c'):
                worker.force_save(roi_img)
            else:
                self.roi.move(key)
                self.roi.resize_by_key(key)

            if monotonic() - self.last_time > 1:
                worker.load(roi_img)
                self.last_time = monotonic()

        cap.release()
        worker.queue.join()
        worker.stop()
        worker.join()
        cv2.destroyAllWindows()

    def run(self):
        if isinstance(self.path, str):
            self.run_from_file()
        else:
            self.run_from_camera()


# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        app = Video(roi_h=600, roi_w=600, path=sys.argv[1], filter=False)
    else:
        app = Video(path=0)

    app.run()