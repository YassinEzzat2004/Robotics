# Underwater Camera Quality Pipeline + IMU Monitor

A real-time image-quality pipeline and PyQt5 dashboard for an underwater (or simulated underwater) camera rig, with live IMU telemetry, a region-of-interest (ROI) selector, and a "heading radar" hot/cold game that guides a pilot back toward headings where good shots were missed.

The system is split into three independently runnable pieces that talk to each other over UDP / local files:

| File | Role |
|---|---|
| `video_pipeline.py` | Core image evaluation/enhancement pipeline, ROI logic, and a standalone OpenCV viewer (camera or video file) |
| `gui_main.py` | PyQt5 GUI: live camera/Pi feed, capture controls, quality stats, IMU telemetry, orientation gauges, heading radar |
| `pi_simulator.py` | Simulates a Raspberry Pi: grabs camera frames (or synthesizes them), reads fake IMU data, and streams both as a single JSON/UDP packet |
| `imu_sensor_mock.py` | Lightweight standalone IMU-only mock (no camera frame), for testing the GUI's telemetry panel without a Pi simulator |

---

## 1. Concept

The rig is meant to simulate an underwater drone/camera capturing frames as it moves. Raw frames are typically degraded (underexposed, blurry, flat, color-cast, noisy), so a pipeline automatically evaluates each captured frame and applies targeted corrections. Frames that still fail after enhancement are flagged; if the **same spot** keeps failing 3 times in a row, that heading is logged as a "target" on a compass radar so the pilot can manually steer back and force-capture a shot at that exact heading later.

---

## 2. Components in detail

### 2.1 `video_pipeline.py`

#### `ImageEvaluator`
Runs a battery of checks on a frame and returns a list of failure reasons:

- **Exposure** (`mean_L`, `low_clip`, `high_clip`) — mean lightness in LAB space outside `60–190`, or too many near-black/near-white pixels.
- **Blur** (`blur`) — Laplacian variance ≤ 50.
- **Feature richness** (`flat`) — ORB keypoint density ≤ 20% of max.
- **Color cast** (`color_cast`) — blue/green vs. red channel ratio > 3 (typical underwater cast).
- **Quality** (`quality`) — only checked if everything else passes; uses the **BRISQUE** no-reference quality metric (`pyiqa`) and flags if score > 50.

#### `ImageEnhancer`
Applies targeted corrections only for the failures that were actually detected:

- `enhance_exposure` — CLAHE (adaptive histogram equalization) on the L channel.
- `enhance_sharpness` — basic 3×3 sharpening kernel.
- `orthogonal_sharpener` — Sobel-gradient-aware sharpening that boosts only near-horizontal/near-vertical high-gradient edges (used on **every** frame before evaluation, and also as the **only** processing step on force-captured frames).
- `enhance_features` — histogram equalization for low-detail/flat scenes.
- `denoise` — `fastNlMeansDenoisingColored` for BRISQUE-flagged frames.
- `color_correction` — shifts LAB a/b channels toward neutral gray to fix color cast.

#### `FramePipeline`
Orchestrates evaluate → (if failed) enhance → re-evaluate, and returns `(image, passed)`.

#### `ROI`
A draggable/resizable region-of-interest rectangle with:
- Keyboard movement (`w a s d`) and resizing (`e r f g`) for the OpenCV demo.
- Mouse drag/resize/scroll-zoom support, also reused by the PyQt5 `CameraWidget`.

#### `FrameWorker`
A background `threading.Thread` with an internal queue, so frame processing never blocks the capture loop:

- **Normal path** (`load()`) — applies `orthogonal_sharpener`, runs it through `FramePipeline`, and saves the result to `modified/` or `rejected/` (raw copy always goes to `original/`). Tracks **consecutive rejections**; every time the streak hits `streak_threshold` (default 3), it fires an `on_persistent_reject` callback with the rejected frame and resets the streak.
- **Forced path** (`force_save()`) — bypasses `FramePipeline` entirely. Only `orthogonal_sharpener` is applied, then the frame is written straight to `modified/forced_*.jpg` (raw copy to `original/forced_*_raw.jpg`). Used for the pilot's manual "I know this is a good shot" override.

#### `Video`
A self-contained OpenCV demo app (camera or video file) that wires all of the above together with hotkeys:

| Key | Action |
|---|---|
| `Esc` | Quit |
| `t` / `y` | Start / stop automatic capture (camera mode) |
| `c` | Force-capture current ROI (bypasses evaluation) |
| `w a s d` | Move ROI |
| `e r f g` | Resize ROI (file mode) |
| mouse drag/scroll | Move/resize/zoom ROI |

Run directly:
```bash
python video_pipeline.py            # camera 0, underwater simulation off by default settings
python video_pipeline.py video.mp4  # process a video file, 600x600 ROI, no underwater filter
```

---

### 2.2 `gui_main.py`

A PyQt5 dashboard combining the camera pipeline with live IMU telemetry from `pi_simulator.py` / `imu_sensor_mock.py`.

**Graceful degradation:** if `pyiqa`/`torch`/`video_pipeline` fail to import, the GUI falls back to lightweight internal stubs (`ImageEnhancer`, `ROI`, `FrameWorker`) so the window still opens and **force-capture still works** — only automatic quality evaluation is unavailable (status tile shows `UNAVAIL`).

**Layout:**

- **Left panel**
  - AUR logo / title header
  - `CameraWidget` — renders the live feed (own frames or frames received from the Pi simulator over UDP) with aspect-ratio-correct scaling and full mouse coordinate mapping back to the underlying ROI object (LMB drag = move, RMB drag = resize, scroll = zoom).
  - **▶ START CAPTURE** — clears `original/`, `modified/`, `rejected/`, resets all counters and the heading radar, and starts a `FrameWorker`. Toggling again stops it.
  - **⊕ FORCE CAPTURE** — saves the current ROI immediately, bypassing evaluation. Can be used even without START CAPTURE active (spins up its own worker on demand). If targets are logged on the heading radar, this button is **gated**: it only works when the heading needle is in the "hot" zone (flashes red and refuses otherwise).
  - "Underwater Sim" checkbox — toggles the synthetic blue/green color cast applied to the displayed/saved frame.
  - Stat tiles: `FRAMES`, `ACCEPTED`, `REJECTED`, `FORCED`, `PIPELINE` (READY/UNAVAIL).
  - **Persistent reject thumbnail** — shows the last frame that triggered 3 consecutive rejections, plus a live "pip" streak badge (0–3) showing progress toward the next trigger.

- **Right panel — IMU Telemetry**
  - Connection status indicator (waiting / online).
  - Three `OrientationGauge` arc widgets for roll/pitch/yaw.
  - Numeric tiles for accelerometer (g), gyroscope (°/s), magnetometer (µT).
  - Temperature and timestamp tiles.
  - **Heading Radar** — a compass-rose widget that is the centerpiece of the "hot/cold" workflow:
    - The needle always tracks the **magnetometer-derived heading** (not the IMU's separately-computed yaw), so it matches the simulator's `mag` field exactly.
    - Each time a persistent (3×) rejection fires, the current heading is logged as a glowing "target" dot on the compass.
    - A heat bar + status text (`FREEZING` → `COLD` → `WARM` → `FIRE — CAPTURE READY` → `ON TARGET`) tells the pilot how close the current heading is to the nearest unresolved target.
    - Successfully force-capturing while hot removes the nearest target dot.

**Data flow for camera frames:**
1. `IMUListener` (a `QThread`) binds a UDP socket on `127.0.0.1:5005`, receives JSON packets, decodes the embedded base64 JPEG (if present) into the thread-safe `frame_q`, and emits the rest of the payload as a `data_received` Qt signal (IMU fields only — no numpy objects cross thread boundaries via Qt signals).
2. The 30 fps `cam_timer` in `MainWindow` drains `frame_q` for the newest Pi frame (falling back to a local `cv2.VideoCapture` if no Pi frames have arrived), applies the underwater color simulation, draws the ROI, and updates `CameraWidget`.
3. `_on_imu_data` updates gauges/tiles and recomputes `_mag_heading` from `mx`/`my`, which drives the heading radar needle.
4. Persistent rejects are pushed onto the worker thread via a callback, queued in `_persist_rej_queue`, and drained on the main thread's timer tick to update the thumbnail and log a new radar target safely.

**Run order:**
```bash
python pi_simulator.py        # or imu_sensor_mock.py for IMU-only testing
python gui_main.py
```

---

### 2.3 `pi_simulator.py`

Stands in for the physical Raspberry Pi. Each tick (default 10 Hz, configurable):

1. Grabs a frame from a real camera if available, otherwise synthesizes an animated HSV gradient frame with a timestamp overlay.
2. JPEG-encodes and base64-encodes the frame.
3. Generates simulated IMU readings (accelerometer, gyroscope, magnetometer, derived roll/pitch/yaw, temperature) using sinusoids + Gaussian noise so all gauges move realistically.
4. Bundles frame + IMU into one JSON packet and sends it over UDP to the GUI, warning if the packet would exceed the safe UDP payload size (60 KB headroom under the 65 507-byte limit).

```bash
python pi_simulator.py
python pi_simulator.py --host 192.168.1.10 --port 5005 --camera 1
python pi_simulator.py --no-camera --fps 15 --quality 50 --width 480 --height 360
```

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Destination host |
| `--port` | `5005` | UDP port |
| `--camera` | `0` | Camera index |
| `--fps` | `10` | Transmit rate (Hz) |
| `--quality` | `60` | JPEG quality (1–95) |
| `--width` / `--height` | `640` / `480` | Capture resolution |
| `--no-camera` | off | Force synthetic test-pattern frames |

### 2.4 `imu_sensor_mock.py`

A simpler standalone mock that sends **IMU-only** packets (no `frame` field) at a fixed 20 Hz to `127.0.0.1:5005`. Useful for testing the telemetry panel, gauges, and heading radar in isolation without needing a camera or `pi_simulator.py` running. Note its `yaw` is a separately-integrated value (`(t * 5) % 360`), distinct from the magnetometer-derived heading used elsewhere — since it has no `frame` field, the GUI will simply show "NO CAMERA SIGNAL" and fall back to any local webcam.

---

## 3. Installation

```bash
pip install PyQt5 opencv-python numpy pyiqa torch torchvision
```

> `opencv-python-headless` also works for the simulator scripts, but `gui_main.py`'s fallback `cv2` usage and `video_pipeline.py`'s `cv2.imshow`/`cv2.namedWindow` calls require the full `opencv-python` (GUI-enabled) build.

If `pyiqa`/`torch` are unavailable, `gui_main.py` still runs in degraded mode (force-capture only, no automatic evaluation).

---

## 4. Typical session

```bash
# Terminal 1 — simulate the Pi (camera + IMU)
python pi_simulator.py

# Terminal 2 — launch the dashboard
python gui_main.py
```

1. Click **START CAPTURE** — drag/resize the ROI box over the area of interest.
2. Watch `FRAMES` / `ACCEPTED` / `REJECTED` update as the pipeline evaluates each captured frame once per second.
3. If a spot keeps failing 3× in a row, it appears as a red dot on the **Heading Radar** and a thumbnail appears under "PERSISTENT REJECT."
4. Steer back toward that heading — the radar needle changes color and the status text counts down from `FREEZING` to `FIRE — CAPTURE READY`.
5. Click **FORCE CAPTURE** while "hot" to save a sharpened, unevaluated shot directly to `modified/forced_*.jpg` and clear that target.

---

## 5. Output directory structure

Each session (`START CAPTURE`) resets and populates:

```
original/    # raw, unmodified ROI crops (one per processed frame, plus forced_*_raw.jpg)
modified/    # frames that passed evaluation after enhancement, plus all forced_*.jpg
rejected/    # frames that still failed evaluation after enhancement
```

---

## 6. Notes & limitations

- `ImageEvaluator.evaluate_brisque` runs on CPU by default (`pyiqa.create_metric('brisque', device='cpu')`) — adjust if you have a GPU available and want faster scoring.
- UDP has no delivery guarantee or ordering; on a lossy link, occasional frames/IMU samples will simply be dropped, which is acceptable for a live telemetry feed but means this protocol is not suitable as-is for anything safety-critical.
- The heading radar's "target" headings are derived purely from the magnetometer (`mag.x`, `mag.y`) reading at the moment a persistent rejection fires — it does not account for vehicle motion between that moment and when the pilot circles back.
- `imu_sensor_mock.py`'s `yaw` and the magnetometer-derived heading used by the radar are computed independently and will drift apart over time; use `pi_simulator.py` if you want them consistent.
