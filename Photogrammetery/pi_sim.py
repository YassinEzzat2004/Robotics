"""
pi_simulator.py
---------------
Simulates a Raspberry Pi that captures a camera frame, reads IMU sensors,
bundles everything into a single JSON packet, and sends it over UDP.

The JSON structure sent each tick:

{
    "ts":    float,          # elapsed seconds
    "frame": str,            # base64-encoded JPEG of the camera frame
    "accel": {"x","y","z"},  # accelerometer  (g)
    "gyro":  {"x","y","z"},  # gyroscope      (°/s)
    "mag":   {"x","y","z"},  # magnetometer   (µT)
    "orientation": {"roll","pitch","yaw"},
    "temp":  float           # °C
}

If no physical camera is available the simulator synthesises a test-pattern
frame (colour gradient + timestamp text) so the host side can still be
developed and tested without hardware.

Usage:
    python pi_simulator.py                  # camera 0, send to localhost:5005
    python pi_simulator.py --host 192.168.1.10 --port 5005 --camera 1
    python pi_simulator.py --no-camera      # force synthetic frames

Dependencies:
    pip install opencv-python numpy
"""

import argparse
import base64
import json
import math
import random
import socket
import time
import sys

import cv2
import numpy as np


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Raspberry Pi camera + IMU simulator")
parser.add_argument("--host",      default="127.0.0.1", help="Host to send to")
parser.add_argument("--port",      type=int, default=5005, help="UDP port")
parser.add_argument("--camera",    type=int, default=0,   help="Camera index")
parser.add_argument("--fps",       type=int, default=10,  help="Transmit rate (Hz)")
parser.add_argument("--quality",   type=int, default=60,  help="JPEG quality 1-95")
parser.add_argument("--width",     type=int, default=640, help="Frame width")
parser.add_argument("--height",    type=int, default=480, help="Frame height")
parser.add_argument("--no-camera", action="store_true",   help="Use synthetic frames")
args = parser.parse_args()

INTERVAL = 1.0 / args.fps
JPEG_QUALITY = args.quality
MAX_UDP = 60_000   # bytes — stay safely under the 65 507-byte UDP limit


# ── Camera init ───────────────────────────────────────────────────────────────
cap = None
if not args.no_camera:
    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        print(f"[Pi Sim] Camera {args.camera} opened  "
              f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
    else:
        cap.release()
        cap = None
        print("[Pi Sim] Camera not available — falling back to synthetic frames")

if cap is None:
    print(f"[Pi Sim] Generating synthetic {args.width}x{args.height} frames")


def synthetic_frame(t: float) -> np.ndarray:
    """Animated colour-gradient frame with a timestamp overlay."""
    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    # Scrolling hue gradient
    for x in range(args.width):
        hue = int((x / args.width * 180 + t * 20) % 180)
        frame[:, x] = cv2.cvtColor(
            np.array([[[hue, 200, 180]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
    # Timestamp
    cv2.putText(frame, f"t={t:.2f}s  SYNTHETIC", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def grab_frame(t: float) -> np.ndarray:
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            return frame
    return synthetic_frame(t)


def encode_frame(frame: np.ndarray) -> str:
    """JPEG-compress then base64-encode a BGR frame. Returns a UTF-8 string."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── IMU simulation ────────────────────────────────────────────────────────────
def imu_readings(t: float) -> dict:
    # Accelerometer (g) — gentle sinusoidal motion + noise
    ax = math.sin(t * 0.7) * 0.3  + random.gauss(0, 0.02)
    ay = math.cos(t * 0.5) * 0.2  + random.gauss(0, 0.02)
    az = 1.0 + math.sin(t * 1.1)  * 0.05 + random.gauss(0, 0.01)

    # Gyroscope (°/s) — slow drift + noise
    gx = math.sin(t * 0.3) * 15   + random.gauss(0, 0.5)
    gy = math.cos(t * 0.4) * 10   + random.gauss(0, 0.5)
    gz = math.sin(t * 0.2) * 8    + random.gauss(0, 0.3)

    # Magnetometer (µT) — slow heading rotation so the compass needle moves
    mx = math.cos(t * 0.15) * 30  + random.gauss(0, 0.5)
    my = math.sin(t * 0.15) * 30  + random.gauss(0, 0.5)
    mz = 45.0                      + random.gauss(0, 0.3)

    # Derived orientation
    roll  = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
    yaw   = (math.degrees(math.atan2(my, mx)) + 360) % 360  # mag-derived, matches radar

    temp = 28.5 + math.sin(t * 0.05) * 1.5 + random.gauss(0, 0.05)

    return {
        "accel": {"x": round(ax, 4), "y": round(ay, 4), "z": round(az, 4)},
        "gyro":  {"x": round(gx, 3), "y": round(gy, 3), "z": round(gz, 3)},
        "mag":   {"x": round(mx, 2), "y": round(my, 2), "z": round(mz, 2)},
        "orientation": {
            "roll":  round(roll, 2),
            "pitch": round(pitch, 2),
            "yaw":   round(yaw,  2),
        },
        "temp": round(temp, 2),
    }


# ── Main loop ─────────────────────────────────────────────────────────────────
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t = 0.0

print(f"[Pi Sim] Sending to {args.host}:{args.port} "
      f"at {args.fps} Hz  JPEG quality={JPEG_QUALITY}  (Ctrl+C to stop)")

try:
    while True:
        tick_start = time.monotonic()
        t += INTERVAL

        # 1. Grab and encode frame
        frame = grab_frame(t)
        frame_b64 = encode_frame(frame)

        # 2. Build combined packet
        packet = {"ts": round(t, 3), "frame": frame_b64}
        packet.update(imu_readings(t))

        # 3. Serialise
        data = json.dumps(packet).encode("utf-8")

        # 4. Warn if packet is too large for UDP
        if len(data) > MAX_UDP:
            print(f"[Pi Sim] WARNING: packet {len(data)} bytes exceeds UDP limit — "
                  f"lower --quality or --width/--height")

        # 5. Send
        sock.sendto(data, (args.host, args.port))

        # 6. Status line (overwrite in place)
        kb = len(data) / 1024
        sys.stdout.write(f"\r[Pi Sim] t={t:7.2f}s  packet={kb:5.1f} KB  "
                         f"frame={frame.shape[1]}x{frame.shape[0]}")
        sys.stdout.flush()

        # 7. Sleep for the remainder of the interval
        elapsed = time.monotonic() - tick_start
        sleep_t = INTERVAL - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

except KeyboardInterrupt:
    print("\n[Pi Sim] Stopped.")
finally:
    if cap is not None:
        cap.release()
    sock.close()