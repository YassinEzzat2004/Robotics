"""
imu_sensor_mock.py
------------------
Simulates a Raspberry Pi / ESP32 with an IMU sensor (e.g. MPU-6050).
Sends random sensor readings over UDP to localhost:5005 every 50 ms.

Run this in a separate terminal BEFORE launching gui_main.py:
    python imu_sensor_mock.py
"""

import socket
import json
import time
import math
import random

HOST = "127.0.0.1"
PORT = 5005
INTERVAL = 0.05          # 20 Hz

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

t = 0.0
print(f"[IMU Mock] Sending to {HOST}:{PORT} at {1/INTERVAL:.0f} Hz  (Ctrl+C to stop)")

try:
    while True:
        t += INTERVAL

        # ── Accelerometer  (g)  – gentle sinusoidal drift + noise ──
        ax = math.sin(t * 0.7) * 0.3 + random.gauss(0, 0.02)
        ay = math.cos(t * 0.5) * 0.2 + random.gauss(0, 0.02)
        az = 1.0 + math.sin(t * 1.1) * 0.05 + random.gauss(0, 0.01)

        # ── Gyroscope  (°/s)  – slow drift + noise ──
        gx = math.sin(t * 0.3) * 15 + random.gauss(0, 0.5)
        gy = math.cos(t * 0.4) * 10 + random.gauss(0, 0.5)
        gz = math.sin(t * 0.2) * 8  + random.gauss(0, 0.3)

        # ── Magnetometer  (µT) ──
        mx = math.cos(t * 0.1) * 30 + random.gauss(0, 0.5)
        my = math.sin(t * 0.1) * 30 + random.gauss(0, 0.5)
        mz = 45.0 + random.gauss(0, 0.3)

        # ── Derived orientation (roll / pitch from accel) ──
        roll  = math.degrees(math.atan2(ay, az))
        pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
        yaw   = (t * 5) % 360          # fake integrated yaw

        # ── Temperature ──
        temp = 28.5 + math.sin(t * 0.05) * 1.5 + random.gauss(0, 0.05)

        payload = {
            "ts": round(t, 3),
            "accel": {"x": round(ax, 4), "y": round(ay, 4), "z": round(az, 4)},
            "gyro":  {"x": round(gx, 3), "y": round(gy, 3), "z": round(gz, 3)},
            "mag":   {"x": round(mx, 2), "y": round(my, 2), "z": round(mz, 2)},
            "orientation": {
                "roll":  round(roll, 2),
                "pitch": round(pitch, 2),
                "yaw":   round(yaw, 2),
            },
            "temp": round(temp, 2),
        }

        data = json.dumps(payload).encode()
        sock.sendto(data, (HOST, PORT))
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\n[IMU Mock] Stopped.")
finally:
    sock.close()