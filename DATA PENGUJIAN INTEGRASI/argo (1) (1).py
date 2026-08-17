#!/usr/bin/env python3
"""
  Arsitektur komunikasi (v8):
    UI Frontend (ARGO_UI.html)
        ↕ WebSocket (ws://localhost:8765)
    argo_camera.py  ← LiDAR USB (Delta2G)  [opsional]
        ↕ Serial USB 25000 bps
    Arduino Mega (ARGO_Mega_v5.ino)
        ↕ Serial3 (TX3/RX3 pin 14/15)
    ESP32 (motor + MQTT IoT)

  Pesan dari HTML → Python:
    {type:"command",      mode:"STANDBY"|"PICKUP"|"DELIVERY"}
    {type:"set_nav_mode", mode:1|2|3}
    {type:"set_delivery_status", status:"STANDBY|PICKUP|DELIVERY"}
    {type:"waypoint",     waypoints:[{lat,lon},...], navMode:"ekf"|"odo"|"gps"}
    {type:"waypoint_odo", waypoints:[{x,y},...]}
    {type:"dwa_config",   max_speed:0.6,...}
    {type:"serial_cmd",   cmd:"..."}
    {type:"clrwp"}
    {type:"reset"}
    {type:"door"}
    {type:"dwa_start"} / {type:"dwa_stop"}
    {type:"estop"}
    {type:"ping"}
    {type:"get_config"}

  Pesan dari Python → HTML:
    {type:"telemetry",    lat_raw, lon_raw, heading, speed, ...}
    {type:"lidar",        points:[[x,y],...FRAME ROBOT meter: x=depan,y=kiri...], count, hz}  // hasil FILTER; DWA yg ubah ke global
    {type:"status",       serial_connected, lidar_connected, dwa_running, ...}
    {type:"ack",          action:"...", success:true/false}
    {type:"log",          source:"mega", msg:"..."}
    {type:"nmea",         line:"$GPGGA,..."}
    {type:"wp_event",     event:"reached"|"done"|"route_received", index/count:n}
    {type:"pin_event",    event:"pin_check", correct:true/false}
    {type:"gps_event",    event:"ref_set"|"lost"|"ready"}
    {type:"mode_feedback",nav_mode:n, nav_mode_str:"..."}
    {type:"nav_mode_changed", nav_mode:n}
    {type:"delivery_status", status:"..."}
    {type:"ready"}
    {type:"dwa_config_info", config:{...}}
    {type:"pong",         ts:..., mega_ok:..., lidar_ok:...}

  Dependencies:
    pip install websockets pyserial numpy

"""

import asyncio
import json
import math
import os
import struct
import sys
import time
import threading
import queue
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional, Set

import numpy as np

try:
    import enet_camera_bridge as cam
except Exception:
    try:
        import enet_camera_bridge_v4 as cam  # type: ignore
    except Exception:
        cam = None
        print("[WARN] enet_camera_bridge tidak ditemukan -> mode kamera nonaktif.")

# ── Dependency check ──────────────────────────────────────────────
try:
    import serial
except ImportError:
    print("[ERROR] Install dulu: pip install pyserial")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("[ERROR] Install dulu: pip install websockets")
    sys.exit(1)

# =============================================================================
#  KONFIGURASI
# =============================================================================
CONFIG = {
    "mega_port":    "COM15",
    "lidar_port":   "COM3",
    "mega_baud":    250000,   
    "lidar_baud":   230400,
    "ws_host":      "0.0.0.0",
    "ws_port":      8765,
    "use_lidar":    True,   
    "send_to_mega": True,
    "dwa_enabled":  True,
    "use_camera":   False, 
    "show_preview": True, 
    "log_level":    "INFO",
    "log_csv":      True,
}

esp32_pwm = {"l": None, "r": None, "ts": 0.0}

# =============================================================================
#  LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ARGO")
logging.getLogger("websockets").setLevel(logging.WARNING)

# Waktu start untuk uptime
_start_time = time.time()


# =============================================================================
#  NAVIGATION MODE
# =============================================================================
class NavMode:
    GPS_EKF_DWA  = 1  
    ODO_DWA      = 2
    GPS_EKF_ONLY = 3   
    STANDBY  = "STANDBY"
    PICKUP   = "PICKUP"
    DELIVERY = "DELIVERY"
    STR_MAP = {
        0: "STANDBY",
        1: "GPS+EKF+DWA",
        2: "ODO+DWA",
        3: "GPS+EKF (Waypoint)",
    }

    COMMAND_MAP = {
        "STANDBY":  {"delivery": "STANDBY",  "mega_mode": 0},
        "PICKUP":   {"delivery": "PICKUP",   "mega_mode": 1},
        "DELIVERY": {"delivery": "DELIVERY", "mega_mode": 1},
    }


# =============================================================================
#  DWA CONFIG
# =============================================================================
class RobotType(Enum):
    circle    = 0
    rectangle = 1


class DWAConfig:
    def __init__(self):
        self.max_speed             = 0.6
        self.min_speed             = -0.3  
        self.max_yaw_rate          = math.radians(90.0)   
        self.max_accel             = 1.0
        self.max_delta_yaw_rate    = math.radians(180.0)  
        self.v_resolution          = 0.05
        self.yaw_rate_resolution   = math.radians(3.0)
        self.dt                    = 0.1
        self.predict_time          = 3.0   
        self.to_goal_cost_gain     = 1.0    
        self.speed_cost_gain       = 1.5   
        self.obstacle_cost_gain    = 1.8    
        self.robot_stuck_flag_cons = 0.001
        self.robot_type            = RobotType.rectangle
        self.robot_radius          = 1.0
        self.robot_width           = 0.45
        self.robot_length          = 0.65
        self.goal_tolerance        = 2

    def update_from_dict(self, d: dict):
        mapping = {
            "max_speed":           ("max_speed",           float),
            "min_speed":           ("min_speed",           float),
            "max_yaw_rate":        ("max_yaw_rate",        lambda v: math.radians(float(v))),
            "max_accel":           ("max_accel",           float),
            "v_resolution":        ("v_resolution",        float),
            "yaw_rate_resolution": ("yaw_rate_resolution", lambda v: math.radians(float(v))),
            "dt":                  ("dt",                  float),
            "predict_time":        ("predict_time",        float),
            "to_goal_cost_gain":   ("to_goal_cost_gain",   float),
            "speed_cost_gain":     ("speed_cost_gain",     float),
            "obstacle_cost_gain":  ("obstacle_cost_gain",  float),
            "robot_width":         ("robot_width",         float),
            "robot_length":        ("robot_length",        float),
            "goal_tolerance":      ("goal_tolerance",      float),
        }
        for key, (attr, conv) in mapping.items():
            if key in d:
                try:
                    setattr(self, attr, conv(d[key]))
                except Exception as e:
                    log.warning(f"Config update error [{key}]: {e}")
        log.info(f"DWA config updated: max_v={self.max_speed} ob_gain={self.obstacle_cost_gain}")

    def to_dict(self) -> dict:
        return {
            "max_speed":           self.max_speed,
            "min_speed":           self.min_speed,
            "max_yaw_rate":        math.degrees(self.max_yaw_rate),
            "max_accel":           self.max_accel,
            "v_resolution":        self.v_resolution,
            "yaw_rate_resolution": math.degrees(self.yaw_rate_resolution),
            "dt":                  self.dt,
            "predict_time":        self.predict_time,
            "to_goal_cost_gain":   self.to_goal_cost_gain,
            "speed_cost_gain":     self.speed_cost_gain,
            "obstacle_cost_gain":  self.obstacle_cost_gain,
            "robot_width":         self.robot_width,
            "robot_length":        self.robot_length,
            "goal_tolerance":      self.goal_tolerance,
        }


dwa_config = DWAConfig()


CAM_LOOKAHEAD_M   = 1.5                 
CAM_SETPOINT_SLEW = math.radians(120)   
CAM_MAX_ABS_DEG   = 80.0             
CAM_SIGN          = -1.0              

class CamHeading:
    """Bentuk TARGET HEADING (rad, frame robot, + = KIRI) dari tali keep-left.
    Python tidak ber-PID; hanya membentuk setpoint halus."""
    def __init__(self):
        self.th_cmd   = 0.0
        self.had_lane = False
        self.last_log = 0.0  

    @staticmethod
    def _wrap(a):
        return math.atan2(math.sin(a), math.cos(a))

    def _carrot(self, kl, Ld):
        if kl is None or len(kl) == 0:
            return None
        depan = kl[:, 0]
        ahead = np.where(depan >= Ld)[0]
        if len(ahead) == 0:
            return kl[int(np.argmax(depan))]
        return kl[int(ahead[int(np.argmin(depan[ahead]))])]

    def update(self, kl, dt):
        """Return (th_cmd, th_raw) rad, atau (None, None) bila tali hilang."""
        c = self._carrot(kl, CAM_LOOKAHEAD_M)
        if c is None:
            self.had_lane = False
            return None, None
        depan, samping = float(c[0]), float(c[1])
        th_raw = math.atan2(samping, depan)
        if not self.had_lane:
            self.th_cmd = th_raw
            self.had_lane = True
        d = self._wrap(th_raw - self.th_cmd)
        d = max(-CAM_SETPOINT_SLEW * dt, min(CAM_SETPOINT_SLEW * dt, d))
        self.th_cmd = self._wrap(self.th_cmd + d)
        return self.th_cmd, th_raw

cam_heading = CamHeading()
_dwa_pool = ThreadPoolExecutor(max_workers=1)

ENABLE_STUCK_ESCAPE = True


# =============================================================================
#    "A"   = tepi jalan & hazard KAMERA jadi obstacle DWA (butuh kamera/--keepline)
#    "B"   = bias hindaran & rotasi MENJAUHI trotoar (tanpa kamera)
#    "C"   = keduanya
#    "off" = nonaktif
# =============================================================================
ANTI_TROTOAR_MODE = "B"
KEEP_SIDE         = "left"   # sisi yang ditempel robot: "left" -> trotoar di KIRI
SIDE_BIAS_GAIN    = 0.8      # [B] kekuatan hukuman belok ke sisi trotoar (0 = off)
CAM_OB_MAX_PTS    = 40       # [A] maks titik obstacle kamera yg dimasukkan ke DWA

_USE_CAM_EDGE  = ANTI_TROTOAR_MODE in ("A", "C")
_USE_SIDE_BIAS = ANTI_TROTOAR_MODE in ("B", "C")
_SAFE_TURN_DIR = -1.0 if KEEP_SIDE == "left" else 1.0


# =============================================================================
#  [SPEEDBUMP] Turunkan kecepatan saat KAMERA mendeteksi speedbump.
# =============================================================================
BUMP_SLOWDOWN    = True 
BUMP_SPEED       = 0.3     # m/s: batas kecepatan saat speedbump terdeteksi
BUMP_ON_APPROACH = True


def bump_active():
    """[SPEEDBUMP] True bila kamera mendeteksi speedbump (perlu pelan)."""
    if not BUMP_SLOWDOWN or cam is None or not CONFIG.get("use_camera"):
        return False
    try:
        b = cam.get_bump()
    except Exception:
        return False
    active = (b == "on_bump") or (b == "approaching" and BUMP_ON_APPROACH)
    if b != getattr(bump_active, "_last", "__init__"):
        bump_active._last = b
        if b:
            log.info(f"[SPEEDBUMP] kamera deteksi '{b}' -> slowdown={active} "
                     f"(target {BUMP_SPEED} m/s)")
        else:
            log.info("[SPEEDBUMP] speedbump hilang -> kecepatan normal")
    return active

# =============================================================================
#  DWA ALGORITMA
# =============================================================================
def motion(x: np.ndarray, u: list, dt: float) -> np.ndarray:
    x = x.copy()
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def calc_dynamic_window(x: np.ndarray, cfg: DWAConfig) -> list:
    Vs = [cfg.min_speed, cfg.max_speed, -cfg.max_yaw_rate, cfg.max_yaw_rate]
    Vd = [
        x[3] - cfg.max_accel * cfg.dt,
        x[3] + cfg.max_accel * cfg.dt,
        x[4] - cfg.max_delta_yaw_rate * cfg.dt,
        x[4] + cfg.max_delta_yaw_rate * cfg.dt,
    ]
    return [
        max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
        max(Vs[2], Vd[2]), min(Vs[3], Vd[3]),
    ]

def predict_trajectory(x_init: np.ndarray, v: float, y: float, cfg: DWAConfig) -> np.ndarray:
    x = x_init.copy()
    traj = [x.copy()]
    t = 0.0
    while t <= cfg.predict_time:
        x = motion(x, [v, y], cfg.dt)
        traj.append(x.copy())
        t += cfg.dt
    return np.array(traj)


def calc_obstacle_cost(trajectory: np.ndarray, ob: np.ndarray, cfg: DWAConfig) -> float:
    if ob is None or len(ob) == 0:
        return 0.0

    ox, oy = ob[:, 0], ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r  = np.hypot(dx, dy)

    safety_margin   = 0.30   
    safety_distance = 1.3    

    if cfg.robot_type == RobotType.rectangle:
        yaw  = trajectory[:, 2]
        rot  = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw),  np.cos(yaw)]])
        rot  = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ r_ for r_ in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])

        half_len = cfg.robot_length / 2 + safety_margin
        half_wid = cfg.robot_width  / 2 + safety_margin
        if (np.logical_and(
            np.logical_and(local_ob[:, 0] <=  half_len,
                           local_ob[:, 1] <=  half_wid),
            np.logical_and(local_ob[:, 0] >= -half_len,
                           local_ob[:, 1] >= -half_wid)
        )).any():
            return float("inf")
    elif cfg.robot_type == RobotType.circle:
        if (r <= cfg.robot_radius + safety_margin).any():
            return float("inf")

    min_r     = float(np.min(r))
    body_half = (max(cfg.robot_length, cfg.robot_width) / 2) + safety_margin
    clearance = min_r - body_half
    if clearance < safety_distance:
        return 1.0 / max(clearance, 1e-3)
    return 1.0 / min_r


def calc_to_goal_cost(trajectory: np.ndarray, goal: np.ndarray) -> float:
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    err_angle  = math.atan2(dy, dx)
    cost_angle = err_angle - trajectory[-1, 2]
    return abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))


def calc_control_and_trajectory(x, dw, cfg, goal, ob):
    x_init   = x.copy()
    min_cost = float("inf")
    best_u   = [0.0, 0.0]
    best_traj = np.array([x])
    for v in np.arange(dw[0], dw[1], cfg.v_resolution):
        for y in np.arange(dw[2], dw[3], cfg.yaw_rate_resolution):
            traj    = predict_trajectory(x_init, v, y, cfg)
            g_cost  = cfg.to_goal_cost_gain  * calc_to_goal_cost(traj, goal)
            s_cost  = cfg.speed_cost_gain    * (cfg.max_speed - traj[-1, 3])
            if traj[-1, 3] < 0:
                s_cost += 2.0 * cfg.speed_cost_gain * abs(traj[-1, 3])
            ob_cost = cfg.obstacle_cost_gain * calc_obstacle_cost(traj, ob, cfg)
            cost    = g_cost + s_cost + ob_cost
    
            if _USE_SIDE_BIAS:
                cost += SIDE_BIAS_GAIN * max(0.0, (-_SAFE_TURN_DIR) * y)
            better = cost < min_cost - 1e-9
            tie    = abs(cost - min_cost) <= 1e-9 and abs(y) < abs(best_u[1])
            if better or tie:
                min_cost  = cost
                best_u    = [v, y]
                best_traj = traj

    return best_u, best_traj


def dwa_control(x, cfg, goal, ob):
    dw = calc_dynamic_window(x, cfg)
    return calc_control_and_trajectory(x, dw, cfg, goal, ob)


def camera_obstacles_global(x):
    """[ANTI-TROTOAR opsi A] Ambil tepi jalan (corridor) & hazard interior dari
    KAMERA (frame robot [depan,samping]) lalu transform ke GLOBAL pakai pose robot.
    Dipakai sbg obstacle TAMBAHAN khusus perencanaan DWA agar robot tidak membelok
    keluar jalan / ke trotoar. Rumus transform sama dgn mask_to_obstacles_global."""
    if cam is None or not CONFIG.get("use_camera"):
        return np.empty((0, 2))
    try:
        corridor, hazards = cam.get_obstacles()
    except Exception:
        return np.empty((0, 2))
    pts = []
    for arr in (corridor, hazards):
        if arr is not None and len(arr):
            pts.append(np.asarray(arr, float).reshape(-1, 2))
    if not pts:
        return np.empty((0, 2))
    rob = np.vstack(pts)                
    depan = rob[:, 0]; samping = rob[:, 1]
    cy = math.cos(x[2]); sy = math.sin(x[2])
    ox = x[0] + depan * cy - samping * sy
    oy = x[1] + depan * sy + samping * cy
    g = np.column_stack([ox, oy])
    if len(g) > CAM_OB_MAX_PTS:
        d = np.hypot(g[:, 0] - x[0], g[:, 1] - x[1])
        g = g[np.argsort(d)[:CAM_OB_MAX_PTS]]
    return g



# =============================================================================
#  INTEGRASI GUIDANCE ARBITER
# =============================================================================
class GuidanceArbiter:
    def __init__(self):
        self.avoid       = False     
        self.enter_dist  = 2.5       
        self.clear_dist  = 3.0       
        self.clear_hold   = 1.0     
        self._clear_since = None     
        self.front_memory   = 0.7    
        self._last_front    = float("inf")
        self._last_front_ts = 0.0
        self.front_arc   = math.radians(90.0)   
        self.stay_in_dwa = False
        self.cruise_speed = 0.5     

    def nearest_front_dist(self, x, ob):
        """Jarak rintangan terdekat di DEPAN robot (m). inf jika kosong."""
        if ob is None or len(ob) == 0:
            return float("inf")
        rel = ob - x[:2]
        d   = np.hypot(rel[:, 0], rel[:, 1])
        ang = np.arctan2(rel[:, 1], rel[:, 0]) - x[2]
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        front = np.abs(ang) < self.front_arc
        if not front.any():
            return float("inf")
        return float(np.min(d[front]))

    def update_state(self, min_front):
        """Histeresis + debounce waktu: cegah flip-flop akibat deteksi LiDAR putus-putus.
        AVOID dipicu seketika (keselamatan), tapi kembali CRUISE hanya setelah jalur
        bebas BERTURUT-TURUT selama clear_hold detik."""
        now = time.time()

        if math.isinf(min_front):
            if now - self._last_front_ts <= self.front_memory:
                min_front = self._last_front
        else:
            self._last_front    = min_front
            self._last_front_ts = now
        if not self.avoid:
            if min_front < self.enter_dist:
                self.avoid = True
                self._clear_since = None
                log.info(f"[ARBITER] Rintangan {min_front:.2f} m -> AVOID (DWA ambil alih)")
        else:
            if self.stay_in_dwa:
    
                pass
            elif min_front > self.clear_dist:
                if self._clear_since is None:
                    self._clear_since = now
                elif now - self._clear_since >= self.clear_hold:
                    self.avoid = False
                    self._clear_since = None
                    log.info(f"[ARBITER] Bebas {min_front:.2f} m (>= {self.clear_hold:.1f}s) -> CRUISE (PID global)")
            else:
                self._clear_since = None
        return self.avoid


arbiter = GuidanceArbiter()

# =============================================================================
#  STUCK-RECOVERY
# =============================================================================
class StuckRecovery:
    IDLE, REVERSE, ROTATE = 0, 1, 2

    def __init__(self):
        self.state         = self.IDLE
        self.stuck_v_thr   = 0.05                  # m/s
        self.stuck_time    = 1.5                   # s
        self.reverse_dist  = 1.0                   # m
        self.reverse_v     = -0.3                  # m/s
        self.rear_dist     = 1.2                   # m
        self.rear_arc      = math.radians(70.0)
        self.side_dist     = 3.0                   # m
        self.rotate_v      = 0.15                  # m/s
        self.rotate_rate   = math.radians(45.0)    # rad/s
        self.rotate_angle  = math.radians(70.0) 
        self._moving_since = None
        self._ref_xy       = None
        self._ref_yaw      = None
        self._turn_dir     = -1.0                  # -1=KANAN(CW), +1=KIRI(CCW)

    def _rear_blocked(self, x, ob):
        if ob is None or len(ob) == 0:
            return False
        rel = ob - x[:2]
        d   = np.hypot(rel[:, 0], rel[:, 1])
        ang = np.arctan2(rel[:, 1], rel[:, 0]) - x[2]
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        rear = (np.abs(np.abs(ang) - math.pi) < self.rear_arc) & (d < self.rear_dist)
        return bool(rear.any())

    def _dominant_side(self, x, ob):
        """-1: rintangan dominan KIRI -> putar KANAN; +1: dominan KANAN -> putar KIRI."""
        if ob is None or len(ob) == 0:
            return -1.0
        rel = ob - x[:2]
        d   = np.hypot(rel[:, 0], rel[:, 1])
        ang = np.arctan2(rel[:, 1], rel[:, 0]) - x[2]
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        near  = d < self.side_dist
        left  = int(np.sum(near & (ang > 0.1)))    
        right = int(np.sum(near & (ang < -0.1)))   
        if left > right:  return -1.0              
        if right > left:  return +1.0              
        return -1.0

    def reset(self):
        self.state = self.IDLE
        self._ref_xy = None
        self._ref_yaw = None

    def update(self, x, ob, cmd_v, now):
        """Return (v, omega, active). active=False -> pakai output DWA normal."""
        actual_v = abs(float(x[3]))

        if self.state == self.IDLE:
            commanding_fwd = cmd_v > 0.1
            if actual_v >= self.stuck_v_thr or not commanding_fwd or self._moving_since is None:
                self._moving_since = now
            if now - self._moving_since >= self.stuck_time:
                self._turn_dir = self._dominant_side(x, ob)
                if _USE_SIDE_BIAS:
                    self._turn_dir = _SAFE_TURN_DIR   
                side = "KANAN" if self._turn_dir < 0 else "KIRI"
                if self._rear_blocked(x, ob):
                    self.state   = self.ROTATE
                    self._ref_yaw = float(x[2])
                    log.info("[STUCK] Mandek; belakang TERHALANG -> langsung ROTATE %s" % side)
                else:
                    self.state   = self.REVERSE
                    self._ref_xy = (float(x[0]), float(x[1]))
                    log.info("[STUCK] Mandek -> MUNDUR ~%.1f m (belakang bebas), lalu ROTATE %s" % (self.reverse_dist, side))
                self._moving_since = now
            return (0.0, 0.0, False)

        if self.state == self.REVERSE:
            if self._rear_blocked(x, ob):
                self.state = self.ROTATE
                self._ref_yaw = float(x[2])
                log.info("[STUCK] Rintangan muncul di belakang -> stop mundur, ROTATE")
                return (self.rotate_v, self._turn_dir * self.rotate_rate, True)
            dx = x[0] - self._ref_xy[0]; dy = x[1] - self._ref_xy[1]
            if math.hypot(dx, dy) >= self.reverse_dist:
                self.state = self.ROTATE
                self._ref_yaw = float(x[2])
                log.info("[STUCK] Selesai mundur -> ROTATE sambil maju")
                return (self.rotate_v, self._turn_dir * self.rotate_rate, True)
            return (self.reverse_v, 0.0, True)

        if self.state == self.ROTATE:
            dyaw = abs(math.atan2(math.sin(x[2] - self._ref_yaw), math.cos(x[2] - self._ref_yaw)))
            if dyaw >= self.rotate_angle:
                log.info("[STUCK] Rotasi selesai -> kembali ke DWA normal")
                self.reset()
                return (0.0, 0.0, False)
            return (self.rotate_v, self._turn_dir * self.rotate_rate, True)

        return (0.0, 0.0, False)


stuck_recovery = StuckRecovery()


SPEED_MIN = 0.3
SPEED_MAX = 1.5

def set_dwa_speed(value):
    """Set kecepatan maksimum DWA (m/s). Langsung dipakai dwa_config pada loop berikutnya."""
    try:
        v = max(SPEED_MIN, min(SPEED_MAX, float(value)))
    except (TypeError, ValueError):
        return None
    dwa_config.max_speed = v
    log.info(f"[SPEED] DWA max_speed -> {v:.2f} m/s")
    return v

def set_pid_speed(value):
    """Set base/cruise speed PID global (m/s) untuk mode 1 & 2."""
    try:
        v = max(SPEED_MIN, min(SPEED_MAX, float(value)))
    except (TypeError, ValueError):
        return None
    arbiter.cruise_speed = v
    log.info(f"[SPEED] PID cruise_speed -> {v:.2f} m/s")
    return v


# =============================================================================
#  LIDAR PARSER
# =============================================================================

LIDAR_ANGLE_SIGN   = -1.0
LIDAR_ANGLE_OFFSET = 110.0

LIDAR_IGNORE_REAR_SECTOR = True
LIDAR_REAR_IGNORE_START_DEG = 135
LIDAR_REAR_IGNORE_END_DEG   = 225


def lidar_angle_in_sector(deg: int, start_deg: int, end_deg: int) -> bool:
    """True bila deg berada dalam sektor sudut inklusif; mendukung wrap 360."""
    deg = int(deg) % 360
    start_deg = int(start_deg) % 360
    end_deg = int(end_deg) % 360
    if start_deg <= end_deg:
        return start_deg <= deg <= end_deg
    return deg >= start_deg or deg <= end_deg


class LidarParser:
    def __init__(self):
        self.buf         = bytearray()
        self.scan_buf    = []
        self.last_angle  = -1.0
        self.scan_data   = [6.5] * 360
        self.frame_count = 0
        self.hz          = 0.0
        self._last_hz_t  = time.time()

        self._scan_seq         = 0  
        self._persist_seq_seen = -1   
        self._persist_window   = 3    
        self._persist_min_hits = 2  
        self._persist_tol_deg  = 3   
        self._occ_history      = deque(maxlen=self._persist_window)

    def feed(self, raw: bytes) -> bool:
        self.buf.extend(raw)
        new_scan = False
        while len(self.buf) >= 6:
            if self.buf[0] == 0xAA and self.buf[1] == 0x55:
                angle_raw, dist_raw = struct.unpack_from("<HH", self.buf, 2)
                angle = angle_raw / 100.0
                dist  = dist_raw  / 1000.0
                if 0.2 < dist < 6.5:
                    self.scan_buf.append((angle, dist))
                if self.last_angle > 0 and angle < self.last_angle - 300:
                    self._commit_scan()
                    new_scan = True
                self.last_angle = angle
                self.buf = self.buf[6:]
            else:
                self.buf = self.buf[1:]
        return new_scan

    def _commit_scan(self):
        data = [6.5] * 360
        for angle, dist in self.scan_buf:
            idx = int((LIDAR_ANGLE_SIGN * angle - LIDAR_ANGLE_OFFSET) % 360)
            if 0 <= idx < 360:
                data[idx] = min(data[idx], dist)
        self.scan_data = data
        self.scan_buf  = []
        self.frame_count += 1
        self._scan_seq += 1   
        now = time.time()
        if now - self._last_hz_t >= 1.0:
            self.hz = self.frame_count
            self.frame_count = 0
            self._last_hz_t  = now


    def filter_scan(self, min_dist=0.30, max_dist=6.0,
                    neighbor_thresh=0.45, min_neighbors=2) -> list:
        temp = []   # (deg, x, y)
        for i, dist in enumerate(self.scan_data):
            if not (min_dist < dist < max_dist):
                continue
            if LIDAR_IGNORE_REAR_SECTOR and lidar_angle_in_sector(
                i, LIDAR_REAR_IGNORE_START_DEG, LIDAR_REAR_IGNORE_END_DEG
            ):
                continue
            a = math.radians(i)
            temp.append((i, dist * math.cos(a), dist * math.sin(a)))

        spatial = []
        for k, (deg, x1, y1) in enumerate(temp):
            neighbors = sum(
                1 for m, (dg2, x2, y2) in enumerate(temp)
                if k != m and math.hypot(x1 - x2, y1 - y2) < neighbor_thresh
            )
            if neighbors >= min_neighbors:
                spatial.append((deg, x1, y1))

        if self._scan_seq != self._persist_seq_seen:
            self._persist_seq_seen = self._scan_seq
            self._push_occupancy({deg for (deg, _, _) in spatial})

        return [(x, y) for (deg, x, y) in spatial if self._confirmed(deg)]

    def _push_occupancy(self, occ_set: set) -> None:
        tol = self._persist_tol_deg
        expanded = set()
        for d in occ_set:
            for k in range(-tol, tol + 1):
                expanded.add((d + k) % 360)
        self._occ_history.append(expanded)

    def _confirmed(self, deg: int) -> bool:
        hits = sum(1 for occ in self._occ_history if deg in occ)
        return hits >= self._persist_min_hits

    def to_obstacles_global(self, robot_x: np.ndarray) -> np.ndarray:
        rel = self.filter_scan()
        if not rel:
            return np.empty((0, 2))
        c, s = math.cos(robot_x[2]), math.sin(robot_x[2])
        out = [[robot_x[0] + rx * c - ry * s,
                robot_x[1] + rx * s + ry * c] for (rx, ry) in rel]
        return np.array(out)


lidar_parser = LidarParser()


# =============================================================================
#  ROBOT STATE
# =============================================================================
class RobotState:
    def __init__(self):
        self.x            = np.array([0.0, 0.0, math.pi / 2, 0.0, 0.0])
        self.goal         = np.array([0.0, 20.0])
        self.waypoints    = []
        self.wp_index     = 0
        self.goal_reached = False

        self.nav_mode        = 0
        self.auto_mode       = True
        self.delivery_status = NavMode.STANDBY

        # Referensi GPS asal
        self.lat0_ref = 0.0
        self.lon0_ref = 0.0

        # Telemetri mentah dari Mega
        self.telemetry   = {}

        # LiDAR
        self.lidar_data  = [6.5] * 360
        self.lidar_hz    = 0.0

        # Timestamp
        self.ts_mega     = 0.0
        self.ts_lidar    = 0.0

        # DWA output
        self.dwa_v       = 0.0
        self.dwa_w       = 0.0

        # GPS status dari Mega
        self.gps_lost    = False
        self.gps_ready   = False
        self.mega_ready  = False  
        self.mqtt_connected = False 

        self._lock = threading.Lock()

    # ==================================================================
    #  Update dari CSV Mega
    # ==================================================================
    def update_from_mega_csv(self, fields: dict):
        with self._lock:
            if fields.get("mode") is not None:
                csv_mode = int(fields["mode"])
                if 0 <= csv_mode <= 3:
                    self.nav_mode = csv_mode

            if self.nav_mode == NavMode.ODO_DWA:
                if fields.get("odom_x") is not None: self.x[0] = fields["odom_x"]
                if fields.get("odom_y") is not None: self.x[1] = fields["odom_y"]
                if fields.get("theta_gyro_deg") is not None:
                    self.x[2] = math.radians(fields["theta_gyro_deg"] % 360.0)
            else:
                if fields.get("x_ekf") is not None: self.x[0] = fields["x_ekf"]
                if fields.get("y_ekf") is not None: self.x[1] = fields["y_ekf"]
                if fields.get("filteredHeading") is not None:
                    self.x[2] = math.radians((90.0 - fields["filteredHeading"]) % 360.0)

            if fields.get("speed") is not None: self.x[3] = fields["speed"]
            if fields.get("omega_act") is not None:
                self.x[4] = fields["omega_act"]

            if self.nav_mode != NavMode.ODO_DWA:
                gx = fields.get("goal_x"); gy = fields.get("goal_y")
                if gx is not None and gy is not None and not (gx == 0.0 and gy == 0.0):
                    self.goal[0] = gx
                    self.goal[1] = gy
            if fields.get("target_lat") is not None:
                fields["tlat"] = fields["target_lat"]
            if fields.get("target_lon") is not None:
                fields["tlon"] = fields["target_lon"]
            self.telemetry.update(fields)
            self.ts_mega = time.time()


    def update_battery(self, battery: float, current: float, load: float):
        with self._lock:
            self.telemetry["battery"] = battery
            self.telemetry["current"] = current
            self.telemetry["load"]    = load


    def update_system(self, fields: dict):
        with self._lock:
            self.telemetry.update(fields)
            if fields.get("target_lat") is not None:
                self.telemetry["tlat"] = fields["target_lat"]
            if fields.get("target_lon") is not None:
                self.telemetry["tlon"] = fields["target_lon"]
            if fields.get("travelled") is not None:
                self.telemetry["distanceTravelled"] = fields["travelled"]


    def dead_reckon(self, dt: float):
        if dt <= 0 or dt > 1.0:
            return
        with self._lock:
            v     = float(self.x[3])
            yaw   = float(self.x[2])
            omega = float(self.x[4])
            self.x[0] += v * math.cos(yaw) * dt
            self.x[1] += v * math.sin(yaw) * dt
            self.x[2] = (yaw + omega * dt) % (2.0 * math.pi)

    def get_x_copy(self):
        with self._lock:
            return self.x.copy(), self.goal.copy()

    def update_lidar(self, scan_data: list, hz: float):
        with self._lock:
            self.lidar_data = scan_data
            self.lidar_hz   = hz
            self.ts_lidar   = time.time()

    def set_waypoints(self, wps: list, is_odo=False):
        with self._lock:
            self.waypoints    = wps
            self.wp_index     = 0
            self.goal_reached = False
            if is_odo and wps:
                self.goal[0] = float(wps[0].get("x", 0))
                self.goal[1] = float(wps[0].get("y", 0))

            if wps and self.delivery_status == NavMode.STANDBY:
                self.delivery_status = NavMode.DELIVERY

    def clear_waypoints(self):
        with self._lock:
            self.waypoints       = []
            self.wp_index        = 0
            self.goal_reached    = False
            self.delivery_status = NavMode.STANDBY

    def advance_waypoint(self, reached_idx: int):
        with self._lock:
            self.wp_index = reached_idx + 1

    def mark_mission_done(self):
        with self._lock:
            self.goal_reached    = True
            self.delivery_status = NavMode.STANDBY

    def resume_for_new_route(self, count: int = 0):
        with self._lock:
            self.wp_index     = 0
            self.goal_reached = False
            if count > 0 and self.delivery_status == NavMode.STANDBY:
                self.delivery_status = NavMode.DELIVERY

    def reset_position(self):
        with self._lock:
            self.x[0:2] = 0.0
            self.lat0_ref = 0.0
            self.lon0_ref = 0.0

    def set_gps_ref(self, lat0: float, lon0: float):
        with self._lock:
            if self.lat0_ref == 0.0 and lat0 != 0.0:
                self.lat0_ref = lat0
            if self.lon0_ref == 0.0 and lon0 != 0.0:
                self.lon0_ref = lon0

    def get_lidar_snapshot(self):
        with self._lock:
            return self.lidar_data.copy(), self.lidar_hz

    def get_nav_state(self):
        with self._lock:
            return (
                self.nav_mode,
                self.delivery_status,
                self.wp_index,
                len(self.waypoints),
                self.goal_reached,
            )

robot_state = RobotState()

# =============================================================================
#  CSV LOGGER
# =============================================================================
class CSVLogger:
    def __init__(self, filename="argo_dwa_log.csv", enabled=True):
        self._file   = None
        self.enabled = enabled
        if self.enabled:
            try:
                self._file = open(filename, "w", buffering=1 << 16)  
                self._file.write(
                    "time,x,y,yaw,v_cmd,omega_cmd,v_actual,omega_actual,"
                    "goal_x,goal_y,dist_goal,lat,lon,nav_mode\n"
                )
                log.info(f"[CSV] Log aktif → {filename}")
            except Exception as e:
                log.warning(f"[CSV] Gagal buka file log: {e}")
                self._file = None

    def write(self, x, goal, v_cmd, w_cmd, nav_mode):
        if not self._file:
            return
        try:
            t    = time.time()
            dist = math.hypot(goal[0] - x[0], goal[1] - x[1])
            lat  = robot_state.telemetry.get("lat", "")
            lon  = robot_state.telemetry.get("lon", "")
            self._file.write(
                f"{t:.3f},{x[0]:.4f},{x[1]:.4f},{x[2]:.4f},"
                f"{v_cmd:.4f},{w_cmd:.4f},{x[3]:.4f},{x[4]:.4f},"
                f"{goal[0]:.4f},{goal[1]:.4f},{dist:.4f},"
                f"{lat},{lon},{nav_mode}\n"
            )
            self._nwrite = getattr(self, "_nwrite", 0) + 1
            if self._nwrite % 100 == 0:
                self._file.flush()   
        except Exception as e:
            log.warning(f"[CSV] Write error: {e}")

    def close(self):
        if self._file:
            try:
                self._file.close()
                log.info("[CSV] Log disimpan.")
            except Exception:
                pass
            self._file = None


csv_logger = CSVLogger(enabled=CONFIG.get("log_csv", True))

CSV_RECORD_HEADER = [
    "Time", "Lat_GPS", "Lon_GPS", "Lat_0", "Lon_0", "Target_Lat", "Target_Lon",
    "X_GPS", "Y_GPS", "X_EKF", "Y_EKF", "X_Pred", "Y_Pred", "X_ODO", "Y_ODO",
    "Theta", "Theta_Target", "Theta_Err", "VL", "VR", "DWA_V", "DWA_Omega",
    "V_Aktual", "Omega_Aktual", "PWM_L", "PWM_R", "Dist_GPS", "Dist_EKF",
    "Dist_Travelled", "WP", "ETA_kf", "ETA_raw", "X_obs", "Y_obs", "Bump_Sta", "Haz_Sta",
    "Pothole_Sta", "Frame", "FPS", "Latency", "Arbiter",
]


def _csvfmt(v):
    """Format satu sel CSV. None/NaN/Inf -> kosong; float -> 5 desimal."""
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.5f}"
    s = str(v)
    if ("," in s) or ("\n" in s):
        s = s.replace(",", ";").replace("\n", " ")
    return s


def build_record_row(t, x, goal, v_cmd, w_cmd, ob, traj, avoid, nav_mode, telem, latency_ms):
    rs = robot_state.telemetry

    x_ekf = telem.get("x_ekf"); y_ekf = telem.get("y_ekf")
    gps_x = telem.get("gps_x"); gps_y = telem.get("gps_y")

    def _dist(px, py):
        if px is None or py is None:
            return None
        return math.hypot(goal[0] - float(px), goal[1] - float(py))

    dist_ekf = _dist(x_ekf, y_ekf)
    if dist_ekf is None:
        dist_ekf = math.hypot(goal[0] - float(x[0]), goal[1] - float(x[1]))
    dist_gps = _dist(gps_x, gps_y)
    if dist_gps is None:
        dist_gps = telem.get("distance")

    if traj is not None and len(traj) > 0:
        x_pred = float(traj[-1][0]); y_pred = float(traj[-1][1])
    else:
        x_pred = y_pred = None

    x_obs = y_obs = None
    if ob is not None and len(ob) > 0:
        try:
            _ob = np.asarray(ob, dtype=float)
            d = np.hypot(_ob[:, 0] - float(x[0]), _ob[:, 1] - float(x[1]))
            j = int(np.argmin(d))
            x_obs = float(_ob[j, 0]); y_obs = float(_ob[j, 1])
        except Exception:
            pass

    # Status kamera (bump / hazard / pothole / frame / fps).
    cam_bump = cam_pot = None
    cam_hz = cam_frame = None
    cam_fps = None
    if cam is not None:
        try:
            st = cam.get_status()
            cam_bump  = st.get("bump")
            cam_pot   = st.get("pothole")
            cam_hz    = st.get("hazard")
            cam_frame = st.get("frame")
            cam_fps   = st.get("fps")
        except Exception:
            pass

    return [
        round(float(t), 3),                              # Time (epoch detik)
        telem.get("lat"), telem.get("lon"),              # Lat_GPS, Lon_GPS
        robot_state.lat0_ref, robot_state.lon0_ref,      # Lat_0, Lon_0
        telem.get("tlat"), telem.get("tlon"),            # Target_Lat, Target_Lon
        gps_x, gps_y,                                    # X_GPS, Y_GPS
        x_ekf, y_ekf,                                    # X_EKF, Y_EKF
        x_pred, y_pred,                                  # X_Pred, Y_Pred
        float(x[0]), float(x[1]),                        # X_ODO, Y_ODO
        math.degrees(float(x[2])),                       # Theta (deg)
        telem.get("target_heading"), telem.get("heading_err"),  # Theta_Target, Theta_Err (deg, dari Mega)
        telem.get("vL"), telem.get("vR"),                # VL, VR (m/s aktual)
        float(v_cmd), float(w_cmd),                      # DWA_V, DWA_Omega (perintah)
        telem.get("speed"), telem.get("omega"),          # V_Aktual, Omega_Aktual
        telem.get("pwm_l"), telem.get("pwm_r"),          # PWM_L, PWM_R
        dist_gps, dist_ekf,                              # Dist_GPS, Dist_EKF
        telem.get("travelled"),                          # Dist_Travelled
        rs.get("wp", robot_state.wp_index),              # WP
        telem.get("eta_kf"), telem.get("eta_raw"),       # ETA_kf, ETA_raw
        x_obs, y_obs,                                    # X_obs, Y_obs
        cam_bump, cam_hz, cam_pot,                       # Bump_Sta, Haz_Sta, Pothole_Sta
        cam_frame, cam_fps,                              # Frame, FPS
        (round(float(latency_ms), 2) if latency_ms is not None else None),  # Latency (ms)
        ("AVOID" if avoid else "CRUISE"),                # Arbiter
    ]


class ReplayLogger:
    def __init__(self, prefix="argo_record"):
        self._file = None
        self._queue = None
        self._thread = None
        self.prefix = prefix
        self.recording = False
        self.frame_count = 0
        self.filename = None

    def _writer_loop(self):
        while True:
            line = self._queue.get()
            if line is None:
                break
            try:
                if self._file:
                    self._file.write(line)
            except Exception as e:
                log.warning(f"[REC] Write error: {e}")
        try:
            if self._file:
                self._file.flush()
        except Exception:
            pass

    def start(self, filename=None):
        self.stop()
        if filename is None:
            filename = f"{self.prefix}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            rec_dir  = os.path.join(base_dir, "records")
            os.makedirs(rec_dir, exist_ok=True)
            path = os.path.join(rec_dir, os.path.basename(filename))
            self._file = open(path, "w", buffering=1 << 16, newline="")
            self.filename = path
            self.frame_count = 0
            self._file.write(",".join(CSV_RECORD_HEADER) + "\n")
            self._file.flush()
            self._queue = queue.Queue(maxsize=4000)
            self._thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._thread.start()
            self.recording = True
            log.info(f"[REC] MULAI rekam CSV -> {self.filename}")
            return True
        except Exception as e:
            log.warning(f"[REC] Gagal mulai rekam: {e}")
            self._file = None
            self.recording = False
            return False

    def write(self, t, x, goal, v_cmd, w_cmd, ob, traj, min_front, avoid, nav_mode, telem=None, latency_ms=None):
        if not self.recording or self._queue is None:
            return
        try:
            row = build_record_row(t, x, goal, v_cmd, w_cmd, ob, traj,
                                   avoid, nav_mode, telem or {}, latency_ms)
            line = ",".join(_csvfmt(v) for v in row) + "\n"
        except Exception as e:
            log.warning(f"[REC] Format error: {e}")
            return
        try:
            self._queue.put_nowait(line)
            self.frame_count += 1
        except queue.Full:
            pass  

    def stop(self):
        self.recording = False
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass
        if self._thread is not None:
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                pass
            self._thread = None
        if self._file:
            try:
                self._file.close()
                log.info(f"[REC] STOP rekam. {self.frame_count} baris -> {self.filename}")
            except Exception:
                pass
            self._file = None
        self._queue = None

    def close(self):
        self.stop()


replay_logger = ReplayLogger()


# =============================================================================
#  SERIAL MEGA
# =============================================================================
class MegaSerial:

    CSV_COLS = [
        "ms", "gps_x", "gps_y", "x_pred", "y_pred",
        "x_ekf", "y_ekf", "filteredHeading", "speed", "omega_act",
        "vL_act", "vR_act", "headingError", "targetHeading",
        "distance", "dist_ekf", "distanceTravelled",
        "mode", "dwa_v", "dwa_omega",
        "goal_x", "goal_y", "wp", "spd_min", "spd_max",
        "odom_x", "odom_y", "theta_ekf_deg", "eta_raw", "eta_ekf", "theta_gyro_deg",
    ]

    HEADER_ALIAS = {
        "time": "ms", "t": "ms",
        "theta": "filteredHeading", "heading": "filteredHeading",
        "omega": "omega_act",
        "vL": "vL_act", "vR": "vR_act", "vl": "vL_act", "vr": "vR_act",
        "heading_error": "headingError", "heading_err": "headingError",
        "target_heading": "targetHeading",
        "dist_gps": "distance", "dist_raw": "distance",
        "distance_travelled": "distanceTravelled", "travelled": "distanceTravelled",
        "waypoint": "wp",
    }

    def __init__(self):
        self.ser: Optional[serial.Serial] = None
        self._buf      = ""
        self.connected = False
        self._csv_header: Optional[list] = None
        self._lock     = threading.Lock()
        self._reconnect_interval = 5.0
        self._last_reconnect     = 0.0

    def connect(self):
        port = CONFIG["mega_port"]
        baud = CONFIG["mega_baud"]
        try:
            self.ser = serial.Serial(port, baud, timeout=0.05)
            time.sleep(2)
            self.connected = True
            log.info(f"[Mega] Terhubung @ {port} {baud}bps")
        except Exception as e:
            log.error(f"[Mega] Gagal buka port [{port}]: {e}")
            self.connected = False

    def try_reconnect(self):
        now = time.time()
        if now - self._last_reconnect < self._reconnect_interval:
            return False
        self._last_reconnect = now
        port = CONFIG["mega_port"]
        baud = CONFIG["mega_baud"]
        try:
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.ser = serial.Serial(port, baud, timeout=0.05)
            time.sleep(0.5)
            self.connected = True
            self._buf = ""
            log.info(f"[Mega] Reconnected @ {port}")
            return True
        except Exception as e:
            log.debug(f"[Mega] Reconnect gagal: {e}")
            self.connected = False
            return False

    def _write_raw(self, txt: str):
        if self.ser and self.connected:
            try:
                self.ser.write(txt.encode())
            except Exception as e:
                log.warning(f"[Mega TX] Error: {e}")
                self.connected = False

    def send_velocity(self, v: float, omega: float):
        with self._lock:
            self._write_raw(f"{v:.4f},{omega:.4f}\n")

    def send_command(self, cmd: str):
        with self._lock:
            if not cmd.endswith("\n"):
                cmd += "\n"
            self._write_raw(cmd)
        log.info(f"[Mega →] {cmd.strip()}")

    def send_nav_mode(self, mode_int: int):
        if mode_int not in (0, 1, 2, 3):
            log.warning(f"[Mega] Mode tidak valid: {mode_int}")
            return
        self.send_command(f"<MODE,{mode_int}>")

    def send_camera_heading(self, deg: float):
        with self._lock:
            self._write_raw(f"<CAMHDG,{deg:.2f}>\n")

    def send_camera_lost(self):
        with self._lock:
            self._write_raw("<CAMLOST>\n")

    def send_bump(self, active: bool):
        with self._lock:
            self._write_raw(f"<BUMP,{1 if active else 0}>\n")

    def send_status(self, status: str):
        self.send_command(f"<STATUS,{status}>")

    def send_waypoints_gps(self, wps: list):
        valid = [w for w in wps if w.get("lat") is not None and w.get("lon") is not None]
        if not valid:
            return
        self.send_command(f"<TOTALWP,{len(valid)}>")
        time.sleep(0.05)
        for idx, w in enumerate(valid):
            self.send_command(f"<WP,{float(w['lat']):.6f},{float(w['lon']):.6f},{idx}>")
            time.sleep(0.02)
        log.info(f"[Mega] {len(valid)} waypoint GPS dikirim")

    def send_waypoints_odo(self, wps: list):
        valid = [w for w in wps if w.get("x") is not None and w.get("y") is not None]
        if not valid:
            return
        coords = ",".join(f"{float(w['x']):.3f},{float(w['y']):.3f}" for w in valid)
        self.send_command(f"<WXY:{len(valid)},{coords}>")
        log.info(f"[Mega] {len(valid)} waypoint Odo dikirim")

    def read_lines(self) -> list:
        lines = []
        if not self.ser or not self.connected:
            return lines
        try:
            n = self.ser.in_waiting
            if n:
                raw = self.ser.read(n)
                self._buf += raw.decode(errors="ignore")
        except Exception as e:
            log.warning(f"[Mega RX] Error: {e}")
            self.connected = False
            return lines
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                lines.append(line)
        # [FIX-7] Guard buffer overflow
        if len(self._buf) > 4096:
            self._buf = self._buf[-1024:]
        return lines

    CORE_COLS = 17

    def _capture_header(self, line: str):
        cols = [c.strip() for c in line.split(",") if c.strip() != ""]
        if cols:
            cols = [self.HEADER_ALIAS.get(c, c) for c in cols]
            self._csv_header = cols
            log.info(f"[Mega] Header CSV terdeteksi: {len(cols)} kolom")

    def parse_csv_line(self, line: str) -> Optional[dict]:
        if line.startswith("t,") or line.startswith("ms,") or line.startswith("time,"):
            self._capture_header(line)
            return None
        if line.startswith("<") or line.startswith("$") or line.startswith("BATT,") or line.startswith("SYS,") \
           or ":" in line[:12]:
            return None

        parts = line.split(",")
        if len(parts) < self.CORE_COLS:
            return None

        keys = self._csv_header if self._csv_header else self.CSV_COLS
        result = {}
        for i, col in enumerate(keys):
            if i >= len(parts):
                break
            tok = parts[i].strip()
            try:
                result[col] = float(tok)
            except ValueError:
                result[col] = None

        if result.get("x_ekf") is None or result.get("y_ekf") is None:
            return None
        return result

    def stop(self):
        self.send_velocity(0.0, 0.0)


mega = MegaSerial()


# =============================================================================
#  SERIAL LIDAR (Delta2G via USB)
# =============================================================================
class LidarSerial:
    def __init__(self):
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self._last_reconnect = 0.0

    def connect(self):
        if not CONFIG["use_lidar"]:
            log.info("[LiDAR] Dinonaktifkan")
            return
        port = CONFIG["lidar_port"]
        baud = CONFIG["lidar_baud"]
        try:
            self.ser = serial.Serial(port, baud, timeout=0)
            time.sleep(1)
            self.connected = True
            log.info(f"[LiDAR] Terhubung @ {port} {baud}bps")
        except Exception as e:
            log.warning(f"[LiDAR] Gagal [{port}]: {e}")
            self.connected = False

    def try_reconnect(self):
        if not CONFIG["use_lidar"]:
            return False
        now = time.time()
        if now - self._last_reconnect < 5.0:
            return False
        self._last_reconnect = now
        port = CONFIG["lidar_port"]
        baud = CONFIG["lidar_baud"]
        try:
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.ser = serial.Serial(port, baud, timeout=0)
            time.sleep(0.5)
            self.connected = True
            log.info(f"[LiDAR] Reconnected @ {port}")
            return True
        except Exception:
            self.connected = False
            return False

    def read_raw(self, size=2048) -> bytes:
        if not self.ser or not self.connected:
            return b""
        try:
            n = self.ser.in_waiting
            if n:
                return self.ser.read(min(n, size))
        except Exception as e:
            log.warning(f"[LiDAR RX] Error: {e}")
            self.connected = False
        return b""


lidar_serial = LidarSerial()


# =============================================================================
#  WEBSOCKET CLIENTS
# =============================================================================
ws_clients: Set = set()

_bcast_queue = None
_BCAST_QUEUE_MAX = 8


def _get_bcast_queue():
    global _bcast_queue
    if _bcast_queue is None:
        _bcast_queue = asyncio.Queue(maxsize=_BCAST_QUEUE_MAX)
    return _bcast_queue


async def broadcast(msg: dict):
    """[WS-FIX] Non-blocking enqueue. Tidak pernah memblok pemanggil (dwa_loop)."""
    if not ws_clients:
        return
    q = _get_bcast_queue()
    try:
        q.put_nowait(msg)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
            q.put_nowait(msg)
        except Exception:
            pass


async def broadcaster_loop():
    q = _get_bcast_queue()
    while True:
        msg = await q.get()
        if not ws_clients:
            continue
        data = json.dumps(msg, default=str)
        dead = set()
        for ws in ws_clients.copy():
            try:
                await asyncio.wait_for(ws.send(data), timeout=0.25)
            except Exception:
                dead.add(ws)
        if dead:
            ws_clients.difference_update(dead)


# =============================================================================
#  BUILD TELEMETRY MSG
# =============================================================================
def build_telemetry_msg(x, goal, v_cmd, w_cmd, ob, predicted_traj=None) -> dict:
    t  = robot_state.telemetry
    nm = robot_state.nav_mode

    L    = 0.52
    spd_actual = float(t.get("speed") or x[3])
    omg_actual = float(t["omega_act"]) if t.get("omega_act") is not None else float(x[4])
    if t.get("vL_act") is not None and t.get("vR_act") is not None:
        vL = float(t["vL_act"])
        vR = float(t["vR_act"])
    else:
        vL = spd_actual - omg_actual * L / 2
        vR = spd_actual + omg_actual * L / 2

    heading_err = t.get("headingError")

    V_MAX_PWM = 1.4
    cmd_vL = float(v_cmd) - float(w_cmd) * L / 2
    cmd_vR = float(v_cmd) + float(w_cmd) * L / 2
    _mx = max(abs(cmd_vL), abs(cmd_vR))
    if _mx > V_MAX_PWM:
        cmd_vL = cmd_vL / _mx * V_MAX_PWM
        cmd_vR = cmd_vR / _mx * V_MAX_PWM
    pwm_l = int(max(-255, min(255, (cmd_vL / V_MAX_PWM) * 255)))
    pwm_r = int(max(-255, min(255, (cmd_vR / V_MAX_PWM) * 255)))

    if esp32_pwm["l"] is not None and (time.time() - esp32_pwm["ts"]) < 2.0:
        pwm_l = int(esp32_pwm["l"])
    if esp32_pwm["r"] is not None and (time.time() - esp32_pwm["ts"]) < 2.0:
        pwm_r = int(esp32_pwm["r"])

    v_err = float(v_cmd) - spd_actual
    w_err = float(w_cmd) - omg_actual

    msg = {
        "type": "telemetry",

        "lat_raw":    t.get("lat"),
        "lon_raw":    t.get("lon"),
        "lat":        t.get("lat"),
        "lon":        t.get("lon"),

        "heading":        t.get("filteredHeading"),
        "heading_err":    heading_err,
        "target_heading": t.get("targetHeading"),
        "mag_cal":        int(t["mag_cal"])  if t.get("mag_cal")  is not None else None,
        "sys_cal":        int(t["sys_cal"])  if t.get("sys_cal")  is not None else None,

        "speed":    round(spd_actual, 4),
        "omega":    round(omg_actual, 4),
        "vL":       round(vL,  4),
        "vR":       round(vR,  4),
        "dwa_v":    round(float(v_cmd), 4),
        "dwa_w":    round(float(w_cmd), 4),

        "pwm_l":     pwm_l,
        "pwm_r":     pwm_r,
        "v_err":     round(v_err, 4),
        "omega_err": round(w_err, 4),
        "dwa_speed": round(dwa_config.max_speed, 3),
        "pid_speed": round(arbiter.cruise_speed, 3),

        "x_odo":    round(float(x[0]), 4),
        "y_odo":    round(float(x[1]), 4),

        "x_ekf":    t.get("x_ekf"),
        "y_ekf":    t.get("y_ekf"),
        "gps_x":    t.get("gps_x"),
        "gps_y":    t.get("gps_y"),
        "Q":        t.get("Q"),
        "R":        t.get("R"),

        "distance":  t.get("distance"),
        "travelled": t.get("distanceTravelled"),
        "err":       t.get("headingError"),

        "tlat":       t.get("tlat") or t.get("target_lat"),
        "tlon":       t.get("tlon") or t.get("target_lon"),
        "target_lat": t.get("target_lat"),
        "target_lon": t.get("target_lon"),

        "eta":     t.get("ETA_kf"),
        "eta_kf":  t.get("ETA_kf"),
        "eta_raw": t.get("ETA_raw"),

        "wp":       t.get("wp", robot_state.wp_index),
        "wp_idx":   t.get("wp", robot_state.wp_index),
        "wp_total": len(robot_state.waypoints),

        "mode":            nm,
        "Cmd":             robot_state.delivery_status,
        "nav_mode":        nm,
        "nav_mode_str":    NavMode.STR_MAP.get(nm, "UNKNOWN"),
        "delivery_status": robot_state.delivery_status,

        "battery": t.get("battery"),
        "current": t.get("current"),
        "load":    t.get("load"),

        "gps_lost":  robot_state.gps_lost,
        "gps_ready": not robot_state.gps_lost,

        "mega_dwa_v":     t.get("dwa_v"),
        "mega_dwa_omega": t.get("dwa_omega"),

        "goal_x":   round(float(goal[0]), 4),
        "goal_y":   round(float(goal[1]), 4),
        "goal_reached": robot_state.goal_reached,

        "mega_connected":  mega.connected,
        "lidar_connected": lidar_serial.connected,

        "predicted_traj": (
            [[round(float(p[0]), 3), round(float(p[1]), 3)]
             for p in predicted_traj[::3]]
            if predicted_traj is not None and len(predicted_traj) > 0
            else []
        ),
        "obstacles": (
            [[round(float(o[0]), 3), round(float(o[1]), 3)] for o in ob]
            if ob is not None and len(ob) > 0
            else []
        ),
        "ob_count": int(len(ob)) if ob is not None else 0,
        "ts": round(time.time() * 1000),
    }
    return msg


# =============================================================================
#  THREAD: BACA LIDAR
# =============================================================================
def lidar_thread():
    while True:
        if not lidar_serial.connected:
            lidar_serial.try_reconnect()
            time.sleep(1)
            continue
        raw = lidar_serial.read_raw()
        if raw:
            new_scan = lidar_parser.feed(raw)
            if new_scan:
                robot_state.update_lidar(
                    lidar_parser.scan_data.copy(), lidar_parser.hz
                )
        time.sleep(0.005)


# =============================================================================
#  GOAL TRACKING — Sinkronisasi goal ke waypoint aktif (GPS)
# =============================================================================
def sync_goal_to_waypoint():

    with robot_state._lock:
        wps = robot_state.waypoints
        idx = robot_state.wp_index
        if not wps or idx >= len(wps):
            return
        w = wps[idx]
        has_gps = bool(w.get("lat")) and bool(w.get("lon"))
        if (not has_gps) and w.get("x") is not None:
            if robot_state.goal[0] == 0.0 and robot_state.goal[1] == 0.0:
                robot_state.goal[0] = float(w["x"])
                robot_state.goal[1] = float(w["y"])


# =============================================================================
#  PARSE SATU BARIS DARI MEGA
# =============================================================================
async def _process_mega_line(line: str):
    global _dwa_started
    if line.startswith("<DSTAT,"):
        try:
            status = line[7:line.index(">")].strip().upper()
            if status in ("STANDBY", "PICKUP", "DELIVERY"):
                robot_state.delivery_status = status
                _dwa_started = (status != "STANDBY")
                log.info(f"[Mega] DSTAT: {status} (dwa_started={_dwa_started})")
                await broadcast({
                    "type":            "delivery_status",
                    "status":          status,
                    "delivery_status": status,
                    "dwa_running":     _dwa_started,
                })
        except Exception:
            pass
        return

    if line == "ARGO:READY":
        robot_state.mega_ready = True
        log.info("[Mega] ARGO:READY")
        await broadcast({"type": "ready", "msg": "ARGO:READY"})
        return

    if line == "GPS:LOST":
        robot_state.gps_lost = True
        log.warning("[Mega] GPS:LOST")
        await broadcast({"type": "gps_event", "event": "lost", "gps_lost": True})
        return

    if line == "GPS:READY":
        robot_state.gps_lost = False
        log.info("[Mega] GPS:READY")
        await broadcast({"type": "gps_event", "event": "ready", "gps_lost": False})
        return

    if line.startswith("<MODEFB,"):
        try:
            mode_int = int(line[8:line.index(">")])
            mode_str = NavMode.STR_MAP.get(mode_int, "?")
            robot_state.nav_mode = mode_int
            if mode_int in (1, 2):
                _dwa_started = True
            elif mode_int in (0, 3):
                _dwa_started = False
            log.info(f"[Mega] MODEFB: {mode_int} ({mode_str})")
            await broadcast({
                "type":         "mode_feedback",
                "nav_mode":     mode_int,
                "nav_mode_str": mode_str,
            })
        except Exception:
            pass
        return

    if line.startswith("<PINOK,"):
        try:
            val = int(line[7:line.index(">")])
            ok  = (val == 1)
            log.info(f"[Mega] PIN check: {'BENAR' if ok else 'SALAH'}")
            await broadcast({"type": "pin_event", "event": "pin_check", "correct": ok})
        except Exception:
            pass
        return

    if line.startswith("WP:REACHED:"):
        try:
            idx = int(line.split(":")[-1])
            # [FIX-5] Tidak await di dalam lock — pakai method
            robot_state.advance_waypoint(idx)
            log.info(f"[Mega] WP:{idx} tercapai")
            await broadcast({"type": "wp_event", "event": "reached", "index": idx})
        except Exception:
            pass
        return

    if line == "WP:DONE":
        robot_state.mark_mission_done()
        mega.send_velocity(0.0, 0.0)
        log.info("[Mega] WP:DONE — semua waypoint selesai")
        await broadcast({"type": "wp_event", "event": "done"})
        return

    if line.startswith("ACK:W,"):
        n = line[6:]
        log.info(f"[Mega] ACK:W,{n} — waypoint diterima")
        await broadcast({"type": "log", "source": "mega", "msg": line})
        return

    if line.startswith("ROUTE:RECEIVED:"):
        try:
            count = int(line.split(":")[-1])
        except ValueError:
            count = 0

        if count > 0:
            robot_state.resume_for_new_route(count)
            arbiter.avoid          = False
            arbiter._clear_since   = None
            arbiter._last_front    = float("inf")
            arbiter._last_front_ts = 0.0
            log.info(f"[Mega] Rute diterima: {count} waypoint -> latch misi DILEPAS (resume)")
        else:
            log.info(f"[Mega] Rute diterima: {count} waypoint")
        await broadcast({
            "type": "wp_event",
            "event": "route_received",
            "count": count,
        })
        return

    if line.startswith("[TOTALWP]"):
        log.info(f"[Mega] {line}")
        await broadcast({"type": "log", "source": "mega", "msg": line})
        return

    if line.startswith("$"):
        await broadcast({"type": "nmea", "line": line})
        return

    if line.startswith("MQTT:"):
        if line == "MQTT:OK":
            robot_state.mqtt_connected = True
        elif line.startswith("MQTT:FAIL"):
            robot_state.mqtt_connected = False
        await broadcast({"type": "log", "source": "mqtt", "msg": line})
        return

    if line.startswith("<PWM,"):
        try:
            body = line[5:line.index(">")]
            pl, pr = body.split(",")[:2]
            esp32_pwm["l"]  = int(float(pl))
            esp32_pwm["r"]  = int(float(pr))
            esp32_pwm["ts"] = time.time()
        except Exception:
            pass
        return

    if line.startswith("<SPD,"):
        try:
            body = line[5:line.index(">")]
            which, val = body.split(",")[:2]
            which = which.strip().lower()
            if   which == "dwa": set_dwa_speed(val)
            elif which == "pid": set_pid_speed(val)
            log.info(f"[Mega->Speed] {which} = {val} (asal MQTT/ESP32)")
            await broadcast({
                "type":      "speed_info",
                "dwa_speed": round(dwa_config.max_speed, 3),
                "pid_speed": round(arbiter.cruise_speed, 3),
                "ts":        time.time(),
            })
        except Exception:
            pass
        return

    if "[GPS REF SET]" in line:
        log.info("[Mega] GPS Reference ditetapkan")
        await broadcast({"type": "gps_event", "event": "ref_set"})
        return

    if line.startswith("SYS,"):
        try:
            b = line.split(",")
            keys = ["lat", "lon", "gps_x", "gps_y", "target_lat", "target_lon",
                    "gps_count", "ms_since_gps", "Q", "R", "ETA_raw", "ETA_kf",
                    "mag_cal", "sys_cal", "travelled", "hdop", "sats"]
            sysf = {}
            for i, k in enumerate(keys):
                if i + 1 < len(b):
                    try:
                        sysf[k] = float(b[i + 1])
                    except ValueError:
                        pass
            if sysf:
                robot_state.update_system(sysf)
                lat = sysf.get("lat"); lon = sysf.get("lon")
                if lat and lat != 0:
                    robot_state.set_gps_ref(float(lat), 0.0)
                if lon and lon != 0:
                    robot_state.set_gps_ref(0.0, float(lon))
        except (IndexError, ValueError):
            log.debug(f"[Mega] SYS rusak, dilewati: {line}")
        return

    if line.startswith("BATT,"):
        try:
            b = line.split(",")
            volt = float(b[1]); curr = float(b[2]); load = float(b[3])
            robot_state.update_battery(volt, curr, load)
        except (IndexError, ValueError):
            log.debug(f"[Mega] BATT rusak, dilewati: {line}")
        return

    parsed = mega.parse_csv_line(line)
    if parsed:
        robot_state.update_from_mega_csv(parsed)
        # Update lat0/lon0 referensi jika belum ada
        if parsed.get("lat") and parsed["lat"] != 0:
            robot_state.set_gps_ref(float(parsed["lat"]), 0.0)
        if parsed.get("lon") and parsed["lon"] != 0:
            robot_state.set_gps_ref(0.0, float(parsed["lon"]))
        return

    if line and not line.startswith("t,") and not line.startswith("ms,"):
        log.debug(f"[Mega misc] {line}")
        await broadcast({"type": "log", "source": "mega", "msg": line})


# =============================================================================
#  LOOP UTAMA
# =============================================================================
_pos_history: deque = deque(maxlen=10)
_dwa_started = True


async def dwa_loop():
    global _dwa_started
    dt             = dwa_config.dt
    last_time      = time.time()
    last_broadcast = 0.0
    last_dr        = time.time()   
    last_keepline_cmd = 0.0       
    last_bump_cmd = False        
    last_bump_ts  = 0.0        
    last_preview  = 0.0         

    log.info("[DWA] Loop dimulai.")

    while True:
        now = time.time()

        if now - last_time < dt * 0.9:
            await asyncio.sleep(0.005)
            continue
        last_time = now

        if cam is not None and CONFIG["use_camera"] and CONFIG.get("show_preview") and (now - last_preview >= 0.066):
            last_preview = now
            try: cam.render_preview()
            except Exception: pass

        if CONFIG.get("keepline_start") and robot_state.nav_mode != NavMode.GPS_EKF_ONLY:
            if CONFIG["send_to_mega"] and (now - last_keepline_cmd > 1.0):
                last_keepline_cmd = now
                mega.send_nav_mode(NavMode.GPS_EKF_ONLY)
                log.info("[CAM] KEEPLINE: kirim <MODE,3> ke Mega (menunggu MODEFB)...")

        if not mega.connected:
            mega.try_reconnect()
            if now - last_broadcast >= 1.0:
                last_broadcast = now
                await broadcast({
                    "type":             "status",
                    "serial_connected": False,
                    "lidar_connected":  lidar_serial.connected,
                    "dwa_running":      False,
                    "ts":               int(now * 1000),
                })
            await asyncio.sleep(0.1)
            continue

        _ts_before = robot_state.ts_mega   
        lines = mega.read_lines()
        for line in lines:
            await _process_mega_line(line)

        if robot_state.ts_mega == _ts_before:
            robot_state.dead_reckon(now - last_dr)
        last_dr = now

        sync_goal_to_waypoint()

        nm = robot_state.nav_mode
        ds = robot_state.delivery_status

        if nm == 0 or ds == NavMode.STANDBY:
            if now - last_broadcast >= 0.2:
                last_broadcast = now
                x, goal = robot_state.get_x_copy()
                ob = np.empty((0, 2))
                await broadcast(build_telemetry_msg(x, goal, 0.0, 0.0, ob))
            await asyncio.sleep(0.02)
            continue

        x, goal = robot_state.get_x_copy()

        dist_goal = math.hypot(goal[0] - x[0], goal[1] - x[1])
        if robot_state.goal_reached:
            mega.send_velocity(0.0, 0.0)
            if now - last_broadcast >= 0.2:
                last_broadcast = now
                ob = np.empty((0, 2))
                await broadcast(build_telemetry_msg(x, goal, 0.0, 0.0, ob))
            await asyncio.sleep(0.1)
            continue

        if CONFIG["use_lidar"] and lidar_serial.connected:
            ob = lidar_parser.to_obstacles_global(x)
            if len(ob) > 30:
                dist_arr = np.linalg.norm(ob - x[:2], axis=1)
                ob = ob[np.argsort(dist_arr)[:30]]
        else:
            ob = np.empty((0, 2))


        min_front = arbiter.nearest_front_dist(x, ob)
        use_dwa   = arbiter.update_state(min_front)


        if robot_state.auto_mode and nm in (NavMode.GPS_EKF_DWA, NavMode.GPS_EKF_ONLY):
            if use_dwa and nm != NavMode.GPS_EKF_DWA:
                mega.send_nav_mode(NavMode.GPS_EKF_DWA)
                robot_state.nav_mode = nm = NavMode.GPS_EKF_DWA
                _dwa_started = True
                log.info("[ARBITER] AUTO -> Mode 1 (DWA hindari rintangan)")
            elif (not use_dwa) and nm != NavMode.GPS_EKF_ONLY:
                mega.send_nav_mode(NavMode.GPS_EKF_ONLY)
                mega.send_velocity(0.0, 0.0)
                robot_state.nav_mode = nm = NavMode.GPS_EKF_ONLY
                _dwa_started = False
                log.info("[ARBITER] AUTO -> Mode 3 (PID heading ESP32)")

        if nm == NavMode.GPS_EKF_ONLY:
            robot_state.dwa_v = 0.0
            robot_state.dwa_w = 0.0
            if now - last_broadcast >= 0.1:
                last_broadcast = now
                telem3 = build_telemetry_msg(x, goal, 0.0, 0.0, ob)
                await broadcast(telem3)
                replay_logger.write(now, x, goal, 0.0, 0.0, ob,
                                    None, min_front, arbiter.avoid, nm,
                                    telem=telem3, latency_ms=(time.time() - now) * 1000.0)
            await asyncio.sleep(0.02)
            continue

        if not CONFIG["dwa_enabled"]:
            if now - last_broadcast >= 0.2:
                last_broadcast = now
                await broadcast(build_telemetry_msg(x, goal, 0.0, 0.0, ob))
            await asyncio.sleep(0.02)
            continue

        ob_plan = ob
        if _USE_CAM_EDGE:
            _cam_ob = camera_obstacles_global(x)
            if len(_cam_ob):
                ob_plan = np.vstack([ob, _cam_ob]) if len(ob) else _cam_ob
        predicted_traj = None
        _t_dwa0 = time.time()     
        try:
            loop = asyncio.get_running_loop()
            u, predicted_traj = await loop.run_in_executor(
                _dwa_pool, dwa_control, x, dwa_config, goal, ob_plan
            )
        except Exception as e:
            log.error(f"[DWA] Error: {e}")
            u = [0.0, 0.0]
        dwa_latency_ms = (time.time() - _t_dwa0) * 1000.0
        v_cmd, omega_cmd = u

   
        if abs(omega_cmd) < 0.3:
            if   v_cmd > 0: v_cmd = max(v_cmd, 0.1)
            elif v_cmd < 0: v_cmd = min(v_cmd, -0.1)


        if ENABLE_STUCK_ESCAPE:
            _rv, _rw, _ractive = stuck_recovery.update(x, ob, v_cmd, now)
            if _ractive:
                v_cmd, omega_cmd = _rv, _rw
        else:
            stuck_recovery.reset()

        if bump_active() and v_cmd > BUMP_SPEED:
            v_cmd = BUMP_SPEED
        if CONFIG["send_to_mega"]:
            mega.send_velocity(v_cmd, omega_cmd)

        robot_state.dwa_v = v_cmd
        robot_state.dwa_w = omega_cmd

        csv_logger.write(x, goal, v_cmd, omega_cmd, nm)

        if now - last_broadcast >= 0.1:
            last_broadcast = now
            telem = build_telemetry_msg(x, goal, v_cmd, omega_cmd, ob_plan, predicted_traj)
            await broadcast(telem)
            replay_logger.write(now, x, goal, v_cmd, omega_cmd, ob,
                                predicted_traj, min_front, arbiter.avoid, nm,
                                telem=telem, latency_ms=dwa_latency_ms)

        log.debug(
            f"[Mode{nm}] v={v_cmd:.3f} ω={omega_cmd:.4f} | "
            f"pos=({x[0]:.2f},{x[1]:.2f}) goal=({goal[0]:.2f},{goal[1]:.2f}) "
            f"dist={dist_goal:.2f}m ob={len(ob)}"
        )


# =============================================================================
#  LOOP: BROADCAST LIDAR (5 Hz)
# =============================================================================
async def lidar_broadcast_loop():

    while True:
        await asyncio.sleep(0.2)
        if not ws_clients:
            continue
        if CONFIG["use_lidar"] and lidar_serial.connected:
            pts = lidar_parser.filter_scan()
        else:
            pts = []
        await broadcast({
            "type":   "lidar",
            "points": [[round(px, 3), round(py, 3)] for (px, py) in pts],
            "count":  len(pts),
            "hz":     round(lidar_parser.hz, 1),
        })


# =============================================================================
#  LOOP: STATUS BROADCAST (setiap 2 detik)
# =============================================================================
async def status_loop():
    while True:
        await asyncio.sleep(2.0)
        nm, ds, wp_idx, wp_count, _ = robot_state.get_nav_state()
        await broadcast({
            "type":               "status",
            "serial_connected":   mega.connected,
            "mqtt_connected":     robot_state.mqtt_connected,  # [v7] dari ESP32
            "lidar_connected":    lidar_serial.connected,
            "dwa_running":        _dwa_started,
            "lidar_hz":           round(lidar_parser.hz, 1),
            "nav_mode":           nm,
            "nav_mode_str":       NavMode.STR_MAP.get(nm, "?"),
            "delivery_status":    ds,
            "wp_index":           wp_idx,
            "wp_total":           wp_count,
            "mega_ready":         robot_state.mega_ready,
            "version":            "argo_backend v8.0",
            "uptime":             round(time.time() - _start_time),
            "ts":                 int(time.time() * 1000),
        })


# =============================================================================
#  WEBSOCKET HANDLER
# =============================================================================
async def ws_handler(websocket, path=None):
    global _dwa_started

    ws_clients.add(websocket)
    addr = websocket.remote_address
    log.info(f"[WS] HTML terhubung: {addr}  (total={len(ws_clients)})")

    nm, ds, wp_idx, wp_count, _ = robot_state.get_nav_state()
    await websocket.send(json.dumps({
        "type":             "status",
        "serial_connected": mega.connected,
        "mqtt_connected":   robot_state.mqtt_connected,
        "lidar_connected":  lidar_serial.connected,
        "dwa_running":      _dwa_started,
        "nav_mode":         nm,
        "nav_mode_str":     NavMode.STR_MAP.get(nm, "?"),
        "delivery_status":  ds,
        "mega_ready":       robot_state.mega_ready,
        "version":          "argo_backend v7.0",
        "ts":               int(time.time() * 1000),
    }))

    await websocket.send(json.dumps({
        "type":   "dwa_config_info",
        "config": dwa_config.to_dict(),
    }))

    await websocket.send(json.dumps({
        "type":      "record_status",
        "recording": replay_logger.recording,
        "filename":  replay_logger.filename,
        "frames":    replay_logger.frame_count,
    }))

    await websocket.send(json.dumps({
        "type":      "speed_info",
        "dwa_speed": round(dwa_config.max_speed, 3),
        "pid_speed": round(arbiter.cruise_speed, 3),
    }))

    x, goal = robot_state.get_x_copy()
    ob = np.empty((0, 2))
    await websocket.send(json.dumps(
        build_telemetry_msg(x, goal, robot_state.dwa_v, robot_state.dwa_w, ob),
        default=str
    ))

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                log.warning(f"[WS] Bukan JSON: {raw[:80]}")
                continue

            t = msg.get("type", "")

            if t == "command":
                mode_name = str(msg.get("mode", "STANDBY")).upper()
                cmd_info  = NavMode.COMMAND_MAP.get(mode_name)
                if not cmd_info:
                    await websocket.send(json.dumps({
                        "type": "ack", "action": "command",
                        "success": False,
                        "error": f"Mode tidak dikenal: {mode_name}",
                    }))
                    continue

                robot_state.delivery_status = cmd_info["delivery"]
                mega_mode = cmd_info["mega_mode"]

               
                if mega_mode > 0 and robot_state.nav_mode == 0:
                    robot_state.nav_mode = NavMode.GPS_EKF_DWA  # default

                mega.send_nav_mode(mega_mode if mega_mode == 0 else robot_state.nav_mode)
                mega.send_status(mode_name)

                log.info(f"[WS] Command: {mode_name} → Mega MODE={robot_state.nav_mode}")
                await broadcast({
                    "type":            "delivery_status",
                    "status":          robot_state.delivery_status,
                    "nav_mode":        robot_state.nav_mode,
                    "ts":              int(time.time() * 1000),
                })
                await websocket.send(json.dumps({
                    "type":    "ack",
                    "action":  "command",
                    "mode":    mode_name,
                    "success": mega.connected,
                    "ts":      int(time.time() * 1000),
                }))

            elif t == "record":
                action = str(msg.get("action", "")).lower()
                if action == "start":
                    ok = replay_logger.start()
                elif action == "stop":
                    replay_logger.stop()
                    ok = True
                else:
                    ok = False
                await broadcast({
                    "type":      "record_status",
                    "recording": replay_logger.recording,
                    "filename":  replay_logger.filename,
                    "frames":    replay_logger.frame_count,
                    "ts":        int(time.time() * 1000),
                })
                await websocket.send(json.dumps({
                    "type":      "ack",
                    "action":    "record",
                    "recording": replay_logger.recording,
                    "filename":  replay_logger.filename,
                    "success":   ok,
                    "ts":        int(time.time() * 1000),
                }))

            elif t == "set_auto_mode":
                robot_state.auto_mode = bool(msg.get("enabled", True))
                log.info(f"[WS] auto_mode = {robot_state.auto_mode}")
                await broadcast({
                    "type":      "auto_mode_changed",
                    "auto_mode": robot_state.auto_mode,
                    "ts":        int(time.time() * 1000),
                })

            elif t == "set_nav_mode":
                mode_int = int(msg.get("mode", 1))
                if mode_int not in (0, 1, 2, 3):
                    await websocket.send(json.dumps({
                        "type": "ack", "action": "set_nav_mode",
                        "success": False, "error": "Mode harus 0, 1, 2, atau 3",
                    }))
                    continue

                robot_state.nav_mode = mode_int

                if mode_int == NavMode.GPS_EKF_ONLY:
                    _dwa_started = False
                    mega.send_velocity(0.0, 0.0)
                    log.info("[WS] Mode 3 → DWA Python OFF")
                elif mode_int == 0:
                    _dwa_started = False
                    mega.send_velocity(0.0, 0.0)
                    log.info("[WS] Mode 0 → STANDBY")
                else:
                    _dwa_started = True

                mega.send_nav_mode(mode_int)

                await broadcast({
                    "type":         "nav_mode_changed",
                    "nav_mode":     mode_int,
                    "nav_mode_str": NavMode.STR_MAP.get(mode_int, "?"),
                    "dwa_running":  _dwa_started,
                    "ts":           int(time.time() * 1000),
                })
                await websocket.send(json.dumps({
                    "type": "ack", "action": "set_nav_mode",
                    "nav_mode": mode_int, "success": mega.connected,
                    "ts": int(time.time() * 1000),
                }))

            elif t == "set_delivery_status":
                status = str(msg.get("status", "STANDBY")).upper()
                valid  = (NavMode.STANDBY, NavMode.PICKUP, NavMode.DELIVERY)
                if status not in valid:
                    await websocket.send(json.dumps({
                        "type": "ack", "action": "set_delivery_status",
                        "success": False,
                        "error": f"Status tidak valid. Pilih: {valid}",
                    }))
                    continue
                robot_state.delivery_status = status
                mega.send_status(status)
                log.info(f"[WS] Delivery status → {status}")
                await broadcast({
                    "type": "delivery_status",
                    "status": status,
                    "ts": int(time.time() * 1000),
                })
                await websocket.send(json.dumps({
                    "type": "ack", "action": "set_delivery_status",
                    "status": status, "success": True,
                    "ts": int(time.time() * 1000),
                }))

            elif t == "waypoint":
                wps = msg.get("waypoints", [])
                valid_wps = [w for w in wps if w.get("lat") is not None and w.get("lon") is not None]
                robot_state.set_waypoints(valid_wps)
                if valid_wps:
                    arbiter.avoid          = False
                    arbiter._clear_since   = None
                    arbiter._last_front    = float("inf")
                    arbiter._last_front_ts = 0.0
                    # [FIX-7] asyncio.get_running_loop()
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(None, mega.send_waypoints_gps, valid_wps)
                log.info(f"[WS] {len(valid_wps)} waypoint GPS diterima")
                await websocket.send(json.dumps({
                    "type": "ack", "action": "waypoint",
                    "count": len(valid_wps), "success": mega.connected,
                    "ts": int(time.time() * 1000), 
                }))

            elif t == "waypoint_odo":
                wps = msg.get("waypoints", [])
                valid_wps = [w for w in wps if w.get("x") is not None and w.get("y") is not None]
                robot_state.set_waypoints(valid_wps, is_odo=True)
                if valid_wps:
                    arbiter.avoid          = False
                    arbiter._clear_since   = None
                    arbiter._last_front    = float("inf")
                    arbiter._last_front_ts = 0.0
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(None, mega.send_waypoints_odo, valid_wps)
                log.info(f"[WS] {len(valid_wps)} waypoint Odo diterima")
                await websocket.send(json.dumps({
                    "type": "ack", "action": "waypoint_odo",
                    "count": len(valid_wps), "success": mega.connected,
                    "ts": int(time.time() * 1000),
                }))

            elif t == "dwa_config":
                dwa_config.update_from_dict(msg)
                await websocket.send(json.dumps({
                    "type":    "ack",
                    "action":  "dwa_config",
                    "config":  dwa_config.to_dict(),
                    "success": True,
                }))

            elif t == "get_config":
                await websocket.send(json.dumps({
                    "type":   "dwa_config_info",
                    "config": dwa_config.to_dict(),
                }))

            elif t == "set_speed":
                applied = {}
                if "dwa" in msg:
                    nv = set_dwa_speed(msg.get("dwa"))
                    if nv is not None: applied["dwa"] = nv
                if "pid" in msg:
                    nv = set_pid_speed(msg.get("pid"))
                    if nv is not None: applied["pid"] = nv
                await broadcast({
                    "type":      "speed_info",
                    "dwa_speed": round(dwa_config.max_speed, 3),
                    "pid_speed": round(arbiter.cruise_speed, 3),
                    "ts":        int(time.time() * 1000),
                })
                await websocket.send(json.dumps({
                    "type":      "ack", "action": "set_speed",
                    "applied":   applied, "success": True,
                    "dwa_speed": round(dwa_config.max_speed, 3),
                    "pid_speed": round(arbiter.cruise_speed, 3),
                }))

            elif t == "serial_cmd":
                cmd = str(msg.get("cmd", "")).strip()
                if cmd:
                    mega.send_command(cmd)
                    log.info(f"[WS→Mega] serial_cmd: {cmd}")

            elif t == "clrwp":
                mega.send_command("<CLRWP>")
                robot_state.clear_waypoints()
                log.info("[WS] CLRWP dikirim")
                await websocket.send(json.dumps({
                    "type": "ack", "action": "clrwp", "success": True,
                }))

            elif t == "reset":
                mega.send_command("RESET")
                robot_state.reset_position()
                log.info("[WS] RESET dikirim")
                await websocket.send(json.dumps({
                    "type": "ack", "action": "reset", "success": True,
                }))

            elif t == "door":
                mega.send_command("DOOR")
                log.info("[WS] DOOR dikirim")
                await websocket.send(json.dumps({
                    "type": "ack", "action": "door", "success": True,
                }))

            elif t == "dwa_start":
                if robot_state.nav_mode == NavMode.GPS_EKF_ONLY:
                    await websocket.send(json.dumps({
                        "type": "ack", "action": "dwa_start",
                        "success": False,
                        "error": "Mode 3: DWA dijalankan Arduino, bukan Python",
                    }))
                    continue
                _dwa_started = True
                log.info("[WS] DWA dimulai")
                await broadcast({
                    "type": "status", "dwa_running": True,
                    "ts": int(time.time() * 1000),
                })

            elif t == "dwa_stop":
                _dwa_started = False
                mega.send_velocity(0.0, 0.0)
                log.info("[WS] DWA dihentikan")
                await broadcast({
                    "type": "status", "dwa_running": False,
                    "ts": int(time.time() * 1000),
                })

            elif t == "estop":
                _dwa_started = False
                mega.send_velocity(0.0, 0.0)
                mega.send_command("<MODE,0>")
                mega.send_command("<STATUS,STANDBY>")
                robot_state.delivery_status = NavMode.STANDBY
                robot_state.nav_mode = 0
                log.warning("[WS] EMERGENCY STOP!")
                await broadcast({
                    "type":            "status",
                    "dwa_running":     False,
                    "estop":           True,
                    "nav_mode":        0,
                    "delivery_status": "STANDBY",
                    "ts":              int(time.time() * 1000),
                })

            elif t == "ping":
                await websocket.send(json.dumps({
                    "type":        "pong",
                    "ts":          int(time.time() * 1000),
                    "mega_ok":     mega.connected,
                    "lidar_ok":    lidar_serial.connected,
                    "dwa_running": _dwa_started,
                    "nav_mode":    robot_state.nav_mode,
                }))

            else:
                log.debug(f"[WS] Tipe tidak dikenal: {t}")

    except websockets.exceptions.ConnectionClosedOK:
        pass
    except Exception as e:
        log.warning(f"[WS] Error [{addr}]: {e}")
    finally:
        ws_clients.discard(websocket)
        log.info(f"[WS] HTML disconnect: {addr}")


# =============================================================================
#  BANNER
# =============================================================================
def print_banner():
    p  = CONFIG["ws_port"]
    mp = CONFIG["mega_port"]
    lp = CONFIG["lidar_port"]
    print(f"""
-------------------------------------------------------------------
|           ARGO Backend v8.0 — HTML ↔ Arduino Mega Bridge        |
-------------------------------------------------------------------
|                     Mega port    : {mp:<46}                     |
|                     LiDAR port   : {lp:<46}                     |
|               WS Server    : ws://localhost:{p:<31}             |
|-----------------------------------------------------------------|
|                    DWA AND BACKEND RUNNING . . .                |
-------------------------------------------------------------------
""")

def camera_thread():
    if cam is None or not CONFIG["use_camera"]:
        return
    last_bump_cmd = False
    last_bump_ts  = 0.0
    last_kl       = 0.0
    while True:
        time.sleep(0.02)
        now = time.time()

        if CONFIG["send_to_mega"]:
            _b = bump_active()
            if _b != last_bump_cmd or (now - last_bump_ts) > 0.3:
                last_bump_cmd = _b
                last_bump_ts  = now
                mega.send_bump(_b)

        if now - last_kl < dwa_config.dt:
            continue
        last_kl = now
        th_cmd = th_raw = None
        try:
            kl = cam.get_keepleft_target()
            th_cmd, th_raw = cam_heading.update(kl, dwa_config.dt)
        except Exception as e:
            log.debug(f"[CAM] get_keepleft_target err: {e}")
        if th_cmd is None:
            if time.time() - cam_heading.last_log >= 0.5:
                cam_heading.last_log = time.time()
                log.info("[CAM->Mega] tali hilang -> kirim <CAMLOST> (fallback SEKETIKA ke waypoint)")
            if CONFIG["send_to_mega"]:
                try: mega.send_camera_lost()
                except Exception: pass
            try: cam.set_target_heading(None)
            except Exception: pass
        else:
            deg = max(-CAM_MAX_ABS_DEG,
                      min(CAM_MAX_ABS_DEG, math.degrees(th_cmd) * CAM_SIGN))
            if time.time() - cam_heading.last_log >= 0.3:
                cam_heading.last_log = time.time()
                log.info(f"[CAM->Mega] CAMHDG = {deg:+6.1f} deg  (raw {math.degrees(th_raw):+6.1f} deg)")
            if CONFIG["send_to_mega"]:
                mega.send_camera_heading(deg)
            try: cam.set_target_heading(th_cmd, err=th_cmd, raw=th_raw)
            except Exception: pass


async def main_async():
    mega.connect()
    lidar_serial.connect()

    if cam is not None and CONFIG["use_camera"]:
        try:
            cam.start()
            log.info("[CAM] Bridge ENet dimulai (target heading tepi kiri).")
        except Exception as e:
            log.warning(f"[CAM] Gagal start bridge ENet: {e}")

    if CONFIG["use_lidar"]:
        t = threading.Thread(target=lidar_thread, daemon=True)
        t.start()


    if cam is not None and CONFIG["use_camera"]:
        tc = threading.Thread(target=camera_thread, daemon=True)
        tc.start()

    server = await websockets.serve(
        ws_handler,
        CONFIG["ws_host"],
        CONFIG["ws_port"],
        ping_interval=20,
        ping_timeout=30,
    )
    log.info(f"[WS] Server aktif: ws://localhost:{CONFIG['ws_port']}")

    await asyncio.gather(
        dwa_loop(),
        broadcaster_loop(),
        lidar_broadcast_loop(),
        status_loop(),
    )


def main():
    import argparse

    ap = argparse.ArgumentParser(description="ARGO Backend v6.0")
    ap.add_argument("--mega-port",  default=CONFIG["mega_port"],  dest="mega_port")
    ap.add_argument("--lidar-port", default=CONFIG["lidar_port"], dest="lidar_port")
    ap.add_argument("--ws-port",    type=int, default=CONFIG["ws_port"], dest="ws_port")
    ap.add_argument("--lidar",      action="store_true",
                    help="Aktifkan LiDAR (Delta2G) utk deteksi obstacle + auto-switch ke DWA")
    ap.add_argument("--no-lidar",   action="store_true")
    ap.add_argument("--no-send",    action="store_true",
                    help="Jangan kirim v/omega ke Mega (dev mode)")
    ap.add_argument("--no-dwa",     action="store_true")
    ap.add_argument("--no-csv",     action="store_true")
    ap.add_argument("--debug",      action="store_true")
    ap.add_argument("--no-camera",  action="store_true",
                    help="Nonaktifkan bridge ENet (tanpa keep-line kamera)")
    ap.add_argument("--offline",    default=None,
                    help="Path video utk uji keep-line tanpa webcam (mis. Test.mp4)")
    ap.add_argument("--keepline",   action="store_true",
                    help="Langsung masuk Mode 3 (keep-line kamera) saat start")
    args = ap.parse_args()

    CONFIG["mega_port"]  = args.mega_port
    CONFIG["lidar_port"] = args.lidar_port
    CONFIG["ws_port"]    = args.ws_port
    if args.lidar:    CONFIG["use_lidar"]    = True
    if args.no_lidar: CONFIG["use_lidar"]    = False
    if args.no_send:  CONFIG["send_to_mega"] = False
    if args.no_dwa:   CONFIG["dwa_enabled"]  = False
    if args.no_csv:   CONFIG["log_csv"]      = False
    CONFIG["keepline_start"] = args.keepline
    if args.keepline:  CONFIG["use_camera"] = True
    if args.no_camera: CONFIG["use_camera"] = False
    if args.offline and cam is not None:
        cam.VIDEO_SOURCE = args.offline
        log.info(f"[CAM] OFFLINE video: {args.offline}")

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    global _dwa_started
    _dwa_started = CONFIG["dwa_enabled"]

    print_banner()

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        log.info("\nMenghentikan backend...")
    finally:
        mega.stop()
        csv_logger.close()
        replay_logger.close()
        log.info("Backend berhenti.")


if __name__ == "__main__":
    main()