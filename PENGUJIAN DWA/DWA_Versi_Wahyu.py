"""
Mobile robot motion planning sample with Dynamic Window Approach
"""

import math
import struct
import atexit
from enum import Enum
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sys
import numpy as np
import json
import serial
import time
from collections import deque

log_file = open("DWA_4.csv", "w")
log_file.write("time,x,y,yaw,v_cmd,omega_cmd,v_actual,omega_actual\n")

show_animation = True
simulation = True
send_to_robot = False

robot_port = 'COM8'
lidar_port = 'COM9'
# baudrate = 230400
lidar_buffer = bytearray()

ser_lidar = None
ser_robot = None
dwa_started = False
USE_OBSTACLE = False
buffer_points = []
last_angle = 0
last_dist = {}

if send_to_robot:
    ser_robot = serial.Serial(robot_port, 115200, timeout=0.1)
    time.sleep(2)
    print("Serial robot terhubung")
    ser_robot.write(b"0.0,0.0\n")

if not simulation and USE_OBSTACLE:
    ser_lidar = serial.Serial(lidar_port, 230400, timeout=0)
    time.sleep(2)
    print("Serial LiDAR terhubung")
else:
    ser_lidar = None


def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory, all_traj = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory, all_traj


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 0.6  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 360.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 1.0  # [m/ss]
        self.max_delta_yaw_rate = 360.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.05  # [m/s]
        self.yaw_rate_resolution = 5 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.5  # [s]
        self.to_goal_cost_gain = 0.12
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 2.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.rectangle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.45  # [m] for collision check
        self.robot_length = 0.65  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = np.array([[-0.5, 5],
                            [3, 10],
                            [0, 15]
                            # [0.25, 15],
                            # [-0.25, 15]
                            ])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()


def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    # if x[3] < 0.1:
    #     v = max(v, 0.2)

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    all_trajectories = []
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)
            all_trajectories.append(trajectory)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory, all_trajectories


def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """

    if ob is None or len(ob) == 0:
        return 0.0  # tidak ada obstacle → tidak ada cost

    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost

def read_json(ser):

    if ser.in_waiting == 0:
        return None, None

    latest_line = None

    while ser.in_waiting > 0:
        try:
            latest_line = ser.readline().decode(errors='ignore').strip()
        except:
            return None, None

    if latest_line is None or not latest_line.startswith("{"):
        return None, None

    try:
        data = json.loads(latest_line)

        x = np.array([
            float(data["x"]),
            float(data["y"]),
            float(data["yaw"]),
            float(data["v"]),
            float(data["w"])
        ])

        goal = np.array([
            float(data["goal_x"]),
            float(data["goal_y"])
        ])

        return x, goal

    except:
        return None, None
    
def read_lidar_points(ser_lidar):
    global lidar_buffer

    # tambah data baru ke buffer
    data = ser_lidar.read(1024)
    lidar_buffer.extend(data)

    points = []

    # proses per paket (10 byte)
    while len(lidar_buffer) >= 6:

        if lidar_buffer[0] == 0xAA and lidar_buffer[1] == 0x55:

            packet = lidar_buffer[2:6]

            if len(packet) == 4:
                angle_raw, dist_raw = struct.unpack('HH', packet)

                angle = angle_raw / 100.0
                dist = dist_raw / 1000.0

                if 0.35 < dist < 6.5:
                    points.append((angle, dist))

            lidar_buffer = lidar_buffer[6:]

        else:
            lidar_buffer = lidar_buffer[1:]

    return points

def update_lidar_only():
    global lidar_scatter, latest_lidar_points, last_angle, buffer_points

    if ser_lidar is None:
        return

    new_points = read_lidar_points(ser_lidar)

    for angle, dist in new_points:

        # DETEKSI 1 PUTARAN
        if 'last_angle' in globals():
            if angle < last_angle:
                # tampilkan 1 scan penuh
                pts = []

                for a, d in buffer_points:
                    a = (-a - 110) % 360

                    if 0.35 < d < 6.5:
                        rad = np.deg2rad(a)
                        x = d * np.cos(rad)
                        y = d * np.sin(rad)

                        pts.append({'pos': (x, y)})

                lidar_scatter.setData(pts)
                latest_lidar_points = buffer_points.copy()

                buffer_points.clear()

        buffer_points.append((angle, dist))
        last_angle = angle

def lidar_to_obstacles_global(x, lidar_points):

    temp_points = []

    for angle_deg, dist in lidar_points:

        # FILTER JARAK (perketat)
        if dist < 0.2 or dist > 6.0:
            continue

        # TRANSFORM SUDUT
        angle_deg = (-angle_deg - 100) % 360

        # FILTER DEPAN SAJA
        if not (angle_deg >= 270 or angle_deg <= 90):
            continue

        angle = math.radians(angle_deg)

        # GLOBAL POSITION
        ox = x[0] + dist * math.cos(x[2] + angle)
        oy = x[1] + dist * math.sin(x[2] + angle)

        temp_points.append((ox, oy))

    # NEIGHBOR FILTER (ANTI-NOISE)
    filtered = []

    for i, (x1, y1) in enumerate(temp_points):
        count = 0

        for j, (x2, y2) in enumerate(temp_points):
            if i == j:
                continue

            if np.hypot(x1 - x2, y1 - y2) < 0.3:
                count += 1

        if count >= 1:  # minimal punya tetangga
            filtered.append([x1, y1])

    if len(filtered) == 0:
        return np.empty((0,2))

    return np.array(filtered)

# def sensor_to_obstacles(x, distances):

#     sensor_angles = np.radians([0,45,90,270,315])
#     obstacles = []

#     for d, angle in zip(distances, sensor_angles):

#         # # SKIP kalau terlalu jauh
#         # if d > 2.5:
#         #     continue

#         # # SKIP kalau aneh
#         # if d < 0.05:
#         #     continue

#         ox = x[0] + d * math.cos(x[2] + angle)
#         oy = x[1] + d * math.sin(x[2] + angle)

#         obstacles.append([ox, oy])

#         if d > 2.5:
#             continue
#         if d < 0.05:
#             continue
    
#     if len(obstacles) == 0:
#         return np.empty((0,2))

#     return np.array(obstacles)


def main(gx=0.0, gy=20.0, robot_type=RobotType.circle):
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    if simulation:
        x = np.array([0.0, 0.0, math.pi / 2.0, 0.0, 0.0])
        # goal position [x(m), y(m)]
        goal = np.array([gx, gy])
        ob = config.ob
        trajectory = np.array(x)

    # input [forward speed, yaw_rate]
    config.robot_type = robot_type
    ob = config.ob


if __name__ == '__main__':

    # ================= SIMULASI TANPA GUI =================
    if not show_animation:

        x = np.array([0.0, 0.0, math.pi/2, 0.0, 0.0])
        goal = np.array([0.0, 20.0])

        dt = config.dt
        last_time = time.time()

        pos_history = deque(maxlen=10)

        try:
            while True:

                now = time.time()

                # kontrol waktu stabil
                if now - last_time < 0.05:
                    time.sleep(0.001)
                    continue

                last_time += dt

                # ================== BACA DATA ==================
                if simulation:
                    ob = config.ob
                else:
                    x_real, goal_real = read_json(ser_robot)

                    if x_real is None:
                        if send_to_robot:
                            ser_robot.write(b"0.0,0.0\n")
                        continue

                    x[:] = x_real
                    goal[:] = goal_real

                    if USE_OBSTACLE:
                        lidar_points = read_lidar_points(ser_lidar)
                        ob = lidar_to_obstacles_global(x, lidar_points)
                    else:
                        ob = np.empty((0,2))

                # ================== DWA ==================
                u, _ = dwa_control(x, config, goal, ob)

                v, omega = u

                if v > 0:
                    v = max(v, 0.1)
                elif v < 0:
                    v = min(v, -0.1)

                # # LIMIT SPEED
                # v = np.clip(v, 0.0, 0.5)
                # omega = np.clip(omega, -1.0, 1.0)

                # ================== DETEKSI STUCK ==================
                pos_history.append(x[:2])

                if len(pos_history) == pos_history.maxlen:
                    movement = sum(
                        np.linalg.norm(pos_history[i] - pos_history[i-1])
                        for i in range(1, len(pos_history))
                    )

                    if movement < 0.05:
                        print("STUCK → recovery")
                        if send_to_robot:
                            ser_robot.write(b"0.0,1.0\n")
                        time.sleep(0.5)
                        continue

                # ================== KIRIM KE ROBOT ==================
                if send_to_robot:
                    ser_robot.write(f"{v:.2f},{omega:.4f}\n".encode())
                
                if simulation:
                    x = motion(x, u, dt)
                else:
                    if x_real is not None:
                        x[:] = x_real

                # DETEKSI COMMAND TIDAK DIEKSEKUSI
                actual_v = x[3]

                print(f"v={v:.2f}, omega={omega:.2f}, x={x[0]:.2f}, y={x[1]:.2f}")

        except KeyboardInterrupt:
            print("Program dihentikan!")

        finally:
            if send_to_robot:
                ser_robot.write(b"0.0,0.0\n")
                print("Robot STOP")

    # ================= SIMULASI + ANIMASI =================
    else:
        app = QtWidgets.QApplication(sys.argv)
        win = pg.GraphicsLayoutWidget(show=True, title="DWA Real-Time")

        print("Tekan Y di window untuk mulai robot")

        def keyPressEvent(event):
            global dwa_started

            if event.text().upper() == 'Y':
                dwa_started = True
                print("Robot mulai bergerak!")

        # ================= LIDAR RADAR =================
        lidar_plot = win.addPlot(title="LiDAR View")

        lidar_plot.setAspectLocked()
        lidar_plot.setXRange(-2, 2)
        lidar_plot.setYRange(-2, 2)

        # LINGKARAN RADAR
        for r in [0.5, 1.0, 1.5, 2.0]:
            circle = pg.QtWidgets.QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
            circle.setPen(pg.mkPen((100, 100, 100), width=1))
            lidar_plot.addItem(circle)
        # GARIS DEPAN ROBOT
        heading_line = pg.PlotDataItem([0, 2], [0, 0],
                                    pen=pg.mkPen('g', width=2))
        lidar_plot.addItem(heading_line)

        # POSISI ROBOT (TENGAH)
        robot_center = pg.ScatterPlotItem(size=10, brush='r')
        robot_center.setData([0], [0])
        lidar_plot.addItem(robot_center)

        lidar_plot.setAspectLocked()
        lidar_plot.setXRange(-2, 2)
        lidar_plot.setYRange(-2, 2)
        lidar_plot.showGrid(x=True, y=True)

        lidar_scatter = pg.ScatterPlotItem(size=5, brush='c')
        lidar_plot.addItem(lidar_scatter)

        win.nextRow()

        win.keyPressEvent = keyPressEvent

        plot = win.addPlot()

        plot.setAspectLocked()
        plot.setXRange(-5, 15)
        plot.setYRange(-5, 25)
        plot.showGrid(x=True, y=True)

        traj_plot = pg.PlotDataItem(pen=pg.mkPen('g', width=2))
        robot_plot = pg.ScatterPlotItem(size=10, brush='r')
        goal_plot = pg.ScatterPlotItem(size=10, brush='b')
        ob_plot = pg.ScatterPlotItem(size=6, brush='w')

        plot.addItem(traj_plot)
        candidate_plots = []
        for i in range(200):
            p = pg.PlotDataItem(
                pen=pg.mkPen((100,100,100,80), width=1)
            )
            plot.addItem(p)
            candidate_plots.append(p)
        plot.addItem(robot_plot)
        plot.addItem(goal_plot)
        plot.addItem(ob_plot)

        robot_body = pg.PlotDataItem(pen=pg.mkPen('r', width=2))
        plot.addItem(robot_body)

        config.robot_type = RobotType.rectangle

        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([0.0, 20.0])
        trajectory = np.array(x)

        if not simulation:
            ser_robot
            ser_robot.timeout = 0.1
            time.sleep(2)
        # arrow = pg.ArrowItem()
        # plot.addItem(arrow)

        history_plot = pg.PlotDataItem(pen=pg.mkPen('y', width=3))
        plot.addItem(history_plot)
        
        def draw_robot(x, config):
            L = config.robot_length
            W = config.robot_width

            # koordinat rectangle (center di 0,0)
            corners = np.array([
                [ L/2,  W/2],
                [ L/2, -W/2],
                [-L/2, -W/2],
                [-L/2,  W/2],
                [ L/2,  W/2]  # tutup polygon
            ])

            # rotasi
            yaw = x[2]
            R = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw),  np.cos(yaw)]
            ])

            rotated = corners @ R.T

            # translasi ke posisi robot
            rotated[:, 0] += x[0]
            rotated[:, 1] += x[1]

            return rotated
        goal_reached = False

        def update():
            if not dwa_started:
                return
            
            global x, trajectory, goal_reached

            if goal_reached:
                return

            # ================= PILIH OBSTACLE =================
            if simulation:
                ob = config.ob  # simulasi
            else:
                # ambil state robot
                x_real, goal_real = read_json(ser_robot)

                if x_real is not None:
                    x[:] = x_real
                    goal[:] = goal_real
                else:
                    return

                # ambil lidar
                # lidar_points = read_lidar_points(ser_lidar)
                # step = max(1, len(lidar_points)//120)
                # lidar_points = lidar_points[::step]

                # konversi ke obstacle
                if USE_OBSTACLE:
                    lidar_points = latest_lidar_points
                    ob = lidar_to_obstacles_global(x, lidar_points)

                    # ambil jarak semua obstacle
                    dist = np.linalg.norm(ob - x[:2], axis=1)

                    # filter jarak aman
                    mask = (dist < 4.5) & (dist > 0.3)
                    ob = ob[mask]

                    # ambil obstacle terdekat (setelah bersih)
                    if len(ob) > 30:
                        idx = np.argsort(dist)
                        ob = ob[idx[:30]]
                        
                else:
                    ob = np.empty((0,2))

                # # LIMIT OBSTACLE
                # if len(ob) > 500:
                #     dist = np.linalg.norm(ob - x[:2], axis=1)
                #     idx = np.argsort(dist)
                #     ob = ob[idx[:50]]

            u, predicted_trajectory, all_traj = dwa_control(x, config, goal, ob)

            v = u[0]
            omega = u[1]

            # ===== LOG DATA =====
            t = time.time()

            v_cmd = v
            omega_cmd = omega

            v_actual = x[3]
            omega_actual = x[4]

            log_file.write(f"{t},{x[0]},{x[1]},{x[2]},{v_cmd},{omega_cmd},{v_actual},{omega_actual}\n")
            log_file.flush()

            if np.linalg.norm(goal - x[:2]) < 0.5:
                print("Goal reached!")
                goal_reached = True
                if send_to_robot:
                    ser_robot.write(b"0.0,0.0\n")  # STOP
                    x = motion(x, u, config.dt)

            traj_plot.setData(predicted_trajectory[:, 0], predicted_trajectory[:, 1])

            if goal_reached:
                history_plot.setData(trajectory[:, 0], trajectory[:, 1])
                return

            # ================= KIRIM KE ROBOT =================
            if send_to_robot:
                v = u[0]
                omega = u[1]

                if v < 0.1:
                    v = max(v, 0.1)

                msg = f"{v:.2f},{omega:.4f}\n"
                ser_robot.write(msg.encode())

            now = time.time()

            if 'last_print' not in globals():
                last_print = now

            dt_debug = now - last_print
            last_print = now
            
            print(f"v={v:.2f} m/s | omega={omega:.4f} rad/s | x={x[0]:.2f} | y={x[1]:.2f} | yaw={x[2]:.2f}")

            x = motion(x, u, config.dt)
            trajectory = np.vstack((trajectory, x))

            traj_plot.setData(predicted_trajectory[:, 0], predicted_trajectory[:, 1])
            robot_plot.setData([x[0]], [x[1]])
            goal_plot.setData([goal[0]], [goal[1]])

            # arrow.setPos(x[0], x[1])
            # arrow.setStyle(angle=np.degrees(x[2]))

            body = draw_robot(x, config)
            robot_body.setData(body[:, 0], body[:, 1])

            if len(ob) > 0:
                ob_plot.setData(ob[:, 0], ob[:, 1])
            
            if goal_reached:
                history_plot.setData(trajectory[:, 0], trajectory[:, 1])

            for i, traj in enumerate(all_traj):
                if i < len(candidate_plots):
                    candidate_plots[i].setData(
                        traj[:,0],
                        traj[:,1]
                    )

            for i in range(len(all_traj), len(candidate_plots)):
                candidate_plots[i].setData([], [])

        
        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(20)
        lidar_timer = QtCore.QTimer()
        lidar_timer.timeout.connect(update_lidar_only)
        lidar_timer.start(10)

        sys.exit(app.exec())

def stop_robot():
    global ser_robot
    try:
        if send_to_robot and ser_robot is not None:
            ser_robot.write(b"0.0,0.0\n")
            print("Robot STOP (exit)")
    except:
        pass

def close_log():
    try:
        log_file.close()
        print("Log saved")
    except:
        pass

atexit.register(close_log)

atexit.register(stop_robot)