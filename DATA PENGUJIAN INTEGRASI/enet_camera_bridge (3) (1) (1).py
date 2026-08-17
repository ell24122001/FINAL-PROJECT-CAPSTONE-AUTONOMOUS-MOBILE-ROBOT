import cv2, numpy as np  
INFER_BACKEND = "onnx"

_HAS_TORCH = False
torch = None
nn = None
F = None
if INFER_BACKEND == "torch":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
    print("torch     :", torch.__version__)
    print("cuda ok   :", torch.cuda.is_available())
else:
    import types as _types
    nn = _types.SimpleNamespace(Module=object)
    import onnxruntime as ort
    print("onnxruntime:", ort.__version__)
print("opencv    :", cv2.__version__)


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


INFER_THREADS = 0   
try:
    import multiprocessing as _mp
    _ncpu = _mp.cpu_count()
except Exception:
    _ncpu = 4
_nthreads = INFER_THREADS if INFER_THREADS > 0 else max(1, _ncpu - 2)  
os.environ["OMP_NUM_THREADS"]      = str(_nthreads)
os.environ["MKL_NUM_THREADS"]      = str(_nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(_nthreads)
if _HAS_TORCH:
    torch.set_num_threads(_nthreads)      
print("[ENet] CPU inference threads =", _nthreads, "(core terdeteksi:", _ncpu, ")")

MODEL_PATH      = r"best_model.pth"    
ONNX_MODEL_PATH = r"enet_model.onnx"    
HOMOGRAPHY_PATH = r"homography.txt"
CAMERA_MATRIX_PATH = r"C:\Argo_integrat_v1\camera_matrix.npy"
DIST_COEFFS_PATH   = r"C:\Argo_integrat_v1\dist_coeffs.npy"

# ====== MODEL / GAMBAR ======
NUM_CLASSES = 4
IMG_HEIGHT  = 256
IMG_WIDTH   = 512
CALIB_WIDTH  = 1920   
CALIB_HEIGHT = 1080

# ====== KELAS ======
DRIVABLE_CLASS_IDXS = [1, 3]  
SPEED_BUMP_CLASS    = 3

# ====== PROYEKSI PIKSEL -> TANAH ======
USE_HOMOGRAPHY = True     
USE_UNDISTORT  = False
                           
CAM_HEIGHT = 0.5; FOV_H = 90; FOV_V = 60   

# ====== EKSTRAKSI BOUNDARY ======
BOUNDARY_ROWS_START  = 0.6
BOUNDARY_SAMPLE_STEP = 4
MIN_DRIVABLE_PX = 16

SHOW_KEEPLEFT_LINE = True
KEEPLEFT_OFFSET_M  = 1.20

BORDER_MARGIN_PX = 3
OB_MIN_DIST = 0.3
OB_MAX_DIST = 4.0
MAX_OB_POINTS = 40

# ====== HAZARD INTERIOR ======
DETECT_INTERIOR_HAZARD = True   
HAZARD_MIN_RUN = 8             
MAX_HAZARD_POINTS = 8      

# ====== SPEED BUMP ======
SPEED_BUMP_DETECTION_ROWS = 0.7
SPEED_BUMP_MIN_PX = 20
SPEED_BUMP_ZONE_TOP = 0.85

# ====== VISUALISASI ======
PLOT_RANGE_X = (-5, 5)   # samping (m)
PLOT_RANGE_Y = (-1, 6)   # depan (m)
ROBOT_WIDTH  = 0.45
ROBOT_LENGTH = 0.65
START_FRAME = 2400     
MAX_FRAMES = 120         
FRAME_STEP = 1          

DEVICE = (torch.device("cuda" if torch.cuda.is_available() else "cpu")) if _HAS_TORCH else "cpu"
print("Device:", DEVICE)

# ====== WEBCAM (real-time) ======
CAM_INDEX = 0        
CAM_REQ_WIDTH = 1280     
CAM_REQ_HEIGHT = 720
FRAME_PROCESS_EVERY = 1  
SHOW_WINDOW = True     
PREVIEW_SCALE = 2.0     
PREVIEW_RESIZABLE = False 
PREVIEW_WINDOW = 'ARGO ENet preview'

# ====== SUMBER INPUT (troubleshooting bertahap) ======

VIDEO_SOURCE    = None        
VIDEO_LOOP      = True         
VIDEO_FPS_LIMIT = 0            

H_matrix = None
if USE_HOMOGRAPHY:
    try:
        H_matrix = np.loadtxt(HOMOGRAPHY_PATH)
        print('[ENet] Homography loaded:', H_matrix.shape)
    except Exception as e:
        print('[ENet] WARNING: homography gagal dimuat -> pakai geometri sederhana.', e)
        USE_HOMOGRAPHY = False

undistort_K = undistort_D = None
if USE_UNDISTORT:
    try:
        undistort_K = np.load(CAMERA_MATRIX_PATH)
        undistort_D = np.load(DIST_COEFFS_PATH)
        print('[ENet] Undistort aktif.')
    except Exception as e:
        print('[ENet] WARNING: kalibrasi gagal dimuat -> undistort off.', e)
        USE_UNDISTORT = False

def maybe_undistort(frame_bgr):
    if USE_UNDISTORT and undistort_K is not None:
        return cv2.undistort(frame_bgr, undistort_K, undistort_D)
    return frame_bgr

# Homografi dihitung pada foto kalibrasi 1920x1080, mask 512x256.
# Skala-kan koordinat mask ke resolusi kalibrasi sebelum dikali H.
SCALE_U = CALIB_WIDTH / IMG_WIDTH
SCALE_V = CALIB_HEIGHT / IMG_HEIGHT

def pixel_to_ground(u, v):

    if USE_HOMOGRAPHY and H_matrix is not None:
        pt = np.array([u * SCALE_U, v * SCALE_V, 1.0])   
        r = H_matrix @ pt; r /= r[2]
        samping, depan = r[0], r[1]   
        return depan, samping
    norm_v = (v - IMG_HEIGHT) / IMG_HEIGHT
    norm_u = (u - IMG_WIDTH / 2) / (IMG_WIDTH / 2)
    angle_v = np.radians(abs(norm_v) * FOV_V / 2)
    angle_h = np.radians(norm_u * FOV_H / 2)
    if angle_v < 1e-6:
        return None, None
    x_local = CAM_HEIGHT / np.tan(angle_v)
    y_local = x_local * np.tan(angle_h)
    return x_local, y_local

if _HAS_TORCH:
    import torch.nn as nn
    import torch.nn.functional as F

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1); module.bias.data.zero_()

class InitialBlock(nn.Module):
    def __init__(self, in_channels, use_prelu=True):
        super().__init__()
        self.pool  = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv  = nn.Conv2d(in_channels, 16 - in_channels, 3, padding=1, stride=2)
        self.bn    = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU(16) if use_prelu else nn.ReLU(inplace=True)
    def forward(self, x):
        x = torch.cat((self.pool(x), self.conv(x)), dim=1)
        return self.prelu(self.bn(x))

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels=None, dilation=1, downsample=False,
                 proj_ratio=4, upsample=False, asymetric=False,
                 regularize=True, p_drop=None, use_prelu=True):
        super().__init__()
        self.pad = 0; self.upsample = upsample; self.downsample = downsample
        if out_channels is None: out_channels = in_channels
        else: self.pad = out_channels - in_channels
        if regularize: assert p_drop is not None
        if downsample: assert not upsample
        elif upsample: assert not downsample
        inter_channels = in_channels // proj_ratio
        if upsample:
            self.spatil_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn_up = nn.BatchNorm2d(out_channels)
            self.unpool = nn.MaxUnpool2d(2, 2)
        elif downsample:
            self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv1  = nn.Conv2d(in_channels, inter_channels, 2 if downsample else 1,
                                stride=2 if downsample else 1, bias=False)
        self.bn1    = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        if asymetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, (1,5), padding=(0,2)),
                nn.BatchNorm2d(inter_channels),
                nn.PReLU() if use_prelu else nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, inter_channels, (5,1), padding=(2,0)))
        elif upsample:
            self.conv2 = nn.ConvTranspose2d(inter_channels, inter_channels, 3, padding=1,
                                            output_padding=1, stride=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3,
                                   padding=dilation, dilation=dilation, bias=False)
        self.bn2    = nn.BatchNorm2d(inter_channels)
        self.prelu2 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        self.conv3  = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn3    = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu_out   = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
    def forward(self, x, indices=None, output_size=None):
        identity = x
        if self.upsample:
            assert (indices is not None) and (output_size is not None)
            identity = self.bn_up(self.spatil_conv(identity))
            if identity.size() != indices.size():
                pad = (indices.size(3)-identity.size(3),0,indices.size(2)-identity.size(2),0)
                identity = F.pad(identity, pad, "constant", 0)
            identity = self.unpool(identity, indices=indices)
        elif self.downsample:
            identity, idx = self.pool(identity)
            if self.pad > 0:
                extras = torch.zeros((identity.size(0), self.pad, identity.size(2), identity.size(3)), device=x.device)
                identity = torch.cat((identity, extras), dim=1)
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.prelu2(self.bn2(x))
        x = self.prelu3(self.bn3(self.conv3(x)))
        if self.regularizer is not None: x = self.regularizer(x)
        if identity.size() != x.size():
            pad = (identity.size(3)-x.size(3),0,identity.size(2)-x.size(2),0)
            x = F.pad(x, pad, "constant", 0)
        x = self.prelu_out(x + identity)
        if self.downsample: return x, idx
        return x

class ENet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.initial      = InitialBlock(in_channels)
        self.bottleneck10 = BottleNeck(16, 64, downsample=True, p_drop=0.01)
        self.bottleneck11 = BottleNeck(64, p_drop=0.01); self.bottleneck12 = BottleNeck(64, p_drop=0.01)
        self.bottleneck13 = BottleNeck(64, p_drop=0.01); self.bottleneck14 = BottleNeck(64, p_drop=0.01)
        self.bottleneck20 = BottleNeck(64, 128, downsample=True, p_drop=0.1)
        self.bottleneck21 = BottleNeck(128, p_drop=0.1); self.bottleneck22 = BottleNeck(128, dilation=2, p_drop=0.1)
        self.bottleneck23 = BottleNeck(128, asymetric=True, p_drop=0.1); self.bottleneck24 = BottleNeck(128, dilation=4, p_drop=0.1)
        self.bottleneck25 = BottleNeck(128, p_drop=0.1); self.bottleneck26 = BottleNeck(128, dilation=8, p_drop=0.1)
        self.bottleneck27 = BottleNeck(128, asymetric=True, p_drop=0.1); self.bottleneck28 = BottleNeck(128, dilation=16, p_drop=0.1)
        self.bottleneck31 = BottleNeck(128, p_drop=0.1); self.bottleneck32 = BottleNeck(128, dilation=2, p_drop=0.1)
        self.bottleneck33 = BottleNeck(128, asymetric=True, p_drop=0.1); self.bottleneck34 = BottleNeck(128, dilation=4, p_drop=0.1)
        self.bottleneck35 = BottleNeck(128, p_drop=0.1); self.bottleneck36 = BottleNeck(128, dilation=8, p_drop=0.1)
        self.bottleneck37 = BottleNeck(128, asymetric=True, p_drop=0.1); self.bottleneck38 = BottleNeck(128, dilation=16, p_drop=0.1)
        self.bottleneck40 = BottleNeck(128, 64, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck41 = BottleNeck(64, p_drop=0.1, use_prelu=False); self.bottleneck42 = BottleNeck(64, p_drop=0.1, use_prelu=False)
        self.bottleneck50 = BottleNeck(64, 16, upsample=True, p_drop=0.1, use_prelu=False)
        self.bottleneck51 = BottleNeck(16, p_drop=0.1, use_prelu=False)
        self.fullconv = nn.ConvTranspose2d(16, num_classes, 3, padding=1, output_padding=1, stride=2, bias=False)
        initialize_weights(self)
    def forward(self, x):
        x = self.initial(x); sz1 = x.size()
        x, indices1 = self.bottleneck10(x)
        x = self.bottleneck11(x); x = self.bottleneck12(x); x = self.bottleneck13(x); x = self.bottleneck14(x)
        sz2 = x.size(); x, indices2 = self.bottleneck20(x)
        x = self.bottleneck21(x); x = self.bottleneck22(x); x = self.bottleneck23(x); x = self.bottleneck24(x)
        x = self.bottleneck25(x); x = self.bottleneck26(x); x = self.bottleneck27(x); x = self.bottleneck28(x)
        x = self.bottleneck31(x); x = self.bottleneck32(x); x = self.bottleneck33(x); x = self.bottleneck34(x)
        x = self.bottleneck35(x); x = self.bottleneck36(x); x = self.bottleneck37(x); x = self.bottleneck38(x)
        x = self.bottleneck40(x, indices=indices2, output_size=sz2)
        x = self.bottleneck41(x); x = self.bottleneck42(x)
        x = self.bottleneck50(x, indices=indices1, output_size=sz1)
        x = self.bottleneck51(x)
        return self.fullconv(x)

def check_speed_bump(pred_np):
    H, W = pred_np.shape
    row_on = int(H * SPEED_BUMP_ZONE_TOP); row_start = int(H * SPEED_BUMP_DETECTION_ROWS)
    if np.sum(pred_np[row_on:, :] == SPEED_BUMP_CLASS) >= SPEED_BUMP_MIN_PX:
        return 'on_bump'
    if np.sum(pred_np[row_start:row_on, :] == SPEED_BUMP_CLASS) >= SPEED_BUMP_MIN_PX:
        return 'approaching'
    return None

POTHOLE_CLASS  = 2
POTHOLE_MIN_PX = 20
def check_pothole(pred_np):
    H, W = pred_np.shape
    row_start = int(H * 0.5)
    if np.sum(pred_np[row_start:, :] == POTHOLE_CLASS) >= POTHOLE_MIN_PX:
        return 'pothole'
    return None


def _corridor_hazard_cols(row_classes, lo, hi):
    out = []
    if not DETECT_INTERIOR_HAZARD or hi - lo < HAZARD_MIN_RUN:
        return out
    inside = row_classes[lo:hi + 1]
    is_haz = ~np.isin(inside, DRIVABLE_CLASS_IDXS)
    run_start = None
    for k in range(len(is_haz) + 1):
        active = k < len(is_haz) and is_haz[k]
        if active and run_start is None:
            run_start = k
        elif not active and run_start is not None:
            width = k - run_start
            if width >= HAZARD_MIN_RUN:
                out.append((lo + run_start + width // 2, width))
            run_start = None
    return out

def mask_to_obstacles_global(pred_np, robot_state):
    x_robot, y_robot, yaw = robot_state[0], robot_state[1], robot_state[2]
    H, W = pred_np.shape
    left_pts, right_pts, hazard_pts = [], [], []
    row_start = int(H * BOUNDARY_ROWS_START)
    def to_global(col, row):
        x_local, y_local = pixel_to_ground(col, row)
        if x_local is None or not (OB_MIN_DIST < x_local < OB_MAX_DIST):
            return None
        ox = x_robot + x_local * np.cos(yaw) - y_local * np.sin(yaw)
        oy = y_robot + x_local * np.sin(yaw) + y_local * np.cos(yaw)
        return [ox, oy]
    for row in range(H - 1, row_start, -BOUNDARY_SAMPLE_STEP):
        drivable = np.where(np.isin(pred_np[row], DRIVABLE_CLASS_IDXS))[0]
        if len(drivable) < MIN_DRIVABLE_PX:
            continue
        lo, hi = int(drivable[0]), int(drivable[-1])
        edge_cols = []
        if lo > BORDER_MARGIN_PX:
            edge_cols.append((lo, left_pts))
        if hi < W - 1 - BORDER_MARGIN_PX:
            edge_cols.append((hi, right_pts))
        for col, bucket in edge_cols:
            p = to_global(col, row)
            if p is not None:
                bucket.append(p)
        for col, _w in _corridor_hazard_cols(pred_np[row], lo, hi):
            p = to_global(col, row)
            if p is not None:
                hazard_pts.append(p)
    def cap(pts, limit):
        arr = np.array(pts) if len(pts) else np.empty((0, 2))
        if len(arr) > limit:
            d = np.hypot(arr[:, 0] - x_robot, arr[:, 1] - y_robot)
            arr = arr[np.argsort(d)[:limit]]
        return arr
    half = MAX_OB_POINTS // 2
    left_arr = cap(left_pts, half)
    corridor = np.vstack([left_arr, cap(right_pts, half)])
    hazards = cap(hazard_pts, MAX_HAZARD_POINTS)
    return corridor, hazards, left_arr

PALETTE = [(128,64,128),(232,35,244),(70,70,70),(255,255,255)] 
def colorize_overlay(pred_np, resized_rgb):
    colored = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)
    for i, c in enumerate(PALETTE):
        colored[pred_np == i] = c
    overlay = cv2.addWeighted(resized_rgb, 0.55, colored, 0.45, 0)
    row_start = int(IMG_HEIGHT * BOUNDARY_ROWS_START)
    left_px, right_px = [], []        
    for row in range(IMG_HEIGHT - 1, row_start, -BOUNDARY_SAMPLE_STEP):
        drivable = np.where(np.isin(pred_np[row], DRIVABLE_CLASS_IDXS))[0]
        if len(drivable) < MIN_DRIVABLE_PX:
            left_px.append(None); right_px.append(None)
            continue
        lo, hi = int(drivable[0]), int(drivable[-1])
        # [FIX-BORDER] tepi yang menempel pinggir frame = bukan boundary asli -> putus
        left_px.append((lo, row) if lo > BORDER_MARGIN_PX else None)
        right_px.append((hi, row) if hi < IMG_WIDTH - 1 - BORDER_MARGIN_PX else None)
        for col, _w in _corridor_hazard_cols(pred_np[row], lo, hi):
            cv2.drawMarker(overlay, (col, row), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 9, 2)
    _draw_polyline(overlay, right_px, (80, 255, 120))
    _draw_left_and_keepleft(overlay, pred_np)
    # cv2.putText(overlay, "merah=tepi kiri  kuning=keep-left 0.75m  magenta=target heading PID",
    #             (8, IMG_HEIGHT - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
    #             (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _draw_polyline(overlay, pts, color):
    """Gambar polyline boundary (titik + garis penghubung), putus saat None."""
    prev = None
    for p in pts:
        if p is None:
            prev = None
            continue
        cv2.circle(overlay, p, 3, color, -1)
        if prev is not None:
            cv2.line(overlay, prev, p, color, 2)
        prev = p


def _draw_left_and_keepleft(overlay, pred_np):
    """[VIZ] Gambar tepi KIRI (merah) + garis keep-left 0.75 m (kuning) secara
    MULUS. Kumpulkan titik tepi kiri ASLI (bukan pinggir frame), fit kurva
    derajat-2 di bidang TANAH, lalu proyeksikan balik ke piksel. Menghilangkan
    zigzag akibat noise deteksi tepi per-baris."""
    if not USE_HOMOGRAPHY or H_matrix is None:
        return
    row_start = int(IMG_HEIGHT * BOUNDARY_ROWS_START)
    depan_arr, samp_arr, dirs = [], [], []
    for row in range(IMG_HEIGHT - 1, row_start, -BOUNDARY_SAMPLE_STEP):
        drivable = np.where(np.isin(pred_np[row], DRIVABLE_CLASS_IDXS))[0]
        if len(drivable) < MIN_DRIVABLE_PX:
            continue
        lo, hi = int(drivable[0]), int(drivable[-1])
        if lo <= BORDER_MARGIN_PX:            # tepi kiri tak terlihat -> lewati
            continue
        depan_l, samp_l = pixel_to_ground(lo, row)
        _dr, samp_r = pixel_to_ground(hi, row)
        if depan_l is None or samp_l is None or samp_r is None:
            continue
        depan_arr.append(depan_l); samp_arr.append(samp_l)
        dirs.append(1.0 if samp_r > samp_l else -1.0)
    if len(depan_arr) < 3:                     # data terlalu sedikit -> jangan gambar
        return
    depan_arr = np.asarray(depan_arr, float)
    samp_arr  = np.asarray(samp_arr, float)
    direction = 1.0 if np.mean(dirs) >= 0 else -1.0
    order = np.argsort(depan_arr)
    d_s, s_s = depan_arr[order], samp_arr[order]
    deg = 2 if len(d_s) >= 5 else 1
    try:
        fit = np.poly1d(np.polyfit(d_s, s_s, deg))
    except Exception:
        return
    d_lin = np.linspace(d_s.min(), d_s.max(), 40)

    def _project(lateral_offset):
        out = []
        for d in d_lin:
            px = ground_to_pixel(d, float(fit(d)) + lateral_offset * direction)
            if px is None:
                out.append(None); continue
            u, v = px
            out.append((u, v) if (0 <= u < IMG_WIDTH and 0 <= v < IMG_HEIGHT) else None)
        return out

    _draw_polyline(overlay, _project(0.0),               (255, 80, 80))   # tepi kiri (merah, mulus)
    _draw_polyline(overlay, _project(KEEPLEFT_OFFSET_M), (255, 255, 0))   # keep-left (kuning)


def _draw_keep_left_line(overlay, pred_np):
    """[VIZ][DEPRECATED, tak dipakai] Gambar garis target keep-left (KUNING): sejajar tepi KIRI jalan,
    digeser KEEPLEFT_OFFSET_M meter ke arah DALAM jalan. Inilah jalur lateral
    yang ingin dijaga robot (cost #4 keep-left di DWA)."""
    if not SHOW_KEEPLEFT_LINE or not USE_HOMOGRAPHY or H_matrix is None:
        return
    row_start = int(IMG_HEIGHT * BOUNDARY_ROWS_START)
    pts = []
    for row in range(IMG_HEIGHT - 1, row_start, -BOUNDARY_SAMPLE_STEP):
        drivable = np.where(np.isin(pred_np[row], DRIVABLE_CLASS_IDXS))[0]
        if len(drivable) < MIN_DRIVABLE_PX:
            pts.append(None)            # putus garis bila baris tak ada jalan
            continue
        lo, hi = int(drivable[0]), int(drivable[-1])
        if lo <= BORDER_MARGIN_PX:
            pts.append(None)
            continue
        depan_l, samp_l = pixel_to_ground(lo, row)
        _depan_r, samp_r = pixel_to_ground(hi, row)
        if depan_l is None or samp_l is None or samp_r is None:
            pts.append(None)
            continue
        direction   = 1.0 if samp_r > samp_l else -1.0
        samp_target = samp_l + KEEPLEFT_OFFSET_M * direction
        px = ground_to_pixel(depan_l, samp_target)
        if px is None:
            pts.append(None)
            continue
        u, v = px
        pts.append((u, v) if (0 <= u < IMG_WIDTH and 0 <= v < IMG_HEIGHT) else None)
    prev = None
    for p in pts:
        if p is None:
            prev = None
            continue
        cv2.circle(overlay, p, 2, (255, 255, 0), -1)       
        if prev is not None:
            cv2.line(overlay, prev, p, (255, 255, 0), 2)
        prev = p
    cv2.putText(overlay, f"keep-left {KEEPLEFT_OFFSET_M:.2f}m (kuning)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

_ort_sess = None
_ort_input_name = None
if INFER_BACKEND == "onnx":
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = _nthreads
    _ort_sess = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=_so, providers=["CPUExecutionProvider"])
    _ort_input_name = _ort_sess.get_inputs()[0].name
    model = None
    print("[ENet] ONNX model loaded:", ONNX_MODEL_PATH, "| threads =", _nthreads)
else:
    model = ENet(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        key = "model_state_dict" if "model_state_dict" in ckpt else ("state_dict" if "state_dict" in ckpt else None)
        model.load_state_dict(ckpt[key] if key else ckpt)
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("[ENet] Model loaded (torch).")

import threading, time

_lock = threading.Lock()
_win_ready = [False]   
_latest_corridor = np.empty((0, 2))
_latest_hazard = np.empty((0, 2))
_latest_left_edge = np.empty((0, 2))
_latest_keepleft = np.empty((0, 2))       
_latest_target_heading = None           
_latest_heading_err = None               
_latest_heading_raw = None                
_latest_frame_id = 0
_latest_bump = None
_latest_pothole = None                    
_latest_fps = 0.0                         
_running = False
_thread = None
_cap = None
_latest_disp = None  

def _capture_loop():
    global _latest_corridor, _latest_hazard, _latest_left_edge, _latest_keepleft, _latest_bump, _running, _cap, _latest_frame_id, _latest_disp, _latest_pothole, _latest_fps
    use_video = bool(VIDEO_SOURCE)
    if use_video:
        _cap = cv2.VideoCapture(VIDEO_SOURCE)
        src_label = 'VIDEO ' + str(VIDEO_SOURCE)
    else:
        _cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        src_label = 'webcam index ' + str(CAM_INDEX)
    if not _cap.isOpened():
        print('[ENet-bridge] ERROR: sumber tidak terbuka ->', src_label)
        _running = False
        return
    if not use_video:
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_REQ_WIDTH)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_REQ_HEIGHT)
    else:
        _vid_total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        _vid_fps   = _cap.get(cv2.CAP_PROP_FPS) or 0.0
        print('[ENet-bridge] Info video: total_frame=%d fps_asli=%.1f' % (_vid_total, _vid_fps))
    print('[ENet-bridge] Sumber aktif: %s - inferensi berjalan...' % src_label)
    frame_id = 0
    zero_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    _fps_val = 0.0
    _fps_prev = None
    while _running:
        ret, frame = _cap.read()
        if not ret:
            if use_video:
                if VIDEO_LOOP:
                    _cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # ulang dari awal
                    continue
                print('[ENet-bridge] Video selesai.')
                break
            time.sleep(0.01)
            continue
        if use_video and VIDEO_FPS_LIMIT > 0:
            time.sleep(1.0 / VIDEO_FPS_LIMIT)
        frame_id += 1
        if frame_id % FRAME_PROCESS_EVERY != 0:
            continue
        frame = maybe_undistort(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (IMG_WIDTH, IMG_HEIGHT))
        if INFER_BACKEND == "onnx":
            inp = np.ascontiguousarray((resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...])
            logits = _ort_sess.run(None, {_ort_input_name: inp})[0]
            pred_np = logits[0].argmax(axis=0).astype(np.uint8)
        else:
            inp = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_np = model(inp).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        bump = check_speed_bump(pred_np)
        pothole = check_pothole(pred_np)   # [REC] status pothole utk logging CSV
        # [FPS] hitung laju inferensi (EMA) -> ditampilkan di preview
        _t_now = time.time()
        if _fps_prev is not None:
            _inst = 1.0 / max(1e-6, _t_now - _fps_prev)
            _fps_val = _inst if _fps_val == 0.0 else 0.9 * _fps_val + 0.1 * _inst
        _fps_prev = _t_now

        corridor, hazards, left_edge = mask_to_obstacles_global(pred_np, zero_pose)
        keepleft = _compute_keepleft_target(pred_np)  
        with _lock:
            _latest_corridor = corridor
            _latest_hazard = hazards
            _latest_left_edge = left_edge
            _latest_keepleft = keepleft
            _latest_bump = bump
            _latest_pothole = pothole
            _latest_fps = _fps_val
            _latest_frame_id = frame_id
        if SHOW_WINDOW:
            try:
                overlay = colorize_overlay(pred_np, resized)
                disp = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                _draw_predicted_path(disp)  
                _draw_target_heading(disp)  
                if PREVIEW_SCALE and PREVIEW_SCALE != 1.0:
                    disp = cv2.resize(
                        disp,
                        (int(IMG_WIDTH * PREVIEW_SCALE), int(IMG_HEIGHT * PREVIEW_SCALE)),
                        interpolation=cv2.INTER_NEAREST,
                    )
                _label = 'frame ' + str(frame_id)
                cv2.putText(disp, _label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(disp, _label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2, cv2.LINE_AA)
                _fps_txt = 'FPS %.1f' % _fps_val
                cv2.putText(disp, _fps_txt, (10, 54), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(disp, _fps_txt, (10, 54), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)
                # [PID] HUD nilai target heading & heading error (derajat)
                if _latest_target_heading is not None:
                    _hud = [
                        'target heading: %+6.1f deg' % np.degrees(_latest_target_heading),
                    ]
                else:
                    _hud = ['target heading: -- (tali hilang)']
                _yy = 84
                for _ln in _hud:
                    cv2.putText(disp, _ln, (10, _yy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(disp, _ln, (10, _yy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 0, 255), 2, cv2.LINE_AA)
                    _yy += 26
                with _lock:
                    _latest_disp = disp
            except Exception:
                pass
    if _cap is not None:
        _cap.release()
    print('[ENet-bridge] Webcam dilepas.')

def start():
    global _running, _thread
    if _running:
        return
    _running = True
    _thread = threading.Thread(target=_capture_loop, daemon=True)
    _thread.start()


def render_preview():
    """Tampilkan frame preview terakhir. WAJIB dipanggil dari MAIN THREAD (loop utama).
    OpenCV HighGUI di Windows hanya andal bila namedWindow/imshow/waitKey jalan di
    thread utama; karena itu pembuatan window dipisah dari thread kamera."""
    if not SHOW_WINDOW:
        return
    with _lock:
        disp = _latest_disp
    if disp is None:
        return
    if not _win_ready[0]:
        flag = cv2.WINDOW_NORMAL if PREVIEW_RESIZABLE else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(PREVIEW_WINDOW, flag)
        if PREVIEW_RESIZABLE:
            cv2.resizeWindow(PREVIEW_WINDOW,
                             int(IMG_WIDTH * PREVIEW_SCALE), int(IMG_HEIGHT * PREVIEW_SCALE))
        _win_ready[0] = True
    cv2.imshow(PREVIEW_WINDOW, disp)
    cv2.waitKey(1)


def close_preview():
    """Tutup window preview. Panggil dari MAIN THREAD saat shutdown."""
    try:
        cv2.destroyWindow(PREVIEW_WINDOW)
        cv2.waitKey(1)
    except Exception:
        pass

def get_obstacles():
    with _lock:
        return _latest_corridor.copy(), _latest_hazard.copy()

def get_bump():
    with _lock:
        return _latest_bump

def get_pothole():
    with _lock:
        return _latest_pothole

def get_fps():
    with _lock:
        return float(_latest_fps)

def get_status():
    with _lock:
        return {
            "bump":    _latest_bump,
            "pothole": _latest_pothole,
            "hazard":  int(len(_latest_hazard)),
            "frame":   int(_latest_frame_id),
            "fps":     round(float(_latest_fps), 2),
        }

def get_left_edge():
    with _lock:
        return _latest_left_edge.copy()

def stop():
    global _running
    _running = False


# =========================================================================
# OVERLAY LINTASAN PREDIKSI DWA PADA PREVIEW KAMERA
# Dipanggil argo_local_navtest.py: set_predicted_path([[depan,samping],...])
# (depan,samping = meter, frame robot; konvensi sama dgn pixel_to_ground)
# =========================================================================
_predicted_path = np.empty((0, 2))
_pred_dbg = 0
_H_inv_cache = None


def set_predicted_path(local_pts):
    """Terima Nx2 [depan, samping] (meter, frame robot) utk digambar di preview."""
    global _predicted_path
    if local_pts is None or len(local_pts) == 0:
        arr = np.empty((0, 2))
    else:
        arr = np.asarray(local_pts, dtype=float)
    with _lock:
        _predicted_path = arr


def ground_to_pixel(depan, samping):
    """(depan,samping) meter -> (u,v) piksel MASK. None jika gagal."""
    global _H_inv_cache
    if not USE_HOMOGRAPHY or H_matrix is None:
        return None
    if _H_inv_cache is None:
        try:
            _H_inv_cache = np.linalg.inv(H_matrix)
        except Exception:
            return None
    world = np.array([samping, depan, 1.0])   # world_points = [samping, depan]
    p = _H_inv_cache @ world
    if abs(p[2]) < 1e-9:
        return None
    p = p / p[2]
    u = p[0] / SCALE_U
    v = p[1] / SCALE_V
    return int(round(u)), int(round(v))


def _draw_predicted_path(disp):
    """Gambar polyline lintasan prediksi DWA pada citra preview (BGR)."""
    global _pred_dbg
    with _lock:
        path = _predicted_path.copy()
    if len(path) == 0:
        if _pred_dbg < 3:
            print('[ENet] pred path KOSONG (set_predicted_path belum dipanggil?)')
            _pred_dbg += 1
        return
    prev = None
    drawn = 0
    for depan, samping in path:
        if depan <= 0:
            prev = None
            continue
        px = ground_to_pixel(depan, samping)
        if px is None:
            prev = None
            continue
        u, v = px
        if 0 <= u < IMG_WIDTH and 0 <= v < IMG_HEIGHT:
            cv2.circle(disp, (u, v), 3, (255, 255, 0), -1)
            if prev is not None:
                cv2.line(disp, prev, (u, v), (255, 255, 0), 2)
            prev = (u, v)
            drawn += 1
        else:
            prev = None
    if _pred_dbg < 5:
        print('[ENet] pred path:', len(path), 'titik diterima,', drawn, 'tampil di citra')
        _pred_dbg += 1


def get_frame_id():
    """Nomor frame webcam terakhir yang diproses (untuk keterangan/log)."""
    with _lock:
        return _latest_frame_id


# =========================================================================
def _compute_keepleft_target(pred_np):
    """Nx2 [depan, samping] (m, FRAME ROBOT) garis keep-left (TALI).
    Logika sama dgn _draw_left_and_keepleft: fit derajat-2 tepi KIRI di bidang
    tanah lalu geser KEEPLEFT_OFFSET_M ke DALAM jalan. Kosong bila data kurang."""
    if not USE_HOMOGRAPHY or H_matrix is None:
        return np.empty((0, 2))
    row_start = int(IMG_HEIGHT * BOUNDARY_ROWS_START)
    depan_arr, samp_arr, dirs = [], [], []
    for row in range(IMG_HEIGHT - 1, row_start, -BOUNDARY_SAMPLE_STEP):
        drivable = np.where(np.isin(pred_np[row], DRIVABLE_CLASS_IDXS))[0]
        if len(drivable) < MIN_DRIVABLE_PX:
            continue
        lo, hi = int(drivable[0]), int(drivable[-1])
        if lo <= BORDER_MARGIN_PX:          
            continue
        depan_l, samp_l = pixel_to_ground(lo, row)
        _dr, samp_r = pixel_to_ground(hi, row)
        if depan_l is None or samp_l is None or samp_r is None:
            continue
        depan_arr.append(depan_l); samp_arr.append(samp_l)
        dirs.append(1.0 if samp_r > samp_l else -1.0)
    if len(depan_arr) < 3:
        return np.empty((0, 2))
    depan_arr = np.asarray(depan_arr, float)
    samp_arr  = np.asarray(samp_arr, float)
    direction = 1.0 if np.mean(dirs) >= 0 else -1.0
    order = np.argsort(depan_arr)
    d_s, s_s = depan_arr[order], samp_arr[order]
    deg = 2 if len(d_s) >= 5 else 1
    try:
        fit = np.poly1d(np.polyfit(d_s, s_s, deg))
    except Exception:
        return np.empty((0, 2))
    d_lin = np.linspace(d_s.min(), d_s.max(), 25)
    out = [[float(d), float(fit(d)) + KEEPLEFT_OFFSET_M * direction] for d in d_lin]
    return np.asarray(out, float)


def get_keepleft_target():
    """Nx2 [depan, samping] (m, FRAME ROBOT) garis keep-left untuk diikuti PID."""
    with _lock:
        return _latest_keepleft.copy()


def set_target_heading(th, err=None, raw=None):
    """[PID] Simpan target heading + heading error (rad, frame robot) utk preview.
    th  = setpoint heading (panah MAGENTA), err = error PID, raw = arah tali sesaat.
    th=None -> panah & HUD disembunyikan (mis. tali hilang)."""
    global _latest_target_heading, _latest_heading_err, _latest_heading_raw
    _latest_target_heading = None if th is None else float(th)
    _latest_heading_err = None if err is None else float(err)
    _latest_heading_raw = None if raw is None else float(raw)


def _draw_target_heading(disp):
    """[VIZ] Panah MAGENTA dari posisi robot menuju target heading PID saat ini.
    Digambar pada citra 512x256 (koordinat tanah valid via ground_to_pixel)."""
    th = _latest_target_heading
    if th is None:
        return
    L = 1.5
    p0 = ground_to_pixel(0.25, 0.0)
    p1 = ground_to_pixel(L * float(np.cos(th)), L * float(np.sin(th)))
    if p0 is None or p1 is None:
        return
    try:
        cv2.arrowedLine(disp, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                        (255, 0, 255), 3, cv2.LINE_AA, tipLength=0.25)
        # cv2.putText(disp, 'target heading (magenta)', (8, IMG_HEIGHT - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    except Exception:
        pass
