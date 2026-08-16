# -*- coding: utf-8 -*-
"""
================================================================
 Analisis Navigasi Robot Mobile Otonom
 Robot Navigation Performance Analysis - Single Scenario
================================================================
 Output : folder  Gambar Pengujian 2/
          berisi  01_lintasan_robot.png
                  02_tracking_kecepatan_linear.png
                  03_tracking_kecepatan_angular.png
                  04_deviasi_lateral.png
                  05_jarak_obstacle.png
                  06_ringkasan_metrik.png
================================================================
"""

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from scipy.ndimage import uniform_filter1d

# -- Windows console fix --
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# =============================================================
#  KONFIGURASI  -- sesuaikan di sini
# =============================================================
CSV_FILE = "Pengujian_skenario3_2.csv"
GOAL     = (10.0, 0.0)
SCENARIO = "Skenario 3 - 6 Obstacle"
DPI      = 200

# Konfigurasi obstacle — ukuran fisik 25 x 25 cm = 0.25 x 0.25 m
OBSTACLES = [
    {"pos": (2.7,  0.8),  "radius": 0.25, "label": "Obstacle 1"},
    {"pos": (2.7, -0.6),  "radius": 0.25, "label": "Obstacle 2"},
    {"pos": (4.8,  0.3), "radius": 0.25, "label": "Obstacle 3"},
    {"pos": (4.8, -1.1),  "radius": 0.25, "label": "Obstacle 4"},
    {"pos": (6.5, 0.8),  "radius": 0.25, "label": "Obstacle 5"},
    {"pos": (6.5, -0.6), "radius": 0.25, "label": "Obstacle 6"},
]

# Path otomatis relatif terhadap lokasi script
_DIR       = os.path.dirname(os.path.abspath(__file__))
CSV_FILE   = os.path.join(_DIR, CSV_FILE)
OUTPUT_DIR = os.path.join(_DIR, "Gambar Pengujian 2")
# =============================================================

# -- Palet warna profesional --
C_ACTUAL   = "#0A3D62"   # navy        - data terukur
C_CMD      = "#E55039"   # merah       - referensi / cmd
C_FILL     = "#AED6F1"   # biru muda   - area fill
C_GOAL     = "#1E8449"   # hijau       - goal / jalur ideal
C_GRID     = "#DFE6E9"
C_BG       = "#FFFFFF"
C_FIG      = "#F0F3F4"
C_TEXT     = "#1C2833"
C_ACCENT   = "#F39C12"   # oranye      - aksen highlight
C_OBSTACLE = "#C0392B"   # merah tua   - obstacle

# -- Global rcParams --
plt.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            10,
    "axes.facecolor":       C_BG,
    "figure.facecolor":     C_FIG,
    "axes.edgecolor":       "#BDC3C7",
    "axes.linewidth":       0.8,
    "axes.grid":            True,
    "grid.color":           C_GRID,
    "grid.linewidth":       0.55,
    "grid.alpha":           1.0,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.labelsize":       11,
    "axes.labelcolor":      C_TEXT,
    "axes.labelweight":     "bold",
    "xtick.labelsize":      9.5,
    "ytick.labelsize":      9.5,
    "xtick.color":          C_TEXT,
    "ytick.color":          C_TEXT,
    "legend.fontsize":      9,
    "legend.framealpha":    0.95,
    "legend.edgecolor":     "#BDC3C7",
    "legend.fancybox":      False,
    "lines.linewidth":      2.0,
    "lines.solid_capstyle": "round",
    "savefig.bbox":         "tight",
    "savefig.facecolor":    C_FIG,
    "savefig.dpi":          DPI,
})


def _add_header(fig, title, subtitle=""):
    """Tambah judul bergaya paper dengan garis separator biru."""
    y0 = 0.97 if subtitle else 0.96
    fig.text(0.5, y0, title,
             ha="center", va="top", fontsize=14, fontweight="bold",
             color=C_TEXT, transform=fig.transFigure)
    if subtitle:
        fig.text(0.5, y0 - 0.048, subtitle,
                 ha="center", va="top", fontsize=9.5, color="#566573",
                 transform=fig.transFigure)
    line_y = y0 - (0.08 if subtitle else 0.06)
    fig.add_artist(plt.Line2D([0.05, 0.95], [line_y, line_y],
                               transform=fig.transFigure,
                               color="#2E86C1", linewidth=1.4, alpha=0.65))


def _save(fig, path, label):
    """Simpan dan tutup figure."""
    fig.text(0.98, 0.012, label, ha="right", va="bottom",
             fontsize=7.5, color="#AAB7B8", transform=fig.transFigure)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(path)}")


# =============================================================
#  LOAD & PREPROCESSING
# =============================================================

def load_data(path):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] File tidak ditemukan: {path}")
    df = pd.read_csv(path)
    df["t"]           = df["travel_time"] - df["travel_time"].iloc[0]
    df["dist_goal"]   = np.sqrt((df["x"] - GOAL[0])**2 + (df["y"] - GOAL[1])**2)
    df["v_error"]     = np.abs(df["v_cmd"]     - df["v_actual"])
    df["omega_error"] = np.abs(df["omega_cmd"] - df["omega_actual"])
    df["v_smooth"]    = uniform_filter1d(df["v_actual"],     size=7)
    df["omega_smooth"]= uniform_filter1d(df["omega_actual"], size=7)
    return df


def compute_metrics(df):
    dx, dy   = np.diff(df["x"].values), np.diff(df["y"].values)
    path_len = float(np.sum(np.sqrt(dx**2 + dy**2)))
    obs      = df["min_obstacle_dist"][df["min_obstacle_dist"] > 0]

    # Jarak minimum obstacle
    jarak_min_obs = round(float(obs.min()), 4) if len(obs) > 0 else "N/A"

    # ---------------------------------------------------------------
    # Waktu respon motor (rise time 90%): waktu dari v_cmd > 0
    # pertama hingga v_actual >= 90% v_cmd
    # ---------------------------------------------------------------
    v_cmd_nonzero = df[df["v_cmd"] > 0]
    if len(v_cmd_nonzero) > 0:
        t_cmd_start = v_cmd_nonzero["t"].iloc[0]
        v_target    = v_cmd_nonzero["v_cmd"].iloc[0] * 0.90
        reached     = df[(df["t"] >= t_cmd_start) & (df["v_actual"] >= v_target)]
        waktu_respon = round(float(reached["t"].iloc[0] - t_cmd_start), 4) if len(reached) > 0 else "N/A"
    else:
        waktu_respon = "N/A"

    # ---------------------------------------------------------------
    # Respon sistem terhadap obstacle
    # Definisi: waktu dari deteksi obstacle pertama kali (min_obstacle_dist
    # turun di bawah threshold_detect) hingga omega_cmd berubah signifikan
    # (|delta omega_cmd| >= threshold_steer), menandakan sistem mulai manuver.
    # ---------------------------------------------------------------
    THRESHOLD_DETECT = 2.0   # obstacle dianggap "terdeteksi" jika jarak <= 2 m
    THRESHOLD_STEER  = 0.01  # perubahan omega_cmd >= 0.01 rad/s = manuver dimulai

    respon_obstacle  = "N/A"
    t_deteksi        = "N/A"
    t_manuver        = "N/A"

    obs_detected = df[df["min_obstacle_dist"].between(0, THRESHOLD_DETECT)]
    if len(obs_detected) > 0:
        t_deteksi = round(float(obs_detected["t"].iloc[0]), 4)

        # Cari titik setelah deteksi di mana omega_cmd berubah signifikan
        df_after = df[df["t"] >= obs_detected["t"].iloc[0]].copy()
        domega   = df_after["omega_cmd"].diff().abs()
        steered  = df_after[domega >= THRESHOLD_STEER]

        if len(steered) > 0:
            t_manuver       = round(float(steered["t"].iloc[0]), 4)
            respon_obstacle = round(t_manuver - t_deteksi, 4)

    # ---------------------------------------------------------------
    # Efisiensi lintasan (DIREVISI)
    # Formula lama: dist(start, posisi_akhir) / path_len  → selalu ~1 meski
    # robot belum sampai goal (tidak sensitif terhadap keberhasilan navigasi).
    #
    # Formula baru: proyeksi posisi akhir ke arah start→goal / path_len
    #   • Pembilang = komponen kemajuan robot ke arah goal (progress efektif)
    #   • Penyebut  = panjang lintasan aktual yang ditempuh
    #   • Nilai mendekati 1.0 = robot bergerak hampir lurus ke goal (efisien)
    #   • Nilai < 1 = ada deviasi / detour akibat manuver obstacle
    #   • Penalti otomatis jika robot belum sampai goal atau berputar balik
    #
    # Progress metrik terpisah: completion ratio (% jarak ke goal yang dicapai)
    # ---------------------------------------------------------------
    x0, y0   = df["x"].iloc[0],  df["y"].iloc[0]
    x_end, y_end = df["x"].iloc[-1], df["y"].iloc[-1]
    gx, gy   = GOAL[0] - x0, GOAL[1] - y0
    dist_ideal   = float(np.sqrt(gx**2 + gy**2))
    # Proyeksi vektor (posisi_akhir - start) ke arah unit vektor start→goal
    proj_progress = float(((x_end - x0) * gx + (y_end - y0) * gy) / dist_ideal)
    efisiensi     = round(proj_progress / path_len, 4) if path_len > 0 else "N/A"
    completion    = round(proj_progress / dist_ideal, 4) if dist_ideal > 0 else "N/A"

    return {
        "Durasi Navigasi (s)":              round(df["t"].iloc[-1], 3),
        "Panjang Lintasan (m)":             round(path_len, 4),
        "Kecepatan Rata-rata (m/s)":        round(df["v_actual"].mean(), 4),
        "Kecepatan Maksimum (m/s)":         round(df["v_actual"].max(), 4),
        "Deviasi Lateral Rata-rata (m)":    round(df["y"].abs().mean(), 4),
        "Deviasi Lateral Maksimum (m)":     round(df["y"].abs().max(), 4),
        "Jarak Akhir ke Goal (m)":          round(df["dist_goal"].iloc[-1], 4),
        "Error v Rata-rata (m/s)":          round(df["v_error"].mean(), 4),
        "Error v Maksimum (m/s)":           round(df["v_error"].max(), 4),
        "Error omega Rata-rata (rad/s)":    round(df["omega_error"].mean(), 4),
        "Min. Clearance Obstacle (m)":      round(float(obs.min()),  4) if len(obs) > 0 else "N/A",
        "Avg. Clearance Obstacle (m)":      round(float(obs.mean()), 4) if len(obs) > 0 else "N/A",
        "Jarak Minimum ke Obstacle (m)":    jarak_min_obs,
        "Waktu Respon Motor (s)":           waktu_respon,
        "t Deteksi Obstacle (s)":           t_deteksi,
        "t Manuver Dimulai (s)":            t_manuver,
        "Respon Sistem → Obstacle (s)":     respon_obstacle,
        "Completion Ratio (0-1)":           completion,
        "Efisiensi Lintasan (0-1) [rev]":   efisiensi,
    }


# =============================================================
#  HELPER — BULAN SABIT
# =============================================================

def _draw_crescent(ax, cx, cy, r_outer=0.55, r_inner=0.38,
                   offset_x=0.18, color="#E67E22", alpha=0.92, zorder=8):
    """Gambar simbol bulan sabit menggunakan Path (filled outer - inner circle)."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    theta = np.linspace(0, 2 * np.pi, 360)

    # Lingkaran luar (bulan penuh)
    xo = cx + r_outer * np.cos(theta)
    yo = cy + r_outer * np.sin(theta)

    # Lingkaran dalam (bayangan — digeser ke kanan agar membentuk sabit)
    xi = cx + offset_x + r_inner * np.cos(theta[::-1])
    yi = cy             + r_inner * np.sin(theta[::-1])

    # Gabungkan menjadi satu path tertutup (outer CCW + inner CW = even-odd fill)
    verts = (
        list(zip(xo, yo)) + [(xo[0], yo[0])] +
        list(zip(xi, yi)) + [(xi[0], yi[0])]
    )
    codes = (
        [Path.MOVETO] + [Path.LINETO] * (len(xo) - 1) + [Path.CLOSEPOLY] +
        [Path.MOVETO] + [Path.LINETO] * (len(xi) - 1) + [Path.CLOSEPOLY]
    )
    path  = Path(verts, codes)
    patch = PathPatch(path, facecolor=color, edgecolor="white",
                      linewidth=0.8, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)


# =============================================================
#  FIGURE 1 — LINTASAN ROBOT + DEVIASI LATERAL (satu gambar, satu grafik)
# =============================================================

def fig_trajectory(df, out):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.09, right=0.88)
    _add_header(fig,
                "Lintasan Robot & Deviasi Lateral",
                f"{SCENARIO}  |  Goal: {GOAL}  |  Sampel: {len(df)}")

    # -- Lintasan berwarna berdasarkan kecepatan --
    cmap = LinearSegmentedColormap.from_list(
        "nav", ["#AED6F1", C_ACTUAL, C_ACCENT], N=256)
    sc = ax.scatter(df["x"], df["y"],
                    c=df["v_actual"], cmap=cmap,
                    s=14, zorder=4, linewidths=0,
                    vmin=0, vmax=df["v_actual"].max())

    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85, aspect=22)
    cb.set_label("Kecepatan Aktual $v$ (m/s)", fontsize=9.5, fontweight="bold")
    cb.ax.tick_params(labelsize=8.5)
    cb.outline.set_edgecolor("#BDC3C7")

    # -- Deviasi lateral (area fill terhadap y=0) --
    ax.fill_between(df["x"], 0, df["y"],
                    where=(df["y"] >= 0),
                    alpha=0.12, color=C_ACTUAL, label="Deviasi positif (y > 0)")
    ax.fill_between(df["x"], 0, df["y"],
                    where=(df["y"] < 0),
                    alpha=0.12, color=C_CMD,    label="Deviasi negatif (y < 0)")

    # -- Anotasi deviasi maksimum --
    idx_max = df["y"].abs().idxmax()
    yval    = df["y"].iloc[idx_max]
    xval    = df["x"].iloc[idx_max]
    offset  = 1.0 if yval > 0 else -1.0
    ax.annotate(f"Maks. deviasi\n{yval:.4f} m",
                xy=(xval, yval),
                xytext=(xval - 2.0, yval + offset),
                fontsize=8, color=C_TEXT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C_ACCENT, alpha=0.9))

    # -- Info box deviasi --
    ax.text(0.01, 0.04,
            f"Dev. rata-rata = {df['y'].abs().mean():.4f} m\n"
            f"Dev. maks.     = {df['y'].abs().max():.4f} m\n"
            f"Std. Dev       = {df['y'].std():.4f} m",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#BDC3C7", alpha=0.92))

    # ----------------------------------------------------------------
    #  VISUALISASI OBSTACLE — lingkaran radius 0.25 m (bulat sempurna)
    #  Obstacle GANJIL (1, 3, 5): label di ATAS obstacle
    #  Obstacle GENAP  (2, 4, 6): label di BAWAH obstacle  ← REVISI
    # ----------------------------------------------------------------
    for obs_cfg in OBSTACLES:
        ox, oy = obs_cfg["pos"]
        r      = obs_cfg["radius"]
        label  = obs_cfg["label"]

        # Lingkaran filled + outline
        ax.add_patch(plt.Circle(
            (ox, oy), r, color=C_OBSTACLE, alpha=0.45, zorder=5, linewidth=0))
        ax.add_patch(plt.Circle(
            (ox, oy), r, fill=False, edgecolor=C_OBSTACLE,
            linewidth=2.0, linestyle="-", zorder=6))

        # Tentukan nomor obstacle untuk membedakan genap/ganjil
        obs_number = int(label.split()[-1])
        is_even    = (obs_number % 2 == 0)

        if obs_number == 1:
            # Obstacle 1: label di KIRI obstacle agar tidak menghalangi legend
            xy_tip  = (ox - r, oy)           # ujung panah → tepi kiri lingkaran
            xytext  = (ox - 1.8, oy - 0.55)  # kotak teks di kiri bawah
        elif is_even:
            # Label di BAWAH obstacle
            xy_tip  = (ox, oy - r)          # ujung panah → tepi bawah lingkaran
            xytext  = (ox - 1.2, oy - 0.9)  # posisi kotak teks di bawah
        else:
            # Label di ATAS obstacle (perilaku semula)
            xy_tip  = (ox, oy + r)           # ujung panah → tepi atas lingkaran
            xytext  = (ox - 1.2, oy + 0.5)  # posisi kotak teks di atas

        ax.annotate(
            f"{label}\n({ox}, {oy})",
            xy=xy_tip,
            xytext=xytext,
            fontsize=8, color=C_OBSTACLE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_OBSTACLE, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_OBSTACLE, alpha=0.88),
            zorder=8)

        # Patch dummy untuk legenda (radius=0, tidak terlihat di plot)
        ax.add_patch(plt.Circle(
            (0, 0), 0, color=C_OBSTACLE, alpha=0.45,
            label=f"{label} {obs_cfg['pos']}"))
    # ----------------------------------------------------------------

    ax.scatter(df["x"].iloc[0], df["y"].iloc[0],
               s=140, color=C_GOAL, zorder=7, marker="o",
               edgecolors="white", linewidths=1.5, label="Start (0, 0)")
    ax.scatter(*GOAL, s=260, color=C_CMD, zorder=7, marker="*",
               edgecolors="white", linewidths=1, label=f"Goal {GOAL}")
    ax.annotate("Start", xy=(df["x"].iloc[0], df["y"].iloc[0]),
                xytext=(0.4, 1.2), fontsize=9, color=C_GOAL, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_GOAL, lw=0.8))
    ax.annotate("Goal", xy=GOAL,
                xytext=(GOAL[0] - 1.2, 1.2), fontsize=9, color=C_CMD, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_CMD, lw=0.8))
    ax.axhline(0, color=C_GOAL, lw=1.2, ls="--", alpha=0.55,
               label="Jalur ideal (y = 0)")

    ax.set_xlim(-0.5, GOAL[0] + 0.8)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Posisi X (m)")
    ax.set_ylabel("Posisi Y (m)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    _save(fig, out, "Fig. 1 - Lintasan Robot & Deviasi Lateral")


def fig_lateral(df, out):
    """Stub — sudah digabung ke fig_trajectory. Tidak menghasilkan file."""
    pass


# =============================================================
#  FIGURE 2 — TRACKING KECEPATAN LINEAR & ANGULAR (satu gambar, dua grafik)
# =============================================================

def fig_velocity(df, out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.10, right=0.96, hspace=0.12)
    _add_header(fig,
                "Tracking Kecepatan Linear & Angular",
                f"{SCENARIO}  |  Referensi vs Terukur")

    # -- Subplot atas: Kecepatan Linear --
    ax1.plot(df["x"], df["v_cmd"],                          # <-- ganti df["t"] → df["x"]
             color=C_CMD, lw=1.4, ls="--", alpha=0.85,
             label="$v_{cmd}$ (Referensi)", zorder=3)
    ax1.plot(df["x"], df["v_smooth"],                       # <-- ganti df["t"] → df["x"]
             color=C_ACTUAL, lw=2.2,
             label="$v_{actual}$ (Terukur)", zorder=4)
    ax1.fill_between(df["x"], df["v_cmd"], df["v_smooth"],  # <-- ganti df["t"] → df["x"]
                     where=(df["v_smooth"] < df["v_cmd"]),
                     alpha=0.18, color=C_CMD, label="Deviasi (under)")
    ax1.fill_between(df["x"], df["v_cmd"], df["v_smooth"],  # <-- ganti df["t"] → df["x"]
                     where=(df["v_smooth"] >= df["v_cmd"]),
                     alpha=0.18, color=C_ACTUAL, label="Deviasi (over)")

    rmse_v = np.sqrt(np.mean(df["v_error"]**2))
    ax1.text(0.98, 0.96,
             f"RMSE       = {rmse_v:.4f} m/s\n"
             f"Mean error = {df['v_error'].mean():.4f} m/s\n"
             f"Max error  = {df['v_error'].max():.4f} m/s",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8.5,
             color=C_TEXT,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#BDC3C7", alpha=0.92))
    ax1.set_xlim(left=0)                                    # <-- batas kiri tetap 0
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.set_ylabel("Kecepatan Linear $v$ (m/s)")
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.set_title("(a) Kecepatan Linear", fontsize=10, color=C_TEXT,
                  loc="left", pad=6)

    # -- Subplot bawah: Kecepatan Angular --
    ax2.plot(df["x"], df["omega_cmd"],                      # <-- ganti df["t"] → df["x"]
             color=C_CMD, lw=1.4, ls="--", alpha=0.85,
             label=r"$\omega_{cmd}$ (Referensi)", zorder=3)
    ax2.plot(df["x"], df["omega_smooth"],                   # <-- ganti df["t"] → df["x"]
             color=C_ACTUAL, lw=2.2,
             label=r"$\omega_{actual}$ (Terukur)", zorder=4)
    ax2.fill_between(df["x"], df["omega_cmd"], df["omega_smooth"],  # <-- ganti df["t"] → df["x"]
                     alpha=0.14, color=C_FILL)
    ax2.axhline(0, color="#95A5A6", lw=0.9, ls=":", zorder=1)

    rmse_w = np.sqrt(np.mean(df["omega_error"]**2))
    ax2.text(0.98, 0.04,
             f"RMSE       = {rmse_w:.4f} rad/s\n"
             f"Mean error = {df['omega_error'].mean():.4f} rad/s\n"
             f"Max error  = {df['omega_error'].max():.4f} rad/s",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=8.5,
             color=C_TEXT,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#BDC3C7", alpha=0.92))
    ax2.set_xlim(0, GOAL[0] + 0.5)                         # <-- sesuaikan range ke panjang X
    ax2.set_ylim(-0.5, 0.5)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.set_xlabel("Posisi X Robot (m)")                    # <-- label sumbu X diubah
    ax2.set_ylabel(r"Kecepatan Angular $\omega$ (rad/s)")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.set_title("(b) Kecepatan Angular", fontsize=10, color=C_TEXT,
                  loc="left", pad=6)

    _save(fig, out, "Fig. 2 - Tracking Kecepatan Linear & Angular")


def fig_omega(df, out):
    """Stub — sudah digabung ke fig_velocity. Tidak menghasilkan file."""
    pass


# =============================================================
#  FIGURE 5 — JARAK MINIMUM KE OBSTACLE
# =============================================================

def fig_clearance(df, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.10, right=0.96)
    _add_header(fig,
                "Jarak Obstacle",
                f"{SCENARIO}  |  Batas keamanan: 0.5 m")

    obs = df[df["min_obstacle_dist"] > 0].copy()
    if len(obs) > 0:
        x_col = obs["x"]   # sumbu horizontal: posisi X robot

        ax.fill_between(x_col, 0, obs["min_obstacle_dist"],
                        alpha=0.15, color=C_ACTUAL)
        ax.plot(x_col, obs["min_obstacle_dist"],
                color=C_ACTUAL, lw=2.2,
                marker="o", markersize=4.5,
                markevery=max(1, len(obs) // 25),
                markerfacecolor=C_ACTUAL, markeredgecolor="white",
                markeredgewidth=0.8,
                label="Jarak min. ke obstacle", zorder=4)
        ax.axhline(0.5, color=C_CMD, lw=1.5, ls="--", alpha=0.85,
                   label="Batas aman (0.5 m)", zorder=3)
        ax.fill_between(x_col, 0, 0.5, alpha=0.07, color=C_CMD)

        danger = obs[obs["min_obstacle_dist"] < 0.5]
        if len(danger) > 0:
            ax.scatter(danger["x"], danger["min_obstacle_dist"],
                       s=55, color=C_CMD, zorder=5,
                       edgecolors="white", linewidths=0.7,
                       label="Di bawah batas aman")

        # -- Titik kecil penanda posisi obstacle --
        for obs_cfg in OBSTACLES:
            ox = obs_cfg["pos"][0]

            # Titik kecil di sumbu X = posisi obstacle
            ax.scatter(ox, 0, s=80, color=C_OBSTACLE, zorder=7,
                       marker="o", edgecolors="white", linewidths=0.8,
                       label=f"Posisi obstacle (X={ox} m)")

            # Garis vertikal penanda posisi obstacle
            ax.axvline(ox, color=C_OBSTACLE, lw=1.2, ls=":",
                       alpha=0.55, zorder=3)

            # Label posisi
            ax.text(ox + 0.08, 0.02,
                    f"Obstacle\nX = {ox} m",
                    fontsize=7.8, color=C_OBSTACLE, fontweight="bold",
                    va="bottom", zorder=9)

        ax.text(0.98, 0.96,
                f"Min. clearance = {obs['min_obstacle_dist'].min():.4f} m\n"
                f"Avg. clearance = {obs['min_obstacle_dist'].mean():.4f} m\n"
                f"Jumlah deteksi = {len(obs)} titik",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                color=C_TEXT,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#BDC3C7", alpha=0.92))
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8.5)
    else:
        ax.text(0.5, 0.5,
                "Tidak ada obstacle terdeteksi\npada skenario ini",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="#AAB7B8", fontstyle="italic")

    ax.set_xlim(-0.5, GOAL[0] + 0.8)
    ax.set_xlabel("Posisi X Robot (m)")
    ax.set_ylabel("Jarak Minimum ke Obstacle (m)")
    _save(fig, out, "Fig. 5 - Jarak Obstacle")


# =============================================================
#  FIGURE 6 — TABEL RINGKASAN METRIK
# =============================================================

def fig_metrics(metrics, df, out):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.subplots_adjust(top=0.82, bottom=0.03, left=0.04, right=0.96)
    _add_header(fig,
                "Ringkasan Metrik Performa Navigasi",
                f"{SCENARIO}  |  Total sampel: {len(df)}")
    ax.axis("off")

    units = [
        "detik", "meter", "m/s", "m/s",
        "meter", "meter", "meter",
        "m/s", "m/s", "rad/s",
        "meter", "meter", "meter",
        "detik",   # Waktu Respon Motor
        "detik",   # t Deteksi Obstacle
        "detik",   # t Manuver Dimulai
        "detik",   # Respon Sistem → Obstacle
        "-",       # Completion Ratio
        "-",       # Efisiensi Lintasan [rev]
    ]
    col_labels = ["Parameter", "Nilai", "Satuan"]
    cell_text  = [[k, str(v), u]
                  for (k, v), u in zip(metrics.items(), units)]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width([0, 1, 2])

    HDR  = "#1A5276"
    ODD  = "#EAF2FB"
    EVEN = "#FFFFFF"
    EDGE = "#D5D8DC"

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(EDGE)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(HDR)
            cell.set_text_props(color="white", fontweight="bold", fontsize=10.5)
            cell.set_height(0.09)
        else:
            cell.set_facecolor(ODD if r % 2 == 1 else EVEN)
            cell.set_text_props(color=C_TEXT)
            cell.set_height(0.075)
            if c == 1:
                cell.set_text_props(color=C_ACTUAL, fontweight="bold")

    _save(fig, out, "Fig. 6 - Ringkasan Metrik")


# =============================================================
#  MAIN
# =============================================================

if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  Robot Navigation Analysis  |  Single Scenario Mode")
    print("=" * 58)
    print(f"  File      : {os.path.basename(CSV_FILE)}")
    print(f"  Goal      : {GOAL}")
    print(f"  Skenario  : {SCENARIO}")
    print(f"  Output    : {OUTPUT_DIR}")
    print("=" * 58)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df      = load_data(CSV_FILE)
    metrics = compute_metrics(df)

    print("\n  Metrik performa:")
    for k, v in metrics.items():
        print(f"    {k:<38} : {v}")

    print("\n  Menyimpan grafik ke folder Gambar Pengujian 2/ ...")

    fig_trajectory(df, os.path.join(OUTPUT_DIR, "01_lintasan_robot.png"))
    fig_velocity  (df, os.path.join(OUTPUT_DIR, "02_tracking_kecepatan_linear.png"))
    fig_omega     (df, os.path.join(OUTPUT_DIR, "03_tracking_kecepatan_angular.png"))
    fig_lateral   (df, os.path.join(OUTPUT_DIR, "04_deviasi_lateral.png"))
    fig_clearance (df, os.path.join(OUTPUT_DIR, "05_jarak_obstacle.png"))
    fig_metrics   (metrics, df, os.path.join(OUTPUT_DIR, "06_ringkasan_metrik.png"))

    print(f"\n  Selesai. Semua file tersimpan di:")
    print(f"  {OUTPUT_DIR}\n")