# -*- coding: utf-8 -*-
"""
 CARA BACA SPREADSHEET:
   - Kolom 5-6 (label/nilai) : parameter posisi & state robot saat ini
     (Xt, Yt, Theta_t, Vt, Omega_t, Goals x/y, X obs, Y obs, dst.)
   - Baris header (kolom 'No','V','Omega', lalu x+1..x+30, y+1..y+30,
     theta+1..theta+30) : 1 baris = 1 kandidat (V, Omega) dalam
     dynamic window, dengan lintasan hasil prediksi 30 langkah (dt=0.1s,
     predict_time=3s)
   - Kolom Head_cost, Dist_cost, Vel_cost, G : komponen biaya tiap
     kandidat (G = biaya total terbobot)
   - Sel G_min / V_CMD / OMEGA_CMD : kandidat dengan G minimum,
     dipakai untuk verifikasi bahwa kandidat terbaik yang dideteksi
     skrip ini sama dengan hasil pilihan spreadsheet
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle, FancyArrow
from matplotlib.transforms import Affine2D

# =============================================================
#  KONFIGURASI
# =============================================================
XLSX_FILE   = "Analisis_Komputasi_DWA.xlsx"
SHEET_NAME  = "Sheet1"
HEADER_ROW  = 17          # baris berisi 'No', 'V', 'Omega', 'x+1', 'y+1', ...
N_STEPS     = 30           # jumlah langkah prediksi (predict_time / dt)
ROBOT_RADIUS = 0.35        # perkiraan radius body robot untuk visual obstacle (m)
ROBOT_LENGTH = 0.65        # panjang body robot (m) - searah heading
ROBOT_WIDTH  = 0.45        # lebar body robot (m) - tegak lurus heading
DPI = 200

_DIR       = os.path.dirname(os.path.abspath(__file__))
XLSX_FILE  = os.path.join(_DIR, XLSX_FILE)
OUT_FILE   = os.path.join(_DIR, "dwa_lintasan_kandidat.png")

C_BG, C_FIG, C_TEXT = "#FFFFFF", "#F0F3F4", "#1C2833"
C_BEST   = "#E74C3C"
C_ROBOT  = "#0A3D62"
C_GOAL   = "#1E8449"
C_OBS    = "#7D3C98"
C_GRID   = "#DFE6E9"

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
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.labelsize":       11,
    "axes.labelweight":     "bold",
    "legend.fontsize":      8.5,
    "legend.framealpha":    0.95,
    "legend.edgecolor":     "#BDC3C7",
    "savefig.bbox":         "tight",
    "savefig.facecolor":    C_FIG,
    "savefig.dpi":          DPI,
})


# =============================================================
#  PARSING SPREADSHEET
# =============================================================

def get_param(df, label, search_col=5, val_col=6):
    """Cari nilai skalar berdasarkan label pada kolom label/nilai."""
    for i in range(len(df)):
        cell = df.iloc[i, search_col]
        if isinstance(cell, str) and cell.strip() == label:
            return float(df.iloc[i, val_col])
    raise ValueError(f"Parameter '{label}' tidak ditemukan di spreadsheet.")


def find_col(header_row, label):
    for i, v in enumerate(header_row):
        if isinstance(v, str) and v.strip() == label:
            return i
    return None


def load_dwa(path, sheet=SHEET_NAME, header_row=HEADER_ROW, n_steps=N_STEPS):
    raw = pd.read_excel(path, sheet_name=sheet, header=None)

    state = {
        "Xt":      get_param(raw, "Xt"),
        "Yt":      get_param(raw, "Yt"),
        "Theta_t": get_param(raw, "Theta_t"),
        "Vt":      get_param(raw, "Vt"),
        "Omega_t": get_param(raw, "Omega_t"),
        "Gx":      get_param(raw, "Goals x"),
        "Gy":      get_param(raw, "Goals y"),
        "Xo":      get_param(raw, "X obs"),
        "Yo":      get_param(raw, "Y obs"),
    }

    hdr = raw.iloc[header_row]
    col_no    = find_col(hdr, "No")
    col_v     = col_no + 1   # kolom 'V' kandidat langsung setelah 'No'
    col_omega = col_no + 2   # kolom 'Omega' kandidat langsung setelah 'V'
    col_G     = find_col(raw.iloc[header_row - 1], "G")
    col_head  = find_col(raw.iloc[header_row - 1], "Head_cost")
    col_dist  = find_col(raw.iloc[header_row - 1], "Dist_cost")
    col_vel   = find_col(raw.iloc[header_row - 1], "Vel_cost")

    x_cols = [find_col(hdr, f"x+{k}") for k in range(1, n_steps + 1)]
    y_cols = [find_col(hdr, f"y+{k}") for k in range(1, n_steps + 1)]

    # -- baris kandidat: kolom 'No' berisi angka, di bawah header --
    is_candidate = raw.iloc[:, col_no].apply(lambda v: isinstance(v, (int, float)) and pd.notna(v))
    cand_rows = raw.index[is_candidate & (raw.index > header_row)]

    candidates = []
    for r in cand_rows:
        traj_x = raw.loc[r, x_cols].astype(float).values
        traj_y = raw.loc[r, y_cols].astype(float).values
        candidates.append({
            "no":     int(raw.iloc[r, col_no]),
            "v":      float(raw.iloc[r, col_v]),
            "omega":  float(raw.iloc[r, col_omega]),
            "x":      np.concatenate([[state["Xt"]], traj_x]),
            "y":      np.concatenate([[state["Yt"]], traj_y]),
            "G":          float(raw.iloc[r, col_G])    if col_G    is not None else np.nan,
            "head_cost":  float(raw.iloc[r, col_head]) if col_head is not None else np.nan,
            "dist_cost":  float(raw.iloc[r, col_dist]) if col_dist is not None else np.nan,
            "vel_cost":   float(raw.iloc[r, col_vel])  if col_vel  is not None else np.nan,
        })

    cand_df = pd.DataFrame(candidates)
    best = cand_df.loc[cand_df["G"].idxmin()]
    return state, cand_df, best


# =============================================================
#  VISUALISASI
# =============================================================

def plot_dwa(state, cand_df, best, out_path):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.0, 0.02], wspace=0.28)
    ax  = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    fig.suptitle("Visualisasi Kandidat Lintasan DWA (Dynamic Window Approach)",
                 fontsize=15, fontweight="bold", color=C_TEXT, y=0.985)
    fig.text(0.5, 0.945,
              f"T = 13.4 s (Skenario 2 integrasi)  |  {len(cand_df)} kandidat (V,\u03c9) dievaluasi  |  "
              f"predict time = 3.0 s, dt = 0.1 s (30 langkah)",
              ha="center", fontsize=9.5, color="#566573")
    fig.add_artist(plt.Line2D([0.04, 0.98], [0.925, 0.925], transform=fig.transFigure,
                    color="#2E86C1", lw=1.4, alpha=0.65))

    # -- colormap biaya G: hijau (bagus/rendah) -> kuning -> merah (buruk/tinggi) --
    cmap = LinearSegmentedColormap.from_list("cost", ["#27AE60", "#F4D03F", "#CB4335"], N=256)
    gmin, gmax = cand_df["G"].min(), cand_df["G"].max()
    norm = plt.Normalize(gmin, gmax)

    # -- semua lintasan kandidat, warna berdasar biaya G, beserta titik prediksinya --
    for _, c in cand_df.iterrows():
        color = cmap(norm(c["G"]))
        ax.plot(c["x"], c["y"], color=color, lw=1.0, alpha=0.55, zorder=2)
        ax.scatter(c["x"][1:], c["y"][1:], color=color, s=5, alpha=0.55, zorder=3, linewidths=0)

    # -- kandidat terbaik: ditonjolkan --
    ax.plot(best["x"], best["y"], color=C_BEST, lw=3.2, zorder=6,
            label=f"Kandidat terbaik (No. {best['no']})", solid_capstyle="round")
    ax.scatter(best["x"][1:], best["y"][1:], color=C_BEST, s=22, zorder=7,
               edgecolors="white", linewidths=0.6)

    # -- posisi & badan robot (persegi panjang 65x45 cm, berorientasi sesuai heading) --
    Xt, Yt, Th = state["Xt"], state["Yt"], state["Theta_t"]
    robot_rect = Rectangle((-ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2), ROBOT_LENGTH, ROBOT_WIDTH,
                            facecolor=C_ROBOT, edgecolor="white", linewidth=1.4,
                            zorder=8, label="Posisi & orientasi robot")
    robot_rect.set_transform(Affine2D().rotate(Th).translate(Xt, Yt) + ax.transData)
    ax.add_patch(robot_rect)

    # -- obstacle --
    ax.add_patch(Circle((state["Xo"], state["Yo"]), ROBOT_RADIUS, color=C_OBS,
                 alpha=0.25, zorder=4))
    ax.scatter([state["Xo"]], [state["Yo"]], s=110, color=C_OBS, marker="X",
               zorder=8, edgecolors="white", linewidths=1.2, label="Obstacle terdeteksi")

    # -- goal: sangat jauh dari fan kandidat (puluhan meter), jadi jangan
    #    dipakai untuk autoscale -- tampilkan sebagai panah arah + jarak --
    dist_goal = np.hypot(state["Gx"] - Xt, state["Gy"] - Yt)
    bearing = np.arctan2(state["Gy"] - Yt, state["Gx"] - Xt)

    # -- colorbar biaya --
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, pad=0.015, shrink=0.85, aspect=24)
    cb.set_label("Biaya Total (G) - semakin rendah semakin baik", fontsize=9, fontweight="bold")
    cb.ax.tick_params(labelsize=8)

    info = (f"Kandidat terbaik: No. {int(best['no'])}\n"
            f"V = {best['v']:.3f} m/s   \u03c9 = {best['omega']:.5f} rad/s\n"
            f"G total     = {best['G']:.4f}\n"
            f"Head_cost   = {best['head_cost']:.4f}\n"
            f"Dist_cost   = {best['dist_cost']:.4f}\n"
            f"Vel_cost    = {best['vel_cost']:.4f}\n"
            f"Jarak ke goal = {dist_goal:.2f} m")
    ax.text(0.015, 0.015, info, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.6, color=C_TEXT, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=C_BEST, linewidth=1.3, alpha=0.94))

    ax.set_xlabel("Posisi X (m)")
    ax.set_ylabel("Posisi Y (m)")
    ax.set_aspect("equal", adjustable="datalim")

    # -- panah arah goal (skala relatif terhadap luas area kandidat, bukan skala nyata) --
    ax.relim(); ax.autoscale_view()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    span = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
    arrow_goal_len = 0.22 * span
    ax.add_patch(FancyArrow(Xt, Yt, arrow_goal_len * np.cos(bearing), arrow_goal_len * np.sin(bearing),
                 width=0.012 * span, head_width=0.05 * span, head_length=0.05 * span,
                 color=C_GOAL, zorder=8, alpha=0.85, length_includes_head=True,
                 label=f"Arah menuju Goal ({dist_goal:.1f} m)"))
    ax.annotate(f"arah Goal\n({dist_goal:.1f} m)",
                xy=(Xt + arrow_goal_len * np.cos(bearing), Yt + arrow_goal_len * np.sin(bearing)),
                xytext=(6, 6), textcoords="offset points", fontsize=8, color=C_GOAL, fontweight="bold")

    ax.legend(loc="upper left", fontsize=8.5)

    # =========================================================
    #  Panel kanan: biaya G tiap kandidat (No urut) + terbaik
    # =========================================================
    order = cand_df.sort_values("no")
    colors_bar = [C_BEST if n == best["no"] else "#AAB7B8" for n in order["no"]]
    ax2.bar(order["no"], order["G"], color=colors_bar, width=0.8, zorder=3)
    ax2.axhline(best["G"], color=C_BEST, lw=1.1, ls="--", alpha=0.7, zorder=2)
    ax2.annotate(f"G minimum = {best['G']:.4f}\n(No. {int(best['no'])})",
                 xy=(best["no"], best["G"]), xytext=(0, 22), textcoords="offset points",
                 ha="center", fontsize=8, fontweight="bold", color=C_BEST,
                 arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.1))
    ax2.set_xlabel("No. Kandidat")
    ax2.set_ylabel("Biaya Total (G)")
    ax2.set_title("Biaya Tiap Kandidat", fontsize=10.5, color=C_TEXT, loc="left")

    fig.text(0.98, 0.012, "Sumber: Analisis_Komputasi_DWA.xlsx", ha="right", va="bottom",
              fontsize=7.5, color="#AAB7B8", transform=fig.transFigure)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(out_path)}")


# =============================================================
#  MAIN
# =============================================================

if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  Visualisasi Kandidat Lintasan DWA")
    print("=" * 58)
    print(f"  File   : {os.path.basename(XLSX_FILE)}")

    state, cand_df, best = load_dwa(XLSX_FILE)

    print(f"  Jumlah kandidat  : {len(cand_df)}")
    print(f"  Posisi robot     : ({state['Xt']:.3f}, {state['Yt']:.3f}) m, "
          f"theta={state['Theta_t']:.3f} rad")
    print(f"  Goal             : ({state['Gx']:.3f}, {state['Gy']:.3f}) m")
    print(f"  Obstacle         : ({state['Xo']:.3f}, {state['Yo']:.3f}) m")
    print(f"  Kandidat terbaik : No.{int(best['no'])}  V={best['v']:.3f} m/s  "
          f"Omega={best['omega']:.5f} rad/s  G={best['G']:.4f}")

    plot_dwa(state, cand_df, best, OUT_FILE)
    print(f"\n  Selesai. Gambar tersimpan di:\n  {OUT_FILE}\n")
