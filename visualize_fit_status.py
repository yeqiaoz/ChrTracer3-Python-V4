"""
Visualize raw image crops for different fit status categories (ok, low_quality, no_peak, low_amp).
Shows readout channel max-projection (XY) + XZ side view + Z profile for representative spots.

Run: python3 visualize_fit_status.py
Output: fit_status_examples.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.ndimage import shift as nd_shift, gaussian_filter
from scipy.optimize import curve_fit

APP_DIR = Path("/dobby/yeqiao/software_code/ChrTracer3_py_app_V4_test")
sys.path.insert(0, str(APP_DIR))
from orca_drift import read_inf, read_dax

# ── Paths ─────────────────────────────────────────────────────────────────
DAX_DIR = Path("/dobby/yeqiao/analysis/ORCA_ChrTracer3_processing_optimization"
               "/240402_Granta519cl97_6hWO_MYC5p_30minHyb_30step_analysis"
               "/masked_images/ChrTracer3")
OUT_DIR = DAX_DIR / "analysis_260330_v4opt"
FIG_DIR = Path(__file__).resolve().parent

readout_folders = sorted(DAX_DIR.glob("Readout_*"))

FITS_CSV = OUT_DIR / "allFits.csv"
BOX_HALF = 15
N_CH = 2
NM_XY = 108
NM_Z = 150
N_EXAMPLES = 3  # examples per status

# Colors per status
STATUS_COLORS = {
    "ok": "#16a085",
    "low_quality": "#e67e22",
    "no_peak": "#c0392b",
    "low_amp": "#8e44ad",
}
STATUS_LABELS = {
    "ok": "OK",
    "low_quality": "Low Quality",
    "no_peak": "No Peak",
    "low_amp": "Low Amplitude",
}
STATUS_REASONS = {
    "ok": "Good Gaussian fit, passes all QC checks",
    "low_quality": "h_fit / h_raw < 0.25 — Gaussian poorly matches the raw peak",
    "no_peak": "Fitted center drifted >12 px from expected position, or no valid peak found",
    "low_amp": "Raw peak intensity h_raw < 200 — too dim to fit reliably",
}


def crop_volume(stack, cx, cy, box_half, n_ch):
    """Extract per-channel 3D crops."""
    n_frames, H, W = stack.shape
    crops = {}
    for ch in range(n_ch):
        ch_stack = stack[ch::n_ch]
        y0 = max(0, cy - box_half); y1 = min(H, cy + box_half + 1)
        x0 = max(0, cx - box_half); x1 = min(W, cx + box_half + 1)
        crops[ch] = ch_stack[:, y0:y1, x0:x1].astype(np.float32)
    return crops


def load_crop(fov, hyb, locus_x, locus_y, reg_data):
    """Load fiducial and readout crops for a spot."""
    reg_row = reg_data.iloc[hyb - 1]
    total_dx = int(reg_row.get("xshift", 0) + reg_row.get("xshift2", 0))
    total_dy = int(reg_row.get("yshift", 0) + reg_row.get("yshift2", 0))
    cx = locus_x + total_dx
    cy = locus_y + total_dy
    folder = readout_folders[hyb - 1]
    inf = read_inf(folder / f"ConvZscan_{fov:02d}.inf")
    H, W, N = inf["height"], inf["width"], inf["n_frames"]
    stack = read_dax(folder / f"ConvZscan_{fov:02d}.dax", H, W, N)
    crops = crop_volume(stack, cx, cy, BOX_HALF, N_CH)
    del stack
    return crops[0], crops[1]  # fiducial, readout


def find_peak_z_profile(vol):
    """Return (max_proj_xy, xz_slice, z_profile, peak_y, peak_x)."""
    maxproj = vol.max(axis=0)
    blur = gaussian_filter(maxproj, sigma=1.0)
    py, px = np.unravel_index(np.argmax(blur), blur.shape)
    # Z profile at peak XY (average 3x3)
    nz, h, w = vol.shape
    y0 = max(0, py - 1); y1 = min(h, py + 2)
    x0 = max(0, px - 1); x1 = min(w, px + 2)
    z_profile = vol[:, y0:y1, x0:x1].mean(axis=(1, 2))
    # XZ slice through peak Y
    xz = vol[:, py, :]  # (nZ, W)
    return maxproj, xz, z_profile, py, px


def gauss1d(z, h, z0, sigma, bg):
    return bg + h * np.exp(-0.5 * ((z - z0) / sigma) ** 2)


def fit_z_profile(z_profile):
    """Try to fit 1D Gaussian to Z profile. Returns (h, z0, sigma, bg) or None."""
    z = np.arange(len(z_profile), dtype=float)
    try:
        bg0 = np.percentile(z_profile, 10)
        h0 = z_profile.max() - bg0
        z0 = float(np.argmax(z_profile))
        p0 = [h0, z0, 2.0, bg0]
        popt, _ = curve_fit(gauss1d, z, z_profile, p0=p0,
                            bounds=([0, 0, 0.5, 0], [np.inf, len(z), 15, np.inf]),
                            maxfev=2000)
        return popt
    except Exception:
        return None


def pick_examples(fov_df, status, n, fovs_to_try=(1, 2, 3, 4, 5)):
    """Pick n examples, trying multiple FOVs if needed."""
    for fov in fovs_to_try:
        sub = fov_df[(fov_df.fov == fov) & (fov_df.status == status) & (fov_df.hybe % 2 == 1)]
        if len(sub) >= n:
            idx = np.linspace(0, len(sub) - 1, n, dtype=int)
            return sub.iloc[idx], fov
        elif len(sub) > 0 and fov == fovs_to_try[-1]:
            return sub.iloc[:n], fov
    # Fall back: any FOV
    sub = fov_df[(fov_df.status == status) & (fov_df.hybe % 2 == 1)]
    if len(sub) >= n:
        idx = np.linspace(0, len(sub) - 1, n, dtype=int)
        return sub.iloc[idx], int(sub.iloc[0].fov)
    return sub.iloc[:n], int(sub.iloc[0].fov) if len(sub) > 0 else 1


def main():
    print("Loading fit results...")
    df = pd.read_csv(FITS_CSV)

    statuses = ["ok", "low_quality", "no_peak", "low_amp"]
    # Preload reg_data per FOV as needed
    reg_cache = {}

    # ── Collect all data first ──
    all_data = {}  # status -> list of (r, fid_crop, dat_crop, fov)
    for status in statuses:
        examples, fov = pick_examples(df, status, N_EXAMPLES)
        if fov not in reg_cache:
            reg_cache[fov] = pd.read_csv(OUT_DIR / f"fov{fov:03d}_regData.csv")

        items = []
        for _, r in examples.iterrows():
            fov_i = int(r.fov)
            if fov_i not in reg_cache:
                reg_cache[fov_i] = pd.read_csv(OUT_DIR / f"fov{fov_i:03d}_regData.csv")
            print(f"  Loading [{status}] fov={fov_i} spot={int(r.spot_id)} hyb={int(r.hybe)}...")
            try:
                fid_crop, dat_crop = load_crop(fov_i, int(r.hybe), int(r.locus_x),
                                               int(r.locus_y), reg_cache[fov_i])
                items.append((r, fid_crop, dat_crop))
            except Exception as e:
                print(f"    Error: {e}")
        all_data[status] = items

    # ── Build figure ──
    # Layout: 4 status rows. Each row has N_EXAMPLES columns.
    # Each cell: top=XY maxproj, middle=XZ side view, bottom=Z profile with fit
    n_rows = len(statuses)
    fig = plt.figure(figsize=(5.5 * N_EXAMPLES + 2.5, 5 * n_rows))

    # Use nested gridspec: outer = status rows, inner = 3 sub-rows per cell
    outer_gs = GridSpec(n_rows, 1, figure=fig, hspace=0.35, top=0.94, bottom=0.03,
                        left=0.08, right=0.98)

    fig.suptitle("Fit Status Examples: What Each Rejection Category Looks Like",
                 fontsize=18, fontweight="bold", y=0.98)

    for si, status in enumerate(statuses):
        items = all_data.get(status, [])
        n_actual = len(items)
        color = STATUS_COLORS[status]

        # Inner grid: 3 rows (XY, XZ, Zprofile) x N_EXAMPLES cols
        inner_gs = outer_gs[si].subgridspec(3, max(N_EXAMPLES, 1),
                                            height_ratios=[1, 0.5, 0.6], hspace=0.3)

        for ei in range(N_EXAMPLES):
            ax_xy = fig.add_subplot(inner_gs[0, ei])
            ax_xz = fig.add_subplot(inner_gs[1, ei])
            ax_zp = fig.add_subplot(inner_gs[2, ei])

            if ei >= n_actual:
                ax_xy.axis("off"); ax_xz.axis("off"); ax_zp.axis("off")
                continue

            r, fid_crop, dat_crop = items[ei]
            spot_id = int(r.spot_id)
            hyb = int(r.hybe)
            fov_i = int(r.fov)

            maxproj, xz, z_profile, py, px = find_peak_z_profile(dat_crop)

            # ── XY max-projection ──
            vmin = np.percentile(maxproj, 5)
            vmax = np.percentile(maxproj, 99.5)
            extent_xy = [0, maxproj.shape[1] * NM_XY, 0, maxproj.shape[0] * NM_XY]
            ax_xy.imshow(maxproj, cmap="inferno", origin="lower",
                         vmin=vmin, vmax=vmax, extent=extent_xy, aspect="equal")
            # Peak crosshair
            ax_xy.axhline(py * NM_XY, color="cyan", linewidth=0.5, alpha=0.5)
            ax_xy.axvline(px * NM_XY, color="cyan", linewidth=0.5, alpha=0.5)
            ax_xy.plot(px * NM_XY, py * NM_XY, "c+", markersize=12, markeredgewidth=1.5)

            # Title with key metrics
            h_raw = r.h if not np.isnan(r.h) else 0
            h_fit = r.h_fit if not np.isnan(r.h_fit) else 0
            ah = h_fit / h_raw if h_raw > 0 else 0
            ax_xy.set_title(f"fov{fov_i} spot{spot_id} hyb{hyb}\n"
                            f"h_raw={h_raw:.0f}  h_fit={h_fit:.0f}  h_fit/h={ah:.2f}",
                            fontsize=9, color=color, fontweight="bold")
            ax_xy.set_xlabel("X (nm)", fontsize=7)
            ax_xy.set_ylabel("Y (nm)", fontsize=7)
            ax_xy.tick_params(labelsize=6)

            # ── XZ side view (through peak Y) ──
            extent_xz = [0, xz.shape[1] * NM_XY, 0, xz.shape[0] * NM_Z]
            vmin_xz = np.percentile(xz, 5)
            vmax_xz = np.percentile(xz, 99)
            ax_xz.imshow(xz, cmap="inferno", origin="lower", aspect="auto",
                         vmin=vmin_xz, vmax=vmax_xz, extent=extent_xz)
            ax_xz.set_xlabel("X (nm)", fontsize=7)
            ax_xz.set_ylabel("Z (nm)", fontsize=7)
            ax_xz.tick_params(labelsize=6)
            wx = r.wx if not np.isnan(r.get("wx", np.nan)) else 0
            wz = r.wz if not np.isnan(r.get("wz", np.nan)) else 0
            ax_xz.set_title(f"XZ slice  |  wx={wx:.0f}  wz={wz:.0f} nm", fontsize=8)

            # ── Z profile + Gaussian fit ──
            z_nm = np.arange(len(z_profile)) * NM_Z
            ax_zp.fill_between(z_nm, z_profile, alpha=0.3, color=color)
            ax_zp.plot(z_nm, z_profile, "-", color=color, linewidth=1.5, label="Raw Z profile")

            # Try Gaussian fit overlay
            popt = fit_z_profile(z_profile)
            if popt is not None:
                z_fine = np.linspace(0, len(z_profile) - 1, 200)
                ax_zp.plot(z_fine * NM_Z, gauss1d(z_fine, *popt), "k--",
                           linewidth=1.2, alpha=0.7, label="1D Gauss fit")

            fq = r.get("fitQuality", np.nan)
            fq_str = f"{fq:.1f}" if not np.isnan(fq) else "N/A"
            ax_zp.set_title(f"fitQuality={fq_str}  bg={r.get('bg', 0):.0f}", fontsize=8)
            ax_zp.set_xlabel("Z (nm)", fontsize=7)
            ax_zp.set_ylabel("Intensity", fontsize=7)
            ax_zp.tick_params(labelsize=6)
            if ei == 0:
                ax_zp.legend(fontsize=7, loc="upper right")

        # Status label on left side
        label_y = outer_gs[si].get_position(fig).y0 + outer_gs[si].get_position(fig).height / 2
        fig.text(0.01, label_y,
                 f"{STATUS_LABELS[status]}\n{STATUS_REASONS[status]}",
                 fontsize=10, fontweight="bold", color=color,
                 va="center", ha="left", rotation=90,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor=color, alpha=0.9))

    out_path = FIG_DIR / "fit_status_examples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
