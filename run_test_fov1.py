"""Quick test: run optimized V4 pipeline on FOV 1 only, then compare with Matlab."""
import sys
import time
from pathlib import Path

APP_DIR = Path("/dobby/yeqiao/software_code/ChrTracer3_py_app_V4_test")
sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import orca_drift
import orca_fit

# ── Paths ──────────────────────────────────────────────────────────────────
DAX_DIR = Path("/dobby/yeqiao/analysis/ORCA_ChrTracer3_processing_optimization"
               "/240402_Granta519cl97_6hWO_MYC5p_30minHyb_30step_analysis"
               "/masked_images/ChrTracer3")
OUT_DIR = Path(__file__).resolve().parent / "analysis_test_fov1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

readout_folders = sorted(DAX_DIR.glob("Readout_*"))
FOV = 1

# ── Parameters (matching Matlab ChrTracer3_FitSpots defaults) ─────────────
FIT_PARAMS = {
    "nm_xy":              108,
    "nm_z":               150,
    "box_half":           15,
    "n_ch":               2,
    "ref_hyb":            1,
    "upsample":           4,       # Matlab actual upsample=4 (from Pars file)
    "max_fine_shift":     4.0,     # Matlab maxXYdrift=4
    "max_fine_shift_z":   6.0,     # Matlab maxZdrift=6
    "fit_half_xy":        4,       # Matlab maxFitWidth=8 → bw=4
    "fit_half_z":         6,       # Matlab maxFitZdepth=12 → bz=6
    "min_h":              200,
    "min_hb_ratio":       1.2,
    "min_ah_ratio":       0.25,
    "max_xy_step":        12.0,
    "max_xy_step_search": 12,      # Matlab maxXYstep (restrict search region)
    "max_z_step":         8,       # Matlab maxZstep (restrict Z search)
    "max_wx_px":          2.0,     # Now in Matlab convention
    "max_wz_sl":          2.5,     # Now in Matlab convention
    "min_signal":         0,
}

# ── Step 1: Drift correction ──────────────────────────────────────────────
print(f"=== Drift correction FOV {FOV} ===")
t0 = time.time()
orca_drift.correct_one_fov(
    fov=FOV,
    readout_folders=readout_folders,
    output_dir=OUT_DIR,
    ref_hyb=1, fid_ch=0, n_ch=2,
    ds=4, crop=150, spot_percentile=75,
    max_fine_shift=5, save_figures=False,
)
print(f"  Drift done in {time.time()-t0:.0f}s")

# ── Step 2: Spot detection ────────────────────────────────────────────────
print(f"\n=== Spot detection FOV {FOV} ===")
ref_folder = readout_folders[0]
inf = orca_drift.read_inf(ref_folder / f"ConvZscan_{FOV:02d}.inf")
H, W, N = inf["height"], inf["width"], inf["n_frames"]
ref_stack = orca_drift.read_dax(ref_folder / f"ConvZscan_{FOV:02d}.dax", H, W, N)
ref_maxproj = orca_drift.fiducial_maxproj(ref_stack, 0, 2)
del ref_stack

spots = orca_fit.detect_spots(ref_maxproj)
spots.to_csv(OUT_DIR / f"fov{FOV:03d}_selectSpots.csv", index=False)
print(f"  Detected {len(spots)} spots")

# ── Step 3: Fit all spots ─────────────────────────────────────────────────
print(f"\n=== Fitting FOV {FOV} ===")
reg_data = pd.read_csv(OUT_DIR / f"fov{FOV:03d}_regData.csv")

t0 = time.time()
all_rows = []
for gen_out in orca_fit.fit_all_spots_stream(
    fov=FOV,
    spots_df=spots,
    readout_folders=readout_folders,
    reg_data=reg_data,
    params=FIT_PARAMS,
    output_dir=OUT_DIR,
):
    pass  # generator drives the fitting

# Load the saved CSV
fits_path = OUT_DIR / f"fov{FOV:03d}_allFits.csv"
if fits_path.exists():
    fits = pd.read_csv(fits_path)
    ok = fits[fits["status"] == "ok"]
    print(f"  Total fits: {len(fits)}, ok: {len(ok)} ({100*len(ok)/len(fits):.1f}%)")
    print(f"  Fitting done in {time.time()-t0:.0f}s")
else:
    print("  No output CSV found — checking for alternative output method")

# ── Quick comparison with Matlab ──────────────────────────────────────────
print("\n=== Quick comparison with Matlab (FOV 1) ===")
MATLAB_DIR = DAX_DIR / "analysis_240410_masked"

# Drift comparison
m_reg = pd.read_csv(MATLAB_DIR / f"fov{FOV:03d}_regData.csv")
p_reg = pd.read_csv(OUT_DIR / f"fov{FOV:03d}_regData.csv")
for col in ["xshift", "yshift"]:
    diff = p_reg[col] - m_reg[col]
    print(f"  Drift {col}: mean_diff={diff.mean():.2f} px, std={diff.std():.2f}, max_abs={diff.abs().max():.0f}")

if fits_path.exists():
    from scipy.spatial import cKDTree
    m_all = pd.read_csv(MATLAB_DIR / "240410_Granta519cl97_6hWO_allFits.csv")
    m_fov = m_all[m_all["fov"] == FOV].copy()
    p_fov = ok if 'ok' in dir() else pd.DataFrame()

    # Match by locus position (same physical spot) — Matlab locusX/Y are in nm
    nm_xy = FIT_PARAMS["nm_xy"]
    m_fov["locusX_px"] = m_fov["locusX"] / nm_xy
    m_fov["locusY_px"] = m_fov["locusY"] / nm_xy

    matched = []
    for hybe in range(1, 61):
        mh = m_fov[m_fov["hybe"] == hybe]
        ph = p_fov[p_fov["hybe"] == hybe] if len(p_fov) > 0 else pd.DataFrame()
        if len(mh) == 0 or len(ph) == 0:
            continue
        tree = cKDTree(mh[["locusX_px", "locusY_px"]].values)
        dists, idxs = tree.query(ph[["locus_x", "locus_y"]].values)
        for pi in range(len(ph)):
            if dists[pi] <= 5:  # within 5 pixels = same locus
                mr = mh.iloc[idxs[pi]]
                pr = ph.iloc[pi]
                matched.append({
                    "hybe": hybe,
                    "dx": pr["x"] - mr["x"], "dy": pr["y"] - mr["y"], "dz": pr["z"] - mr["z"],
                    "wx_ratio": pr["wx"] / mr["wx"], "wz_ratio": pr["wz"] / mr["wz"],
                    "hfit_ratio": pr["h_fit"] / mr["a"],
                })
    if matched:
        mdf = pd.DataFrame(matched)
        print(f"\n  Locus-matched pairs: {len(mdf)}")
        print(f"  dx MAD: {mdf['dx'].abs().median():.1f} nm (includes drift diff)")
        print(f"  dy MAD: {mdf['dy'].abs().median():.1f} nm (includes drift diff)")
        print(f"  dz MAD: {mdf['dz'].abs().median():.1f} nm")
        print(f"  dz mean: {mdf['dz'].mean():.1f} nm")
        print(f"  wx ratio (P/M): {mdf['wx_ratio'].median():.3f} (target ~1.0)")
        print(f"  wz ratio (P/M): {mdf['wz_ratio'].median():.3f} (target ~1.0)")
        print(f"  h_fit/a ratio: {mdf['hfit_ratio'].median():.3f}")
        # Per-hybe check
        for h in [1, 11, 31, 59]:
            hm = mdf[mdf["hybe"] == h]
            if len(hm) > 0:
                print(f"  Hybe {h:2d}: dz MAD={hm['dz'].abs().median():.0f} nm, n={len(hm)}")

print("\nDone.")
