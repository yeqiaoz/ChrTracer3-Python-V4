"""
ChrTracer π full pipeline — drift + spot detection + fitting (2026-03-30).

Processes all 20 FOVs. Skips drift correction for FOVs that already have regData.csv.
Optimized to match Matlab ChrTracer3 output (see CHANGELOG_v4_optimization.md).
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import orca_drift
import orca_fit

# ── Paths ──────────────────────────────────────────────────────────────────
DAX_DIR = Path("/dobby/yeqiao/analysis/ORCA_ChrTracer3_processing_optimization"
               "/240402_Granta519cl97_6hWO_MYC5p_30minHyb_30step_analysis"
               "/masked_images/ChrTracer3")
OUT_DIR = DAX_DIR / "analysis_260330_v4opt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

readout_folders = sorted(DAX_DIR.glob("Readout_*"))
N_FOVS = 20
DRIFT_WORKERS = 4
FIT_WORKERS = 8

# ── Drift correction parameters ───────────────────────────────────────────
DRIFT_PARAMS = dict(
    ref_hyb=1, fid_ch=0, n_ch=2,
    ds=4, crop=200, spot_percentile=75,
    max_fine_shift=5, save_figures=False,
)

# ── Fitting parameters (matched to Matlab ChrTracer3_FitSpots defaults) ──
FIT_PARAMS = {
    "nm_xy":              108,
    "nm_z":               150,
    "box_half":           15,
    "n_ch":               2,
    "ref_hyb":            1,
    "upsample":           4,       # Matlab actual upsample=4
    "max_fine_shift":     4.0,     # Matlab maxXYdrift=4
    "max_fine_shift_z":   6.0,     # Matlab maxZdrift=6
    "fit_half_xy":        4,       # Matlab maxFitWidth=8 -> bw=4
    "fit_half_z":         6,       # Matlab maxFitZdepth=12 -> bz=6
    "min_h":              200,
    "min_hb_ratio":       1.2,
    "min_ah_ratio":       0.25,
    "max_xy_step":        12.0,
    "max_xy_step_search": 12,      # Matlab maxXYstep (restrict search region)
    "max_z_step":         8,       # Matlab maxZstep (restrict Z search)
    "max_wx_px":          2.0,     # Matlab maxSigma=2.0 (Matlab convention)
    "max_wz_sl":          2.5,     # Matlab maxSigmaZ=2.5 (Matlab convention)
    "min_signal":         0,
}

# ── Spot detection parameters ─────────────────────────────────────────────
SPOT_PARAMS = dict(
    threshold_pct=0.997,
    bg_size=50,
    min_dist=30,
    downsample=3,
    border=2,
)


def _run_drift_fov(fov):
    """Module-level wrapper for multiprocessing."""
    orca_drift.correct_one_fov(
        fov=fov,
        readout_folders=readout_folders,
        output_dir=OUT_DIR,
        **DRIFT_PARAMS,
    )
    return fov


# ========================================================================
# STEP 1: Drift correction (skip completed FOVs)
# ========================================================================
def step1_drift():
    print("=" * 60)
    print("STEP 1: Drift correction")
    print("=" * 60)

    fovs_todo = []
    for fov in range(1, N_FOVS + 1):
        csv_path = OUT_DIR / f"fov{fov:03d}_regData.csv"
        if csv_path.exists():
            print(f"  FOV {fov:02d}: regData.csv exists, skipping")
        else:
            fovs_todo.append(fov)

    if not fovs_todo:
        print("  All FOVs already have drift data.\n")
        return

    print(f"  Processing FOVs: {fovs_todo}  ({DRIFT_WORKERS} workers)")
    t0 = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=DRIFT_WORKERS) as executor:
        futures = {executor.submit(_run_drift_fov, fov): fov for fov in fovs_todo}
        for future in as_completed(futures):
            fov = future.result()
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(fovs_todo) - done) / rate if rate > 0 else 0
            print(f"  [{done:2d}/{len(fovs_todo)}] FOV {fov:02d} done  "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    print(f"  Drift done: {done} FOVs in {time.time() - t0:.1f}s\n")


# ========================================================================
# STEP 2: Spot detection (from reference hyb fiducial max-projections)
# ========================================================================
def step2_detect_spots():
    print("=" * 60)
    print("STEP 2: Spot detection")
    print("=" * 60)

    ref_hyb = DRIFT_PARAMS["ref_hyb"]
    ref_folder = readout_folders[ref_hyb - 1]

    for fov in range(1, N_FOVS + 1):
        csv_path = OUT_DIR / f"fov{fov:03d}_selectSpots.csv"
        if csv_path.exists():
            print(f"  FOV {fov:02d}: selectSpots.csv exists, skipping")
            continue

        inf_path = ref_folder / f"ConvZscan_{fov:02d}.inf"
        dax_path = ref_folder / f"ConvZscan_{fov:02d}.dax"

        inf = orca_drift.read_inf(inf_path)
        H, W, N = inf["height"], inf["width"], inf["n_frames"]
        stack = orca_drift.read_dax(dax_path, H, W, N)
        maxproj = orca_drift.fiducial_maxproj(stack, fid_ch=0, n_ch=2)
        del stack

        spots_df = orca_fit.detect_spots(maxproj, **SPOT_PARAMS)
        spots_df.to_csv(csv_path, index=False)
        print(f"  FOV {fov:02d}: {len(spots_df)} spots detected")

    print()


# ========================================================================
# STEP 3: 3D Gaussian fitting
# ========================================================================
def step3_fit():
    print("=" * 60)
    print("STEP 3: 3D Gaussian fitting")
    print("=" * 60)

    t0 = time.time()
    all_rows = []

    for fov in range(1, N_FOVS + 1):
        fits_csv = OUT_DIR / f"fov{fov:03d}_allFits.csv"
        if fits_csv.exists():
            print(f"  FOV {fov:02d}: allFits.csv exists, loading cached")
            df_fov = pd.read_csv(fits_csv)
            all_rows.extend(df_fov.to_dict("records"))
            continue

        spots_csv = OUT_DIR / f"fov{fov:03d}_selectSpots.csv"
        reg_csv = OUT_DIR / f"fov{fov:03d}_regData.csv"

        if not spots_csv.exists() or not reg_csv.exists():
            print(f"  FOV {fov:02d}: missing prerequisite files, skipping")
            continue

        spots_df = pd.read_csv(spots_csv)
        reg_data = pd.read_csv(reg_csv)

        print(f"  FOV {fov:02d}: {len(spots_df)} spots x {len(readout_folders)} hybs ...",
              end="", flush=True)
        t1 = time.time()

        fov_rows = []
        for *_, rows_so_far in orca_fit.fit_all_spots_stream(
            fov=fov,
            spots_df=spots_df,
            readout_folders=readout_folders,
            reg_data=reg_data,
            params=FIT_PARAMS,
            output_dir=None,
            n_workers=FIT_WORKERS,
        ):
            fov_rows = rows_so_far

        df_fov = pd.DataFrame(fov_rows)
        df_fov.to_csv(fits_csv, index=False)
        all_rows.extend(fov_rows)
        n_ok = (df_fov["status"] == "ok").sum()
        print(f" done ({time.time() - t1:.0f}s)  ok={n_ok}/{len(df_fov)}")

    # Merge
    if all_rows:
        merged = (pd.DataFrame(all_rows)
                  .sort_values(["fov", "spot_id", "hybe"])
                  .reset_index(drop=True))
        merged.to_csv(OUT_DIR / "allFits.csv", index=False)
        n_ok = (merged["status"] == "ok").sum()
        print(f"\n  Total: {len(merged):,} rows  ok={n_ok:,}  ({n_ok / len(merged) * 100:.1f}%)")

    print(f"  Fitting done in {time.time() - t0:.0f}s\n")


# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    print(f"Readout folders: {len(readout_folders)}")
    print(f"FOVs: {N_FOVS}")
    print(f"Output: {OUT_DIR}\n")

    step1_drift()
    step2_detect_spots()
    step3_fit()

    print("Pipeline complete.")
