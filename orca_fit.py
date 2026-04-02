"""
orca_fit.py — Spot cropping and 3D Gaussian fitting for ChrTracer3 V3.

V3 change: full 3D Gaussian fit on a small sub-crop (bw=4, bz=6, matching
Matlab FitPsf3D), replacing V2's 2D max-projection + 1D Z-profile approach.

Pipeline per hyb:
  1. Apply drift correction → corrected (cx, cy)
  2. Crop a 3D sub-volume (box_width × box_width × nZ_per_ch) for each channel
  3. Fine sub-pixel alignment of fiducial channel vs reference
  4. Fit 3D Gaussian to readout channel sub-crop (9×9×13 voxels, Matlab-equivalent)
  5. Return fit parameters in nm

Output columns (subset of ChrTracer3 AllFits.csv):
  hybe, x, y, z, h, wx, wy, wz, bg, fitQuality,
  xshift_total, yshift_total, fid_x, fid_y, fid_z, fid_h
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.ndimage import zoom, shift as nd_shift, uniform_filter, maximum_filter, gaussian_filter

from orca_drift import read_inf, read_dax


# ---------------------------------------------------------------------------
# Worker initialiser for ProcessPoolExecutor (must be module-level to pickle)
# ---------------------------------------------------------------------------

_app_dir = str(Path(__file__).parent)


def _worker_init(app_dir: str) -> None:
    """Ensure the app directory is on sys.path in every spawned worker."""
    import sys
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)


# ---------------------------------------------------------------------------
# Quality filter (matches ChrTracer3 Matlab Pars Fit Spots criteria)
# ---------------------------------------------------------------------------

def _find_peak_3d(vol: np.ndarray,
                  max_fit_width: int = 8,
                  max_fit_zdepth: int = 12,
                  peak_blur: float = 0.5,
                  min_peak_height: float = 0.0,
                  ) -> tuple[int, int, int, float]:
    """Find the brightest peak in a 3D volume, matching Matlab FindPeaks3D.

    Matlab's FindPeaks3D:
      1. Excludes border pixels (wb in XY, wz in Z) before searching
      2. Applies Gaussian blur (sigma=peakBlur)
      3. Finds 3D regional maxima via imregionalmax
      4. Keeps the brightest

    We simplify step 3 to argmax (keepBrightest=1 means we only need the
    global max of the blurred interior).

    Returns (z, y, x, h_raw) in the coordinate system of the input volume.
    Returns (-1, -1, -1, 0) if no valid peak found.
    """
    nZ, H, W = vol.shape
    # Border widths matching FindPeaks3D
    wb = min(max_fit_width // 2, H // 2, W // 2)
    wz = min(max_fit_zdepth // 2, nZ // 2) - 1
    wb = max(0, min(wb, H, W))
    wz = max(0, min(wz, H, W))  # Matlab caps wz at min(wz, rows, cols) too

    # Interior region (excluding borders)
    z0 = wz;     z1 = max(nZ - wz, z0 + 1)
    y0 = wb;     y1 = max(H - wb, y0 + 1)
    x0 = wb;     x1 = max(W - wb, x0 + 1)
    interior = vol[z0:z1, y0:y1, x0:x1].astype(float)

    if interior.size == 0:
        return -1, -1, -1, 0.0

    # Gaussian blur (Matlab: imgaussfilt3(Im3, peakBlur))
    if peak_blur > 0:
        interior = gaussian_filter(interior, sigma=peak_blur)

    # Filter by min peak height
    if min_peak_height > 0:
        if interior.max() < min_peak_height:
            return -1, -1, -1, 0.0

    # Find brightest voxel in interior
    pk = np.unravel_index(np.argmax(interior), interior.shape)
    # Map back to full volume coordinates
    pz = z0 + int(pk[0])
    py = y0 + int(pk[1])
    px = x0 + int(pk[2])
    h_raw = float(vol[pz, py, px])  # raw (unblurred) intensity

    return pz, py, px, h_raw


def _check_fit_quality(
    h_raw: float,     # raw peak pixel value (Matlab h = FindPeaks3D output)
    h_fit: float,     # fitted Gaussian amplitude (Matlab a)
    bg: float,        # fitted background (Matlab b)
    x0: float, y0: float,          # fit position in crop pixels
    wx_nm: float, wy_nm: float, wz_nm: float,
    nm_xy: float, nm_z: float,
    params: dict,
) -> str:
    """Return 'ok' or a failure-reason string.

    Criteria match ChrTracer3 Matlab (FitPsf3D + ChrTracer3_FitSpots):

      min_h        — datMinPeakHeight: raw peak intensity (Matlab h from FindPeaks3D)
      min_hb_ratio — datMinHBratio:    h_raw / bg  >= 1.2
      min_ah_ratio — datMinAHratio:    h_fit / h_raw >= 0.25
      max_wx_px    — datMaxFitWidth:   Gaussian sigma in XY pixels
      max_wz_sl    — datMaxFitZdepth:  Gaussian sigma in Z slices
      max_xy_step  — maxXYstep:        fit position within ±maxXYstep px of crop centre
                     (Matlab restricts the search sub-crop to ±maxXYstep around
                      the fiducial; equivalent to requiring the fit position to
                      stay near the crop centre after fine alignment.)
    """
    min_h        = params.get("min_h",        200.0)
    max_wx_px    = params.get("max_wx_px",    2.0)   # Matlab maxSigma=2.0 (now in Matlab convention)
    max_wz_sl    = params.get("max_wz_sl",    2.5)   # Matlab maxSigmaZ=2.5 (now in Matlab convention)
    min_hb_ratio = params.get("min_hb_ratio", 1.2)
    min_ah_ratio = params.get("min_ah_ratio", 0.25)
    max_xy_step  = params.get("max_xy_step",  12.0)   # pixels
    box_half     = params.get("box_half",     15)

    # Raw peak intensity check (Matlab: h >= datMinPeakHeight)
    if h_raw < min_h:
        return "low_amp"
    # Background check
    if bg <= 0:
        return "low_amp"
    # Signal-to-background: h_raw / bg (Matlab: h/b)
    if h_raw / bg < min_hb_ratio:
        return "low_quality"
    # Amplitude-to-peak ratio: h_fit / h_raw (Matlab: a/h)
    if h_raw > 0 and h_fit / h_raw < min_ah_ratio:
        return "low_quality"
    # Width limits
    if wx_nm / nm_xy > max_wx_px or wy_nm / nm_xy > max_wx_px:
        return "wide_spot"
    if wz_nm / nm_z > max_wz_sl:
        return "wide_spot"
    # Positional bounds: fit must be within maxXYstep px of crop centre
    # (equivalent to Matlab's sub-crop restriction around the fiducial)
    if max_xy_step > 0:
        if abs(x0 - box_half) > max_xy_step or abs(y0 - box_half) > max_xy_step:
            return "no_peak"
    return "ok"


# ---------------------------------------------------------------------------
# Gaussian models
# ---------------------------------------------------------------------------

def _gauss1d(x, x0, sigma, h, bg):
    return bg + h * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def _gauss2d_flat(xy, x0, y0, sx, sy, h, bg):
    x, y = xy
    return (bg + h * np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2))).ravel()


def _gauss3d_flat(xyz, x0, y0, sx, sy, z0, sz, h, bg):
    """Standard 3D Gaussian: G = bg + h·exp(-½·((dx/sx)²+(dy/sy)²+(dz/sz)²))."""
    x, y, z = xyz
    return (bg + h * np.exp(-0.5 * (
        ((x - x0) / sx) ** 2 +
        ((y - y0) / sy) ** 2 +
        ((z - z0) / sz) ** 2
    ))).ravel()


def fit_gaussian_1d(profile: np.ndarray) -> tuple[float, float, float, float]:
    """Fit 1D Gaussian. Returns (center, sigma, height, background)."""
    x  = np.arange(len(profile), dtype=float)
    bg = float(profile.min())
    h  = float(profile.max() - bg)
    x0 = float(np.argmax(profile))
    try:
        popt, _ = curve_fit(_gauss1d, x, profile.astype(float),
                            p0=[x0, 2.0, h, bg],
                            bounds=([0, 0.3, 0, -np.inf],
                                    [len(profile), len(profile), np.inf, np.inf]),
                            maxfev=1000)
        return float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3])
    except Exception:
        return x0, 2.0, h, bg


def fit_gaussian_2d(img: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    """Fit 2D Gaussian to image (H, W).
    Returns (x0, y0, sx, sy, h, bg, residual_rms).
    """
    H, W     = img.shape
    yg, xg   = np.mgrid[0:H, 0:W]
    bg       = float(np.percentile(img, 10))
    h        = float(img.max() - bg)
    peak     = np.unravel_index(np.argmax(img), img.shape)
    x0, y0   = float(peak[1]), float(peak[0])

    try:
        popt, pcov = curve_fit(
            _gauss2d_flat,
            (xg.ravel(), yg.ravel()),
            img.ravel().astype(float),
            p0=[x0, y0, 1.5, 1.5, h, bg],
            bounds=([0, 0, 0.3, 0.3, 0, -np.inf],
                    [W,  H,   8,   8, np.inf, np.inf]),
            maxfev=2000,
        )
        fit  = _gauss2d_flat((xg.ravel(), yg.ravel()), *popt).reshape(H, W)
        rmse = float(np.sqrt(np.mean((img - fit) ** 2)))
        return float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]), \
               float(popt[4]), float(popt[5]), rmse
    except Exception:
        return x0, y0, 1.5, 1.5, h, bg, np.nan


def fit_gaussian_3d(
    vol: np.ndarray,
    peak_bound: float = 2.0,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """Fit 3D Gaussian to volume (nZ, H, W) using standard sigma convention.

    Standard convention: G = bg + h*exp(-0.5*((dx/sx)^2+(dy/sy)^2+(dz/sz)^2))
    This differs from Matlab FitPsf3D which uses exp(-((dx/(2s))^2+...)),
    so sx_python = sqrt(2) * wx_matlab for the same PSF.

    Bounds and initial guesses matched to Matlab FitPsf3D defaults:
      - maxSigma=2.0 -> upper bound = 2.0*sqrt(2) = 2.83 in Python convention
      - maxSigmaZ=2.5 -> upper bound = 2.5*sqrt(2) = 3.54
      - peakBound=2 -> fit center within +/-2 px of peak
      - initSigmaXY=1.25 -> 1.25*sqrt(2) = 1.77 in Python convention

    Uses 1-based coordinate grid to match Matlab's meshgrid(1:cols,1:rows,1:stcks).

    Returns (x0, y0, sx, sy, z0, sz, h, bg, rmse, resnorm).
      x0,y0,z0 are 1-based (subtract 1 for 0-based indexing).
      resnorm = sum((vol - fit)^2).
    """
    SQRT2 = np.sqrt(2.0)
    nZ, H, W = vol.shape
    # 1-based grid matching Matlab: meshgrid(1:cols, 1:rows, 1:stcks)
    zg, yg, xg = np.mgrid[1:nZ+1, 1:H+1, 1:W+1]
    data = vol.ravel().astype(float)

    bg0 = float(np.percentile(vol, 10))
    h0  = float(vol.max() - bg0)
    pk  = np.unravel_index(np.argmax(vol), vol.shape)
    # 1-based peak position
    z0i = float(pk[0]) + 1.0
    y0i = float(pk[1]) + 1.0
    x0i = float(pk[2]) + 1.0

    # Matlab-matched bounds (converted to Python sigma convention)
    max_sigma_xy = 2.0 * SQRT2   # Matlab maxSigma=2.0
    min_sigma    = 0.1 * SQRT2   # Matlab minSigma=0.1
    max_sigma_z  = 2.5 * SQRT2   # Matlab maxSigmaZ=2.5
    min_sigma_z  = 0.1 * SQRT2   # Matlab minSigmaZ=0.1
    init_sxy     = 1.25 * SQRT2  # Matlab initSigmaXY=1.25

    try:
        popt, _ = curve_fit(
            _gauss3d_flat,
            (xg.ravel(), yg.ravel(), zg.ravel()),
            data,
            p0=[x0i, y0i, init_sxy, init_sxy, z0i, 2.5 * SQRT2, h0, bg0],
            bounds=(
                [x0i - peak_bound, y0i - peak_bound, min_sigma, min_sigma,
                 z0i - peak_bound, min_sigma_z, 0,       0],
                [x0i + peak_bound, y0i + peak_bound, max_sigma_xy, max_sigma_xy,
                 z0i + peak_bound, max_sigma_z, 2**16,   2**16],
            ),
            maxfev=5000,
        )
        fit       = _gauss3d_flat((xg.ravel(), yg.ravel(), zg.ravel()), *popt).reshape(nZ, H, W)
        residuals = vol.astype(float) - fit
        resnorm   = float(np.sum(residuals ** 2))
        rmse      = float(np.sqrt(np.mean(residuals ** 2)))
        return (float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]),
                float(popt[4]), float(popt[5]), float(popt[6]), float(popt[7]),
                rmse, resnorm)
    except Exception:
        resnorm = float(np.sum((vol.astype(float) - bg0) ** 2))
        return x0i, y0i, init_sxy, init_sxy, z0i, 2.5 * SQRT2, h0, bg0, np.nan, resnorm


# ---------------------------------------------------------------------------
# Volume cropping and channel splitting
# ---------------------------------------------------------------------------

def crop_volume(stack: np.ndarray, cx: int, cy: int, half: int,
                n_ch: int = 2) -> dict[int, np.ndarray]:
    """Crop a ± half-pixel box around (cx, cy) for each channel.

    stack : (N_frames, H, W)
    Returns {ch_idx: (nZ, crop_H, crop_W)} per channel.
    """
    H, W  = stack.shape[1], stack.shape[2]
    r0    = max(0, cy - half);  r1 = min(H, cy + half)
    c0    = max(0, cx - half);  c1 = min(W, cx + half)
    crops = {}
    for ch in range(n_ch):
        frames      = stack[ch::n_ch]          # (nZ, H, W)
        crops[ch]   = frames[:, r0:r1, c0:c1]  # (nZ, crop_H, crop_W)
    return crops


# ---------------------------------------------------------------------------
# Fine sub-pixel alignment within crop (fiducial channel)
# ---------------------------------------------------------------------------

def _xcorr_shift_2d(ref_2d: np.ndarray, mov_2d: np.ndarray,
                    upsample: int) -> tuple[float, float, float]:
    """Compute sub-pixel shift between two 2D images via upsampled
    unnormalized cross-correlation (matching Matlab CorrAlignFast).
    Returns (dy, dx, xcorr_peak_value).
    """
    # Pad to same shape
    H = max(ref_2d.shape[0], mov_2d.shape[0])
    W = max(ref_2d.shape[1], mov_2d.shape[1])
    def _pad(im):
        if im.shape == (H, W):
            return im
        tmp = np.zeros((H, W), dtype=float)
        tmp[:im.shape[0], :im.shape[1]] = im
        return tmp
    ref_p = _pad(ref_2d.astype(float))
    mov_p = _pad(mov_2d.astype(float))

    # Subtract edge baseline (matches Matlab Register3D thresholding)
    for arr in (ref_p, mov_p):
        edges = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        arr -= np.percentile(edges, 90)
        arr[arr < 0] = 0

    # Upsample
    ref_up = zoom(ref_p, upsample, order=1)
    mov_up = zoom(mov_p, upsample, order=1)

    # Unnormalized cross-correlation (Matlab style)
    F_ref = np.fft.fft2(ref_up)
    F_mov = np.fft.fft2(mov_up)
    xcorr = np.fft.fftshift(np.fft.ifft2(np.conj(F_ref) * F_mov).real)

    cy, cx = np.array(xcorr.shape) // 2
    pk     = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    dy     = (pk[0] - cy) / upsample
    dx     = (pk[1] - cx) / upsample
    return dy, dx, float(xcorr[pk])


def fine_align_crop(ref_crop: np.ndarray, mov_crop: np.ndarray,
                    upsample: int = 4,
                    upsample_z: int = 4,
                    max_shift_xy: int = 4,
                    max_shift_z: int = 6) -> tuple[float, float, float, float]:
    """Compute sub-pixel XYZ shift between two 3D crops (nZ, H, W).

    Matches Matlab Register3D approach:
      1. Crop a small box around the center of the crop (±max_shift_xy XY,
         ±max_shift_z Z) — this ensures the correlation focuses on the
         fiducial bead, not background structure.
      2. Upsample the 3D sub-volume.
      3. Subtract 3D edge baseline.
      4. XY shift from max-projection along Z.
      5. Z shift from YZ max-projection (permute [3,2,1], max along dim3).
         The yshift of the YZ correlation = Z shift.

    Returns (dy, dx, dz, xcorr_peak_value).
    """
    nZ, H, W = ref_crop.shape

    # 1. Find fiducial peak in reference to center the crop
    #    (Matlab passes center=[fidTable.x, fidTable.y, fidTable.z] to Register3D)
    fid_cz, fid_cy, fid_cx, _ = _find_peak_3d(
        ref_crop, max_fit_width=8, max_fit_zdepth=12, peak_blur=0.5)
    if fid_cz < 0:
        fid_cx, fid_cy, fid_cz = W // 2, H // 2, nZ // 2

    cx_c, cy_c, cz_c = fid_cx, fid_cy, fid_cz
    bxy = max(4, max_shift_xy)
    bz  = max(4, max_shift_z)

    y0 = max(0, cy_c - bxy);  y1 = min(H, cy_c + bxy + 1)
    x0 = max(0, cx_c - bxy);  x1 = min(W, cx_c + bxy + 1)
    z0 = max(0, cz_c - bz);   z1 = min(nZ, cz_c + bz + 1)

    ref_sub = ref_crop[z0:z1, y0:y1, x0:x1].astype(float)
    mov_sub = mov_crop[z0:z1, y0:y1, x0:x1].astype(float)

    # 2. Upsample 3D (matching Register3D imresize3)
    sz_r, sy_r, sx_r = ref_sub.shape
    up_shape = (round(sz_r * upsample_z), round(sy_r * upsample), round(sx_r * upsample))
    zoom_factors = (upsample_z / 1.0, upsample / 1.0, upsample / 1.0)
    ref_up3 = zoom(ref_sub, zoom_factors, order=1)
    mov_up3 = zoom(mov_sub, zoom_factors, order=1)

    # 3. Edge subtraction (matching Register3D: 90th percentile of 3D edges)
    def _edge_subtract_3d(vol):
        edges = np.concatenate([
            vol[0, :, :].ravel(), vol[-1, :, :].ravel(),   # Z faces
            vol[:, 0, :].ravel(), vol[:, -1, :].ravel(),   # Y faces
            vol[:, :, 0].ravel(), vol[:, :, -1].ravel(),   # X faces
        ])
        baseline = np.percentile(edges, 90)
        vol = vol - baseline
        vol[vol < 0] = 0
        return vol

    ref_up3 = _edge_subtract_3d(ref_up3)
    mov_up3 = _edge_subtract_3d(mov_up3)

    # 4. XY projection correlation (max over Z axis=0)
    ref_xy = ref_up3.max(axis=0)
    mov_xy = mov_up3.max(axis=0)
    F_ref = np.fft.fft2(ref_xy)
    F_mov = np.fft.fft2(mov_xy)
    xcorr_xy = np.fft.fftshift(np.fft.ifft2(np.conj(F_ref) * F_mov).real)
    cy_xy, cx_xy = np.array(xcorr_xy.shape) // 2
    pk_xy = np.unravel_index(np.argmax(xcorr_xy), xcorr_xy.shape)
    dy_xy_up = pk_xy[0] - cy_xy
    dx_xy_up = pk_xy[1] - cx_xy

    # 5. YZ projection correlation (Matlab: permute([3,2,1]) then max over dim3)
    # permute([3,2,1]) of (nZ,H,W) → (W,H,nZ), max over dim3 → (W,H)
    # In numpy: transpose(2,1,0) then max over axis=2 → (nZ_up, H_up) projected over X
    # But Matlab's permute([3,2,1]) on (rows,cols,stks) gives (stks,cols,rows)
    # then max over dim3 (rows) gives (stks, cols)
    # In Python (Z,Y,X), transpose(2,1,0) gives (X,Y,Z), max over axis=2 gives (X,Y)
    # That's wrong. Let me think...
    # Matlab dim order: (rows=Y, cols=X, stks=Z)
    # permute([3,2,1]) → (Z, X, Y)
    # max over dim3 (Y) → (Z, X)
    # Then CorrAlignFast's yshift = Z shift, xshift = X shift
    #
    # Python dim order: (Z, Y, X)
    # To match: we want max over Y → (Z, X)
    ref_yz = ref_up3.max(axis=1)  # max over Y → (nZ_up, W_up)
    mov_yz = mov_up3.max(axis=1)

    F_ref_yz = np.fft.fft2(ref_yz)
    F_mov_yz = np.fft.fft2(mov_yz)
    xcorr_yz = np.fft.fftshift(np.fft.ifft2(np.conj(F_ref_yz) * F_mov_yz).real)
    cy_yz, cx_yz = np.array(xcorr_yz.shape) // 2
    pk_yz = np.unravel_index(np.argmax(xcorr_yz), xcorr_yz.shape)
    dz_up = pk_yz[0] - cy_yz   # Z shift (row direction of ZX projection)
    dx_yz_up = pk_yz[1] - cx_yz  # X shift from YZ projection

    # 6. Combine and rescale (matching Register3D lines 132-138)
    xshift = (dx_xy_up + dx_yz_up) / 2.0 / upsample  # average X from both projections
    yshift = dy_xy_up / upsample                       # Y from XY projection only
    zshift = dz_up / upsample_z                        # Z from YZ projection

    score = max(float(xcorr_xy[pk_xy]), float(xcorr_yz[pk_yz]))
    return yshift, xshift, zshift, score


# ---------------------------------------------------------------------------
# Single-hyb fit
# ---------------------------------------------------------------------------

def fit_one_hyb(
    fov: int,
    hyb: int,                       # 1-based
    test_x: int, test_y: int,       # locus center in full image (pixels)
    reg_row: pd.Series,             # row from regData DataFrame
    ref_crop_fid: np.ndarray,       # reference fiducial crop (nZ, H, W)
    readout_folder: Path,
    params: dict,
) -> dict:
    """Fit one hyb round for a single test locus.

    Returns a dict with fit results and image data for display.
    """
    nm_xy    = params.get("nm_xy", 108)
    nm_z     = params.get("nm_z", 150)
    box_half = params.get("box_half", 15)   # crop = 2*box_half pixels
    upsample = params.get("upsample", 4)
    n_ch     = params.get("n_ch", 2)

    # Apply drift correction
    total_dx = int(reg_row.get("xshift", 0) + reg_row.get("xshift2", 0))
    total_dy = int(reg_row.get("yshift", 0) + reg_row.get("yshift2", 0))
    cx = test_x + total_dx
    cy = test_y + total_dy

    # Read DAX
    dax_name = f"ConvZscan_{fov:02d}.dax"
    inf_name = f"ConvZscan_{fov:02d}.inf"
    dax_path = readout_folder / dax_name

    result = {
        "hybe": hyb, "cx": cx, "cy": cy,
        "xshift_total": total_dx, "yshift_total": total_dy,
    }

    if not dax_path.exists():
        result["status"] = "missing"
        return result

    inf   = read_inf(readout_folder / inf_name)
    H, W, N = inf["height"], inf["width"], inf["n_frames"]
    stack = read_dax(dax_path, H, W, N)

    crops = crop_volume(stack, cx, cy, box_half, n_ch)
    del stack

    fid_crop = crops.get(0)   # fiducial channel (ch0)
    dat_crop = crops.get(1)   # readout channel (ch1)

    # ── Fine alignment using fiducial (XYZ, matching Matlab Register3D) ──
    fine_dy, fine_dx, fine_dz, xcorr_peak = 0.0, 0.0, 0.0, 0.0
    fid_crop_aligned = fid_crop

    max_fine_shift = params.get("max_fine_shift", 5.0)
    max_fine_shift_z = params.get("max_fine_shift_z", 6.0)
    if ref_crop_fid is not None and fid_crop is not None:
        _dy, _dx, _dz, xcorr_peak = fine_align_crop(
            ref_crop_fid, fid_crop, upsample,
            max_shift_xy=int(params.get("max_fine_shift", 4)),
            max_shift_z=int(params.get("max_fine_shift_z", 6)))
        if abs(_dy) <= max_fine_shift and abs(_dx) <= max_fine_shift:
            fine_dy, fine_dx = _dy, _dx
        if abs(_dz) <= max_fine_shift_z:
            fine_dz = _dz
        # Apply fine 3D shift to readout crop
        if dat_crop is not None:
            dat_crop = nd_shift(dat_crop, (-fine_dz, -fine_dy, -fine_dx),
                                order=1, mode="constant", cval=0)
        fid_crop_aligned = nd_shift(fid_crop, (-fine_dz, -fine_dy, -fine_dx),
                                    order=1, mode="constant", cval=0)

    result.update({
        "fine_dx": fine_dx, "fine_dy": fine_dy, "fid_xcorr_peak": xcorr_peak,
        "fid_crop": fid_crop, "fid_crop_aligned": fid_crop_aligned,
        "dat_crop": dat_crop,
    })

    if dat_crop is None or dat_crop.size == 0:
        result["status"] = "no_data"
        return result

    # ── Fit readout channel ─────────────────────────────────────
    # Matching Matlab ChrTracer3_FitSpots: restrict data search to a box
    # around the fiducial peak position (±maxXYstep XY, ±maxZstep Z),
    # then fit 3D Gaussian within ±fit_half_xy/±fit_half_z of data peak.
    nZ, H_cr, W_cr = dat_crop.shape
    fhxy = int(params.get("fit_half_xy", 4))
    fhz  = int(params.get("fit_half_z",  6))
    max_xy_step = int(params.get("max_xy_step_search", 12))  # Matlab maxXYstep
    max_z_step  = int(params.get("max_z_step", 8))           # Matlab maxZstep

    # Find fiducial peak position to define restricted search region
    # (Matlab: FindPeaks3D on kernel with fidMaxFitWidth/fidMaxFitZdepth border exclusion)
    fid_fit_width = int(params.get("fid_max_fit_width", 8))
    fid_fit_zdepth = int(params.get("fid_max_fit_zdepth", 12))
    if ref_crop_fid is not None:
        fid_cz, fid_cy, fid_cx, _ = _find_peak_3d(
            ref_crop_fid,
            max_fit_width=fid_fit_width,
            max_fit_zdepth=fid_fit_zdepth,
            peak_blur=0.5,
        )
        if fid_cz < 0:
            fid_cx, fid_cy = W_cr // 2, H_cr // 2
            fid_cz = nZ // 2
    else:
        fid_cx, fid_cy = W_cr // 2, H_cr // 2
        fid_cz = nZ // 2

    # Restricted search region around fiducial (matching Matlab xs{i}, ys{i}, zs{i})
    rx0 = max(0, fid_cx - max_xy_step);  rx1 = min(W_cr, fid_cx + max_xy_step + 1)
    ry0 = max(0, fid_cy - max_xy_step);  ry1 = min(H_cr, fid_cy + max_xy_step + 1)
    rz0 = max(0, fid_cz - max_z_step);   rz1 = min(nZ,   fid_cz + max_z_step + 1)
    restricted = dat_crop[rz0:rz1, ry0:ry1, rx0:rx1]

    if restricted.size == 0:
        result["status"] = "no_peak"
        return result

    # Find data peak within restricted region using Matlab-matched FindPeaks3D
    # (Gaussian blur + border exclusion, matching FitPsf3D's internal peak finding)
    dat_fit_width = int(params.get("dat_max_fit_width", 2 * fhxy))
    dat_fit_zdepth = int(params.get("dat_max_fit_zdepth", 2 * fhz))
    pk_z_r, pk_y_r, pk_x_r, h_raw = _find_peak_3d(
        restricted,
        max_fit_width=dat_fit_width,
        max_fit_zdepth=dat_fit_zdepth,
        peak_blur=0.5,
    )

    if pk_z_r < 0:
        result["status"] = "no_peak"
        return result

    # Sub-crop for fitting WITHIN the restricted region (matching Matlab FitPsf3D)
    nZ_r, H_r, W_r = restricted.shape
    xs_r = max(0, pk_x_r - fhxy);  xe_r = min(W_r, pk_x_r + fhxy + 1)
    ys_r = max(0, pk_y_r - fhxy);  ye_r = min(H_r, pk_y_r + fhxy + 1)
    zs_r = max(0, pk_z_r - fhz);   ze_r = min(nZ_r, pk_z_r + fhz + 1)
    fit_sub = restricted[zs_r:ze_r, ys_r:ye_r, xs_r:xe_r]

    # 3D Gaussian fit (1-based coords, Python sigma convention internally)
    x0s, y0s, sx, sy, z0s, sz, h_fit, bg, rmse, resnorm = fit_gaussian_3d(fit_sub)

    # Convert fit position to full-crop coordinates:
    # 1-based in sub-crop → 0-based in restricted → 0-based in full crop
    # Matching Matlab: dTable.z = (min(zr)-1+par(5))*nmZpix + (zs{i}(1)-1)*nmZpix
    x0 = (x0s - 1) + xs_r + rx0   # 0-based in full crop
    y0 = (y0s - 1) + ys_r + ry0
    z0 = (z0s - 1) + zs_r + rz0

    # Convert to nm (1-based position * pixel size)
    x_nm = (x0 + 1) * nm_xy
    y_nm = (y0 + 1) * nm_xy
    z_nm = (z0 + 1) * nm_z

    # fitQuality = sum(residuals^2) / h_fit^2 — matches Matlab FitPsf3D resRatio formula
    fit_quality = float(resnorm / (h_fit ** 2 + 1e-10))
    # Convert widths to Matlab convention: divide by sqrt(2)
    SQRT2 = np.sqrt(2.0)
    wx_nm = sx * nm_xy / SQRT2
    wy_nm = sy * nm_xy / SQRT2
    wz_nm = sz * nm_z / SQRT2
    fit_status = _check_fit_quality(h_raw, h_fit, bg, x0, y0,
                                    wx_nm, wy_nm, wz_nm, nm_xy, nm_z, params)

    result.update({
        "status": fit_status,
        "x": x_nm, "y": y_nm, "z": z_nm,
        "h": h_raw, "h_fit": h_fit, "wx": wx_nm, "wy": wy_nm, "wz": wz_nm,
        "bg": bg, "fitQuality": fit_quality, "rmse": rmse,
        "x_px": x0, "y_px": y0, "z_px": z0,
        "dat_maxproj": dat_crop.max(axis=0),
    })
    return result


# ---------------------------------------------------------------------------
# Lean fit from pre-loaded crops (no I/O, no display data stored)
# Used by fit_all_spots_stream for batch fitting.
# ---------------------------------------------------------------------------

def _fit_from_crops(
    fov: int, hyb: int,
    locus_x: int, locus_y: int,
    total_dx: int, total_dy: int,
    cx: int, cy: int,
    ref_crop_fid: np.ndarray | None,
    fid_crop: np.ndarray | None,
    dat_crop: np.ndarray | None,
    params: dict,
) -> dict:
    """Fit one (spot, hyb) given pre-loaded crops — no DAX I/O.

    Equivalent to fit_one_hyb but skips disk reads and omits display
    arrays (fid_crop, dat_crop) from the returned dict.
    """
    nm_xy    = params.get("nm_xy", 108)
    nm_z     = params.get("nm_z", 150)
    upsample = params.get("upsample", 4)

    result = {
        "hybe": hyb, "cx": cx, "cy": cy,
        "xshift_total": total_dx, "yshift_total": total_dy,
    }

    if dat_crop is None or dat_crop.size == 0:
        result["status"] = "no_data"
        return result

    # Fine alignment using fiducial channel (XYZ, matching Matlab Register3D)
    max_fine_shift = params.get("max_fine_shift", 5.0)
    max_fine_shift_z = params.get("max_fine_shift_z", 6.0)
    fine_dy, fine_dx, fine_dz = 0.0, 0.0, 0.0
    if ref_crop_fid is not None and fid_crop is not None:
        _dy, _dx, _dz, _ = fine_align_crop(
            ref_crop_fid, fid_crop, upsample,
            max_shift_xy=int(params.get("max_fine_shift", 4)),
            max_shift_z=int(params.get("max_fine_shift_z", 6)))
        if abs(_dy) <= max_fine_shift and abs(_dx) <= max_fine_shift:
            fine_dy, fine_dx = _dy, _dx
        if abs(_dz) <= max_fine_shift_z:
            fine_dz = _dz
        dat_crop = nd_shift(dat_crop, (-fine_dz, -fine_dy, -fine_dx),
                            order=1, mode="constant", cval=0)

    # Restrict data search to region around fiducial peak (Matlab approach)
    nZ, H_cr, W_cr = dat_crop.shape
    fhxy = int(params.get("fit_half_xy", 4))
    fhz  = int(params.get("fit_half_z",  6))
    max_xy_step = int(params.get("max_xy_step_search", 12))
    max_z_step  = int(params.get("max_z_step", 8))

    # Find fiducial peak to define restricted search region
    # (Matlab: FindPeaks3D on kernel with fidMaxFitWidth/fidMaxFitZdepth border exclusion)
    fid_fit_width = int(params.get("fid_max_fit_width", 8))
    fid_fit_zdepth = int(params.get("fid_max_fit_zdepth", 12))
    if ref_crop_fid is not None:
        fid_cz, fid_cy, fid_cx, _ = _find_peak_3d(
            ref_crop_fid,
            max_fit_width=fid_fit_width,
            max_fit_zdepth=fid_fit_zdepth,
            peak_blur=0.5,
        )
        if fid_cz < 0:
            fid_cx, fid_cy = W_cr // 2, H_cr // 2
            fid_cz = nZ // 2
    else:
        fid_cx, fid_cy = W_cr // 2, H_cr // 2
        fid_cz = nZ // 2

    rx0 = max(0, fid_cx - max_xy_step);  rx1 = min(W_cr, fid_cx + max_xy_step + 1)
    ry0 = max(0, fid_cy - max_xy_step);  ry1 = min(H_cr, fid_cy + max_xy_step + 1)
    rz0 = max(0, fid_cz - max_z_step);   rz1 = min(nZ,   fid_cz + max_z_step + 1)
    restricted = dat_crop[rz0:rz1, ry0:ry1, rx0:rx1]

    if restricted.size == 0:
        result["status"] = "no_peak"
        return result

    # Find data peak within restricted region using Matlab-matched FindPeaks3D
    dat_fit_width = int(params.get("dat_max_fit_width", 2 * fhxy))
    dat_fit_zdepth = int(params.get("dat_max_fit_zdepth", 2 * fhz))
    pk_z_r, pk_y_r, pk_x_r, h_raw = _find_peak_3d(
        restricted,
        max_fit_width=dat_fit_width,
        max_fit_zdepth=dat_fit_zdepth,
        peak_blur=0.5,
    )

    if pk_z_r < 0:
        result["status"] = "no_peak"
        return result

    # Sub-crop for fitting WITHIN the restricted region (matching Matlab FitPsf3D)
    nZ_r, H_r, W_r = restricted.shape
    xs_r = max(0, pk_x_r - fhxy);  xe_r = min(W_r, pk_x_r + fhxy + 1)
    ys_r = max(0, pk_y_r - fhxy);  ye_r = min(H_r, pk_y_r + fhxy + 1)
    zs_r = max(0, pk_z_r - fhz);   ze_r = min(nZ_r, pk_z_r + fhz + 1)
    fit_sub = restricted[zs_r:ze_r, ys_r:ye_r, xs_r:xe_r]

    # 3D Gaussian fit (1-based coords, Python sigma convention internally)
    x0s, y0s, sx, sy, z0s, sz, h_fit, bg, rmse, resnorm = fit_gaussian_3d(fit_sub)

    # Convert fit position to full-crop coordinates:
    # 1-based in sub-crop → 0-based in restricted → 0-based in full crop
    x0 = (x0s - 1) + xs_r + rx0
    y0 = (y0s - 1) + ys_r + ry0
    z0 = (z0s - 1) + zs_r + rz0

    # fitQuality = sum(residuals^2) / h_fit^2 — matches Matlab FitPsf3D resRatio
    fit_quality = float(resnorm / (h_fit ** 2 + 1e-10))
    # Convert widths to Matlab convention: divide by sqrt(2)
    SQRT2 = np.sqrt(2.0)
    wx_nm = sx * nm_xy / SQRT2
    wy_nm = sy * nm_xy / SQRT2
    wz_nm = sz * nm_z / SQRT2
    fit_status = _check_fit_quality(h_raw, h_fit, bg, x0, y0,
                                    wx_nm, wy_nm, wz_nm, nm_xy, nm_z, params)

    result.update({
        "status":       fit_status,
        "x":            (x0 + 1) * nm_xy,
        "y":            (y0 + 1) * nm_xy,
        "z":            (z0 + 1) * nm_z,
        "h":            h_raw,
        "h_fit":        h_fit,
        "wx":           wx_nm,
        "wy":           wy_nm,
        "wz":           wz_nm,
        "bg":           bg,
        "fitQuality":   fit_quality,
        "rmse":         rmse,
        "fine_dx":      fine_dx,
        "fine_dy":      fine_dy,
    })
    return result


# ---------------------------------------------------------------------------
# FitSpot figure (matches ChrTracer3 Matlab style)
# ---------------------------------------------------------------------------

def _norm_for_display(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - lo) / (hi - lo + 1e-10), 0, 1).astype(np.float32)


def make_fitspot_figure(
    ref_crop_fid: np.ndarray,    # (nZ, H, W)
    fid_crop: np.ndarray,        # (nZ, H, W) before alignment
    fid_crop_aligned: np.ndarray,
    dat_maxproj: np.ndarray,     # (H, W)
    x_px: float, y_px: float, z_px: float,
    fov: int, hyb: int,
) -> plt.Figure:
    """4-panel FitSpot figure: fid XY/XZ before and after alignment."""
    def _overlay(ref, mov):
        r = _norm_for_display(ref)
        g = _norm_for_display(mov)
        return np.stack([r, g, np.zeros_like(r)], axis=-1)

    ref_xy = ref_crop_fid.max(axis=0)
    mov_xy = fid_crop.max(axis=0)
    aln_xy = fid_crop_aligned.max(axis=0)

    # XZ: max over Y axis
    ref_xz = ref_crop_fid.max(axis=1)    # (nZ, W)
    mov_xz = fid_crop.max(axis=1)
    aln_xz = fid_crop_aligned.max(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(_overlay(ref_xy, mov_xy), origin="upper")
    axes[0, 0].set_title(f"corr. fid x,y  fov{fov} h{hyb}")

    axes[0, 1].imshow(_overlay(ref_xz.T, mov_xz.T), origin="upper")
    axes[0, 1].set_title("corr. fid x,z")

    axes[1, 0].imshow(_overlay(ref_xy, aln_xy), origin="upper")
    axes[1, 0].plot(x_px, y_px, "r+", markersize=10, markeredgewidth=1.5)
    axes[1, 0].set_title("corr. fid x,y  aligned")

    axes[1, 1].imshow(_overlay(ref_xz.T, aln_xz.T), origin="upper")
    axes[1, 1].plot(x_px, z_px, "r+", markersize=10, markeredgewidth=1.5)
    axes[1, 1].set_title("corr. fid x,z  aligned")

    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streaming fit across all hybs for one test locus
# ---------------------------------------------------------------------------

def fit_test_spot_stream(
    fov: int,
    test_x: int, test_y: int,
    readout_folders: list[Path],
    reg_data: pd.DataFrame,
    params: dict,
    output_dir: Path | None = None,
):
    """Generator: process one hyb at a time.

    Yields (hyb, n_total, result_dict, fig_or_None, rows_so_far).
    """
    n_hybs   = len(readout_folders)
    ref_hyb  = params.get("ref_hyb", 1)
    box_half = params.get("box_half", 15)
    n_ch     = params.get("n_ch", 2)

    # Pre-read reference fiducial crop
    ref_folder  = readout_folders[ref_hyb - 1]
    dax_name    = f"ConvZscan_{fov:02d}.dax"
    inf_name    = f"ConvZscan_{fov:02d}.inf"
    inf         = read_inf(ref_folder / inf_name)
    H, W, N     = inf["height"], inf["width"], inf["n_frames"]
    ref_stack   = read_dax(ref_folder / dax_name, H, W, N)
    ref_crops   = crop_volume(ref_stack, test_x, test_y, box_half, n_ch)
    ref_crop_fid = ref_crops.get(0)
    del ref_stack

    rows = []

    for i, folder in enumerate(readout_folders):
        hyb     = i + 1
        reg_row = reg_data.iloc[i] if i < len(reg_data) else pd.Series(
            {"xshift": 0, "yshift": 0, "xshift2": 0, "yshift2": 0})

        result = fit_one_hyb(
            fov=fov, hyb=hyb,
            test_x=test_x, test_y=test_y,
            reg_row=reg_row,
            ref_crop_fid=ref_crop_fid,
            readout_folder=folder,
            params=params,
        )

        fig = None
        if result.get("status") == "ok" and ref_crop_fid is not None:
            fig = make_fitspot_figure(
                ref_crop_fid=ref_crop_fid,
                fid_crop=result["fid_crop"],
                fid_crop_aligned=result["fid_crop_aligned"],
                dat_maxproj=result["dat_maxproj"],
                x_px=result["x_px"], y_px=result["y_px"], z_px=result["z_px"],
                fov=fov, hyb=hyb,
            )
            # Save figure
            if output_dir is not None:
                fig_dir = output_dir / "FitSpot_test"
                fig_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_dir / f"FitSpot_fov{fov:04d}_h{hyb:04d}.png",
                            dpi=100, bbox_inches="tight")

        # Build row for table
        row = {
            "hybe": hyb,
            "x":   result.get("x", np.nan),
            "y":   result.get("y", np.nan),
            "z":   result.get("z", np.nan),
            "h":   result.get("h", np.nan),
            "h_fit": result.get("h_fit", np.nan),
            "wx":  result.get("wx", np.nan),
            "wy":  result.get("wy", np.nan),
            "wz":  result.get("wz", np.nan),
            "fitQuality": result.get("fitQuality", np.nan),
            "xshift_total": result.get("xshift_total", 0),
            "yshift_total": result.get("yshift_total", 0),
            "status": result.get("status", "missing"),
        }
        rows.append(row)

        yield hyb, n_hybs, result, fig, list(rows)

    # Save full CSV
    if output_dir is not None:
        pd.DataFrame(rows).to_csv(
            output_dir / f"fov{fov:03d}_testSpot_fits.csv", index=False
        )


# ---------------------------------------------------------------------------
# Auto spot detection (Step 6)
# ---------------------------------------------------------------------------

def detect_spots(
    maxproj: np.ndarray,
    threshold_pct: float = 0.997,
    bg_size: int = 50,
    min_dist: int = 30,
    downsample: int = 3,
    border: int = 2,
) -> pd.DataFrame:
    """Detect locus spots in a 2D max-projection image.

    Algorithm (matches ChrTracer3 SpotSelector):
      1. Downsample by `downsample` for speed
      2. Rolling-ball background subtraction (uniform_filter)
      3. Threshold at `threshold_pct` percentile
      4. Find local maxima with min_distance separation
      5. Map coordinates back to full-resolution
      6. Remove spots within `border` pixels of image edge

    Returns DataFrame with columns locusX (col), locusY (row).
    """
    H, W = maxproj.shape

    # 1. Downsample
    img_ds = maxproj[::downsample, ::downsample].astype(float)

    # 2. Background subtract
    bg     = uniform_filter(img_ds, size=max(1, bg_size // downsample))
    img_bg = img_ds - bg
    img_bg = np.clip(img_bg, 0, None)

    # 3. Threshold
    thresh = np.percentile(img_bg, threshold_pct * 100)
    mask   = img_bg > thresh

    # 4. Local maxima: pixel must equal the max in its neighbourhood
    min_dist_ds = max(1, min_dist // downsample)
    footprint   = np.ones((min_dist_ds * 2 + 1, min_dist_ds * 2 + 1))
    local_max   = (img_bg == maximum_filter(img_bg, footprint=footprint)) & mask

    rows_ds, cols_ds = np.where(local_max)

    # 5. Map back to full-res
    rows_fr = rows_ds * downsample
    cols_fr = cols_ds * downsample

    # 6. Border exclusion
    keep = (
        (rows_fr >= border) & (rows_fr < H - border) &
        (cols_fr >= border) & (cols_fr < W - border)
    )
    rows_fr = rows_fr[keep]
    cols_fr = cols_fr[keep]

    return pd.DataFrame({"locusX": cols_fr, "locusY": rows_fr})


# ---------------------------------------------------------------------------
# Fit all validated spots across all hybs for one FOV
# ---------------------------------------------------------------------------

def _fit_hyb_task(args: tuple) -> tuple[int, list]:
    """Top-level picklable worker: fits all spots for one hyb.

    Reads the DAX for that hyb exactly once, crops every spot, fits all.
    This is far cheaper than the old per-spot approach which re-read the
    same DAX file once per spot.

    args = (fov, hyb, folder_str, hw_n, reg_record, spot_list, params)
      spot_list : list of (spot_idx, locus_x, locus_y, ref_crop_fid)
      hw_n      : (H, W, N) image dimensions from the reference .inf

    Returns (hyb, rows).
    """
    fov, hyb, folder_str, hw_n, reg_record, spot_list, params = args
    H, W, N  = hw_n
    folder   = Path(folder_str)
    reg_row  = pd.Series(reg_record)
    box_half = params.get("box_half", 15)
    n_ch     = params.get("n_ch", 2)

    total_dx = int(reg_row.get("xshift", 0) + reg_row.get("xshift2", 0))
    total_dy = int(reg_row.get("yshift", 0) + reg_row.get("yshift2", 0))

    dax_name = f"ConvZscan_{fov:02d}.dax"
    dax_path = folder / dax_name

    rows = []

    if not dax_path.exists():
        for spot_idx, locus_x, locus_y, _ in spot_list:
            rows.append({
                "fov": fov, "spot_id": spot_idx,
                "locus_x": locus_x, "locus_y": locus_y, "hybe": hyb,
                "x": np.nan, "y": np.nan, "z": np.nan, "h": np.nan,
                "wx": np.nan, "wy": np.nan, "wz": np.nan,
                "fitQuality": np.nan,
                "xshift_total": total_dx, "yshift_total": total_dy,
                "status": "missing",
            })
        return hyb, rows

    stack = read_dax(dax_path, H, W, N)

    for spot_idx, locus_x, locus_y, ref_crop_fid in spot_list:
        cx     = locus_x + total_dx
        cy     = locus_y + total_dy
        crops  = crop_volume(stack, cx, cy, box_half, n_ch)
        result = _fit_from_crops(
            fov, hyb, locus_x, locus_y, total_dx, total_dy, cx, cy,
            ref_crop_fid, crops.get(0), crops.get(1), params,
        )
        rows.append({
            "fov":          fov,
            "spot_id":      spot_idx,
            "locus_x":      locus_x,
            "locus_y":      locus_y,
            "hybe":         hyb,
            "x":            result.get("x", np.nan),
            "y":            result.get("y", np.nan),
            "z":            result.get("z", np.nan),
            "h":            result.get("h", np.nan),
            "h_fit":        result.get("h_fit", np.nan),
            "wx":           result.get("wx", np.nan),
            "wy":           result.get("wy", np.nan),
            "wz":           result.get("wz", np.nan),
            "fitQuality":   result.get("fitQuality", np.nan),
            "xshift_total": result.get("xshift_total", 0),
            "yshift_total": result.get("yshift_total", 0),
            "status":       result.get("status", "missing"),
        })

    del stack
    return hyb, rows


def fit_all_spots_stream(
    fov: int,
    spots_df: pd.DataFrame,
    readout_folders: list[Path],
    reg_data: pd.DataFrame,
    params: dict,
    output_dir: Path | None = None,
    n_workers: int = 1,
):
    """Generator: fit every validated spot × every hyb for one FOV.

    Key optimisation: iterates hybs in the outer loop and spots in the inner
    loop so each DAX file is read exactly once per hyb (previously it was read
    once per spot × hyb, i.e. N_spots times more I/O).

    Sequential: yields (spot_idx, n_spots, hyb, n_hybs, result, rows_so_far)
                after every (spot, hyb) pair.
    Parallel  : one worker per hyb (reads DAX once, fits all spots);
                yields (last_spot_idx, n_spots, hyb, n_hybs, last_result, rows_so_far)
                after each hyb completes.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    # Ensure this module is the canonical one in sys.modules so that
    # _fit_hyb_task can be pickled by name.  Streamlit's reloader can
    # create a second copy of the module, breaking ProcessPoolExecutor.
    import sys as _sys
    _sys.modules.setdefault(__name__, _sys.modules[__name__])
    _this = _sys.modules[__name__]
    if _this._fit_hyb_task is not _fit_hyb_task:
        _this._fit_hyb_task = _fit_hyb_task

    n_spots  = len(spots_df)
    n_hybs   = len(readout_folders)
    ref_hyb  = params.get("ref_hyb", 1)
    box_half = params.get("box_half", 15)
    n_ch     = params.get("n_ch", 2)

    # Read reference DAX once to get dimensions and pre-crop reference fiducials
    ref_folder = readout_folders[ref_hyb - 1]
    dax_name   = f"ConvZscan_{fov:02d}.dax"
    inf_name   = f"ConvZscan_{fov:02d}.inf"
    inf        = read_inf(ref_folder / inf_name)
    H, W, N    = inf["height"], inf["width"], inf["n_frames"]
    ref_stack  = read_dax(ref_folder / dax_name, H, W, N)

    spot_list = []
    for s_i, (_, spot_row) in enumerate(spots_df.iterrows(), start=1):
        locus_x      = int(spot_row["locusX"])
        locus_y      = int(spot_row["locusY"])
        ref_crops    = crop_volume(ref_stack, locus_x, locus_y, box_half, n_ch)
        ref_crop_fid = ref_crops.get(0)
        spot_list.append((s_i, locus_x, locus_y, ref_crop_fid))   # 1-based spot id

    del ref_stack

    rows_all = []

    if n_workers == 1:
        # Sequential: hyb-outer, spot-inner — one DAX read per hyb
        for hyb_i, folder in enumerate(readout_folders):
            hyb      = hyb_i + 1
            reg_row  = reg_data.iloc[hyb_i] if hyb_i < len(reg_data) else pd.Series(
                {"xshift": 0, "yshift": 0, "xshift2": 0, "yshift2": 0})
            total_dx = int(reg_row.get("xshift", 0) + reg_row.get("xshift2", 0))
            total_dy = int(reg_row.get("yshift", 0) + reg_row.get("yshift2", 0))

            dax_path = folder / dax_name
            stack    = read_dax(dax_path, H, W, N) if dax_path.exists() else None

            for spot_idx, locus_x, locus_y, ref_crop_fid in spot_list:
                if stack is None:
                    result = {
                        "hybe": hyb, "status": "missing",
                        "xshift_total": total_dx, "yshift_total": total_dy,
                    }
                else:
                    cx     = locus_x + total_dx
                    cy     = locus_y + total_dy
                    crops  = crop_volume(stack, cx, cy, box_half, n_ch)
                    result = _fit_from_crops(
                        fov, hyb, locus_x, locus_y, total_dx, total_dy, cx, cy,
                        ref_crop_fid, crops.get(0), crops.get(1), params,
                    )
                rows_all.append({
                    "fov": fov, "spot_id": spot_idx,
                    "locus_x": locus_x, "locus_y": locus_y, "hybe": hyb,
                    "x":            result.get("x", np.nan),
                    "y":            result.get("y", np.nan),
                    "z":            result.get("z", np.nan),
                    "h":            result.get("h", np.nan),
                    "h_fit":        result.get("h_fit", np.nan),
                    "wx":           result.get("wx", np.nan),
                    "wy":           result.get("wy", np.nan),
                    "wz":           result.get("wz", np.nan),
                    "fitQuality":   result.get("fitQuality", np.nan),
                    "xshift_total": result.get("xshift_total", 0),
                    "yshift_total": result.get("yshift_total", 0),
                    "status":       result.get("status", "missing"),
                })
                yield spot_idx, n_spots, hyb, n_hybs, result, list(rows_all)

            if stack is not None:
                del stack

    else:
        # Parallel: one task per hyb — each worker reads its DAX once
        hw_n        = (H, W, N)
        reg_records = reg_data.to_dict("records")
        tasks = [
            (fov, hyb_i + 1, str(folder), hw_n,
             reg_records[hyb_i] if hyb_i < len(reg_records)
             else {"xshift": 0, "yshift": 0, "xshift2": 0, "yshift2": 0},
             spot_list, params)
            for hyb_i, folder in enumerate(readout_folders)
        ]
        with ProcessPoolExecutor(max_workers=n_workers,
                                  initializer=_worker_init,
                                  initargs=(_app_dir,)) as executor:
            futures = {executor.submit(_fit_hyb_task, t): t[1] for t in tasks}
            for future in as_completed(futures):
                hyb, hyb_rows = future.result()
                rows_all.extend(hyb_rows)
                last = hyb_rows[-1] if hyb_rows else {}
                last_result = {k: last.get(k, np.nan)
                               for k in ("x", "y", "z", "h", "fitQuality", "status")}
                yield (n_spots - 1), n_spots, hyb, n_hybs, last_result, list(rows_all)

    # Save per-FOV CSV — sorted by fov, spot_id, hybe
    if output_dir is not None and rows_all:
        (pd.DataFrame(rows_all)
           .sort_values(["fov", "spot_id", "hybe"])
           .reset_index(drop=True)
           .to_csv(output_dir / f"fov{fov:03d}_allFits.csv", index=False))


def detect_spots_overlay(
    maxproj: np.ndarray,
    spots_df: pd.DataFrame,
    ds: int = 4,
) -> np.ndarray:
    """Render a uint8 RGB image with detected spots marked as red circles.

    Returns (H//ds, W//ds, 3) uint8 array.
    """
    H, W = maxproj.shape
    lo, hi = np.percentile(maxproj, 0.5), np.percentile(maxproj, 99.5)
    disp   = np.clip((maxproj - lo) / (hi - lo + 1e-10), 0, 1)
    disp_ds = (disp[::ds, ::ds] * 255).astype(np.uint8)
    rgb     = np.stack([disp_ds, disp_ds, disp_ds], axis=-1)

    r_cross = 3   # half-size of cross marker in display pixels
    for _, row in spots_df.iterrows():
        sx = int(row["locusX"]) // ds
        sy = int(row["locusY"]) // ds
        sx = np.clip(sx, r_cross, rgb.shape[1] - r_cross - 1)
        sy = np.clip(sy, r_cross, rgb.shape[0] - r_cross - 1)
        rgb[sy - r_cross:sy + r_cross + 1, sx, :] = [255, 50, 50]
        rgb[sy, sx - r_cross:sx + r_cross + 1, :] = [255, 50, 50]

    return rgb
