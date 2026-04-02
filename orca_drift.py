"""
orca_drift.py — Drift correction for ORCA/ChrTracer3 data.

Pipeline per FOV:
  1. Read fiducial max-projection from reference hyb (Readout_001)
  2. For each subsequent hyb:
     a. Coarse shift  — downsample × ds_factor, cross-correlate, find integer peak
     b. Fine shift    — apply coarse, full-res cross-correlate in a cropped window
  3. Save fov{N:03d}_regData.csv
  4. Save CorrAlign/CorrAlign_fov{N:04d}_h{M:04d}.png

Output columns match ChrTracer3 Matlab:
  xshift, yshift, theta, rescale, xshift2, yshift2, theta2, rescale2
  (theta/rescale always 0/1; xshift=coarse, xshift2=fine)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DAX I/O
# ---------------------------------------------------------------------------

def read_inf(inf_path: Path) -> dict:
    """Parse a .inf metadata file → dict with height, width, n_frames."""
    info = {}
    for line in inf_path.read_text().splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            info[key.strip()] = val.strip()
    # frame dimensions = W x H
    dims = info.get("frame dimensions", "").split("x")
    if len(dims) == 2:
        info["width"]  = int(dims[0].strip())
        info["height"] = int(dims[1].strip())
    info["n_frames"] = int(info.get("number of frames", 0))
    return info


def read_dax(dax_path: Path, height: int, width: int, n_frames: int) -> np.ndarray:
    """Read .dax file → (n_frames, height, width) uint16.

    Standard HAL/ChrTracer3 format: frames stored sequentially in C order
    (row-major within each frame). Compatible with Matlab ChrTracer3.
    """
    raw = np.fromfile(dax_path, dtype="<u2")
    return raw.reshape((n_frames, height, width))


def fiducial_maxproj(stack: np.ndarray, fid_ch: int = 0, n_ch: int = 2) -> np.ndarray:
    """Extract fiducial channel frames and max-project along Z."""
    fid_frames = stack[fid_ch::n_ch]   # e.g. frames 0,2,4,...
    return fid_frames.max(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Cross-correlation helpers
# ---------------------------------------------------------------------------

def _corr_align_rotate_scale(im1: np.ndarray, im2: np.ndarray,
                             max_shift: float = np.inf,
                             grad_max: bool = True) -> tuple[float, float, float, np.ndarray]:
    """Match Matlab CorrAlignRotateScale (translation only, no rotation/scale).

    Normalizes images (subtract mean, divide by std), computes cross-correlation,
    and finds peak using either gradient maximum (second derivative minimum,
    matching Matlab gradMax=true) or simple argmax.

    Returns (yshift, xshift, corrPeak, corrMap).
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # Normalize (matches Matlab: subtract mean, divide by std)
    s1 = im1.std()
    s2 = im2.std()
    if s1 > 0:
        im1 = (im1 - im1.mean()) / s1
    if s2 > 0:
        im2 = (im2 - im2.mean()) / s2

    H, W = im1.shape

    # Limit search region
    Hc = int(min(H, max_shift))
    Wc = int(min(W, max_shift))
    Hc2 = round(Hc / 2)
    Wc2 = round(Wc / 2)

    # FFT cross-correlation: ifft(conj(F1)*F2), peak at displacement d
    # No zero-padding — circular wrap is negligible for small shifts on small images
    F1 = np.fft.rfft2(im1)
    F2 = np.fft.rfft2(im2)
    corrM = np.fft.fftshift(np.fft.irfft2(np.conj(F1) * F2))
    del F1, F2

    # Extract center portion around zero-shift (H//2, W//2) after fftshift
    zr = H // 2
    zc = W // 2
    r0 = zr - Hc2
    r1 = zr + Hc2
    c0 = zc - Wc2
    c1 = zc + Wc2
    corrMmini = corrM[r0:r1, c0:c1]

    if grad_max and corrMmini.size > 4:
        # Second derivative (Laplacian) — find minimum = sharpest peak
        gy, gx = np.gradient(corrMmini)
        ddx = np.gradient(gx, axis=1)
        ddy = np.gradient(gy, axis=0)
        laplacian = ddx + ddy
        indmin = np.argmin(laplacian)
        cy, cx = np.unravel_index(indmin, corrMmini.shape)
        corr_peak = -laplacian[cy, cx]
    else:
        indmax = np.argmax(corrMmini)
        cy, cx = np.unravel_index(indmax, corrMmini.shape)
        corr_peak = corrMmini[cy, cx]

    # Zero-shift is at local index Hc2 (since zr - r0 = Hc2)
    # conj(F1)*F2 convention: peak at displacement d (im2 shifted by d from im1)
    xshift = cx - Wc2
    yshift = cy - Hc2
    return float(yshift), float(xshift), float(corr_peak), corrMmini


def coarse_shift(ref_proj: np.ndarray, mov_proj: np.ndarray,
                 ds: int = 4, max_size: int = 400) -> tuple[int, int, np.ndarray]:
    """Integer-pixel shift from downsampled cross-correlation.

    Matches Matlab CorrAlignFast coarse step:
      - Compute relSpeed = sqrt(H*W)/maxSize for downsampling
      - Bilinear downsampling via affine warp
      - Normalized cross-correlation with gradMax peak finding

    Returns (dy, dx, xcorr_map).
    """
    from scipy.ndimage import zoom as nd_zoom

    H, W = ref_proj.shape
    rel_speed = np.sqrt(H * W) / max_size

    if rel_speed > 1:
        scale = 1.0 / rel_speed
        out_h = round(H * scale)
        out_w = round(W * scale)
        ref_ds = nd_zoom(ref_proj.astype(np.float32), (out_h / H, out_w / W), order=1)
        mov_ds = nd_zoom(mov_proj.astype(np.float32), (out_h / H, out_w / W), order=1)
    else:
        rel_speed = 1.0
        ref_ds = ref_proj.astype(np.float32)
        mov_ds = mov_proj.astype(np.float32)

    yshift, xshift, corr_peak, corrMap = _corr_align_rotate_scale(
        ref_ds, mov_ds, max_shift=np.inf, grad_max=False)

    dy = int(round(rel_speed * yshift))
    dx = int(round(rel_speed * xshift))
    return dy, dx, corrMap


def _apply_shift_2d(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Translate a 2D image by (dy, dx) using ScaleRotateShift convention."""
    from scipy.ndimage import shift as nd_shift
    return nd_shift(img, (-dy, -dx), order=1, mode="constant", cval=0)


def fine_shift(ref_proj: np.ndarray, mov_proj: np.ndarray,
               coarse_dy: int, coarse_dx: int,
               crop: int = 200, max_size: int = 400,
               fine_max_shift: int = 30,
               spot_percentile: float = 75) -> tuple[int, int, np.ndarray]:
    """Fine-shift matching Matlab CorrAlignFast fine step.

    - Apply coarse correction to mov
    - Find brightest region in downsampled product of im1 * im2_coarse_aligned
    - Crop fineBox around that region
    - Run CorrAlignRotateScale with gradMax on the cropped region

    Returns (dy2, dx2, xcorr_zoom).
    """
    from scipy.ndimage import zoom as nd_zoom

    H, W = ref_proj.shape
    rel_speed = np.sqrt(H * W) / max_size

    # Apply coarse correction
    mov_shifted = _apply_shift_2d(mov_proj.astype(np.float32), coarse_dy, coarse_dx)

    # Find region with most overlapping signal (Matlab approach)
    subpix = 5
    speed_scale = max(0.01, 1.0 / (2 * rel_speed * subpix))
    ref_small = nd_zoom(ref_proj.astype(np.float32), speed_scale, order=1)
    mov_small = nd_zoom(mov_shifted.astype(np.float64), speed_scale, order=1)
    product = ref_small * mov_small
    pk_prod = np.unravel_index(np.argmax(product), product.shape)
    cy = int(round(pk_prod[0] / speed_scale))
    cx = int(round(pk_prod[1] / speed_scale))
    del ref_small, mov_small, product

    # Crop fineBox around the product peak (Matlab: fineBox = round(maxSize/2))
    fine_box = crop
    x1 = max(0, cx - fine_box)
    x2 = min(W, cx + fine_box)
    y1 = max(0, cy - fine_box)
    y2 = min(H, cy + fine_box)

    ref_crop = ref_proj[y1:y2, x1:x2]
    mov_crop = mov_shifted[y1:y2, x1:x2]
    del mov_shifted

    # Fine correlation with gradMax (Matlab: maxShift = round(relSpeed) + fineMaxShift)
    fine_max = round(rel_speed) + fine_max_shift
    yshift, xshift, corr_peak, corrMap = _corr_align_rotate_scale(
        ref_crop, mov_crop, max_shift=fine_max, grad_max=True)

    dy2 = int(round(yshift))
    dx2 = int(round(xshift))
    return dy2, dx2, corrMap


# ---------------------------------------------------------------------------
# CorrAlign figure
# ---------------------------------------------------------------------------

def _overlay_rgb(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Blend ref (red) and mov (cyan) into an RGB image for display.

    Both images are normalized on the same scale (derived from ref) so that
    zero-padded borders introduced by nd_shift don't inflate the mov baseline.
    """
    lo, hi = np.percentile(ref, 0.5), np.percentile(ref, 99.5)
    def _norm(a):
        return np.clip((a - lo) / (hi - lo + 1e-10), 0, 1).astype(np.float32)

    r   = _norm(ref)
    m   = _norm(mov)
    rgb = np.stack([r, m, m], axis=-1)   # red + cyan overlay
    return rgb


def save_corralign_figure(
    ref_proj: np.ndarray,
    mov_proj: np.ndarray,
    coarse_dy: int, coarse_dx: int,
    fine_dy: int,   fine_dx: int,
    xcorr_zoom: np.ndarray,
    fov: int, hyb: int,
    out_path: Path,
) -> None:
    """Save a 3-panel CorrAlign PNG matching ChrTracer3 Matlab output."""
    from scipy.ndimage import shift as nd_shift

    mov_coarse = nd_shift(mov_proj, (-coarse_dy, -coarse_dx), order=1,
                          mode="constant", cval=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"fov{fov} h1-red h{hyb}-cyan", fontsize=11, y=1.01
    )

    # Panel 1: original overlay
    axes[0].imshow(_overlay_rgb(ref_proj, mov_proj), origin="upper")
    axes[0].set_title("original")

    # Panel 2: after coarse correction
    axes[1].imshow(_overlay_rgb(ref_proj, mov_coarse), origin="upper")
    axes[1].set_title(
        f"coarse correction  xshift={coarse_dx} yshift={coarse_dy}"
    )

    # Panel 3: zoomed xcorr map (fine)
    axes[2].imshow(xcorr_zoom, origin="upper", cmap="hot")
    axes[2].set_title(
        f"fine xcorr  xshift2={fine_dx} yshift2={fine_dy}"
    )

    for ax in axes:
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-FOV drift correction — streaming (yields one hyb at a time)
# ---------------------------------------------------------------------------

def correct_one_fov_stream(
    fov: int,
    readout_folders: list[Path],
    output_dir: Path,
    ref_hyb: int = 1,
    fid_ch: int = 0,
    n_ch: int = 2,
    ds: int = 4,
    crop: int = 150,
    spot_percentile: float = 75,
    max_fine_shift: int = 5,
    save_figures: bool = True,
):
    """Generator: process one hyb per iteration, yielding progress info.

    Yields after each hyb:
        (hyb, total_hybs, row_dict, fig_path_or_None, rows_so_far)

    Saves fov{N:03d}_regData.csv and CorrAlign PNGs to output_dir.
    """
    corralign_dir = output_dir / "CorrAlign"
    dax_name      = f"ConvZscan_{fov:02d}.dax"
    inf_name      = f"ConvZscan_{fov:02d}.inf"
    n_hybs        = len(readout_folders)

    # Read reference
    ref_folder = readout_folders[ref_hyb - 1]
    inf        = read_inf(ref_folder / inf_name)
    H, W, N    = inf["height"], inf["width"], inf["n_frames"]

    ref_stack  = read_dax(ref_folder / dax_name, H, W, N)
    ref_proj   = fiducial_maxproj(ref_stack, fid_ch, n_ch)
    del ref_stack

    rows = []

    for i, folder in enumerate(readout_folders):
        hyb      = i + 1
        fig_path = None

        if hyb == ref_hyb:
            row = {"xshift": 0, "yshift": 0, "theta": 0, "rescale": 1,
                   "xshift2": 0, "yshift2": 0, "theta2": 0, "rescale2": 1}
            rows.append(row)
            yield hyb, n_hybs, row, fig_path, list(rows)
            continue

        dax_path = folder / dax_name
        if not dax_path.exists():
            row = {"xshift": np.nan, "yshift": np.nan, "theta": 0, "rescale": 1,
                   "xshift2": np.nan, "yshift2": np.nan, "theta2": 0, "rescale2": 1}
            rows.append(row)
            yield hyb, n_hybs, row, fig_path, list(rows)
            continue

        curr_stack = read_dax(dax_path, H, W, N)
        curr_proj  = fiducial_maxproj(curr_stack, fid_ch, n_ch)
        del curr_stack

        dy,  dx,  _coarse_map = coarse_shift(ref_proj, curr_proj)
        del _coarse_map
        dy2, dx2, xcorr_zoom = fine_shift(ref_proj, curr_proj, dy, dx,
                                           crop=crop)

        # If fine shift is implausibly large it found a false xcorr peak;
        # discard it and rely on coarse only.
        if abs(dx2) > max_fine_shift or abs(dy2) > max_fine_shift:
            dx2, dy2 = 0, 0

        row = {"xshift": dx, "yshift": dy, "theta": 0, "rescale": 1,
               "xshift2": dx2, "yshift2": dy2, "theta2": 0, "rescale2": 1}
        rows.append(row)

        if save_figures:
            fig_path = corralign_dir / f"CorrAlign_fov{fov:04d}_h{hyb:04d}.png"
            save_corralign_figure(
                ref_proj, curr_proj,
                dy, dx, dy2, dx2, xcorr_zoom,
                fov, hyb, fig_path,
            )

        del curr_proj, xcorr_zoom
        import gc; gc.collect()

        yield hyb, n_hybs, row, fig_path, list(rows)

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"fov{fov:03d}_regData.csv", index=False)


def correct_one_fov(fov, readout_folders, output_dir, ref_hyb=1,
                    fid_ch=0, n_ch=2, ds=4, crop=150, spot_percentile=75,
                    max_fine_shift=5, save_figures=True) -> pd.DataFrame:
    """Non-streaming wrapper — runs to completion and returns DataFrame."""
    rows = []
    for hyb, n_hybs, row, fig_path, _ in correct_one_fov_stream(
        fov, readout_folders, output_dir,
        ref_hyb=ref_hyb, fid_ch=fid_ch, n_ch=n_ch,
        ds=ds, crop=crop, spot_percentile=spot_percentile,
        max_fine_shift=max_fine_shift, save_figures=save_figures,
    ):
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 0 conversion: one sequential dat-file pass per location
# ---------------------------------------------------------------------------

def convert_one_location(loc_tasks: list) -> list:
    """Convert all readouts for one location in a single sequential dat-file pass.

    All tasks must share the same dat_file_strs / frames_per_file_list / H / W
    (i.e. come from the same Vutara location directory).

    Each task tuple:
        (loc_name, loc_num, readout_num, interleaved_idx,
         dat_file_strs, frames_per_file_list, H, W, frame_bytes,
         out_path_str, write_tiff)

    Strategy
    --------
    1. Build a map  global_frame_idx → [(task_idx, stack_position)]
       covering every frame needed by any readout of this location.
    2. Walk each .dat file forward, frame by frame.  Copy needed frames
       directly into pre-allocated stacks; skip unneeded frames with a
       single forward seek (no backward seeking, no re-opening files).
    3. Write every DAX (and optional TIFF) after the read pass completes.

    Returns list of (loc_name, readout_num, dax_path_str, n_frames, error_or_None)
    in the same order as loc_tasks.
    """
    if not loc_tasks:
        return []

    (loc_name, loc_num, _, _,
     dat_file_strs, frames_per_file_list, H, W, _,
     out_path_str, write_tiff) = loc_tasks[0]

    dat_files   = [Path(p) for p in dat_file_strs]
    out_path    = Path(out_path_str)
    frame_bytes = H * W * 2

    # Map global_idx → [(task_idx, position_in_stack)]
    frame_to_slots: dict = {}
    for t_idx, task in enumerate(loc_tasks):
        for pos, gidx in enumerate(task[3]):   # task[3] = interleaved_idx
            frame_to_slots.setdefault(gidx, []).append((t_idx, pos))

    # Pre-allocate output stacks
    stacks = [np.empty((len(task[3]), H, W), dtype="<u2") for task in loc_tasks]

    # ── Single sequential forward pass through every dat file ────────────────
    global_offset = 0
    for dat_file, fpf in zip(dat_files, frames_per_file_list):
        with open(dat_file, "rb") as fh:
            for local_idx in range(fpf):
                gidx = global_offset + local_idx
                if gidx in frame_to_slots:
                    raw   = np.frombuffer(fh.read(frame_bytes), dtype="<u2")
                    frame = raw.reshape(H, W)
                    for t_idx, pos in frame_to_slots[gidx]:
                        stacks[t_idx][pos] = frame
                else:
                    fh.seek(frame_bytes, 1)   # forward skip only
        global_offset += fpf

    # ── Write DAX (and optionally TIFF) for each readout ─────────────────────
    results = []
    for t_idx, task in enumerate(loc_tasks):
        (loc_name, loc_num, readout_num, interleaved_idx,
         _, _, H, W, _, out_path_str, write_tiff) = task
        try:
            dax_path_out = out_path / f"Readout_{readout_num:03d}" / f"ConvZscan_{loc_num:02d}.dax"
            dax_path_out.parent.mkdir(parents=True, exist_ok=True)

            stack = stacks[t_idx]
            stack.astype("<u2").tofile(dax_path_out)

            inf_text = (
                f"information file for\n\nmachine name = matlab-storm\n"
                f"data_type = 16 bit integers (binary, little endian)\n"
                f"frame dimensions = {W} x {H}\n"
                f"frame size = {H * W}\n"
                f"number of frames = {len(interleaved_idx)}\n"
                f"hstart=1\nhend={H}\nvstart=1\nvend={W}\n"
            )
            dax_path_out.with_suffix(".inf").write_text(inf_text)

            if write_tiff:
                import tifffile
                tiff_dir = out_path / "TIFFs" / loc_name
                tiff_dir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    tiff_dir / f"output_T{readout_num:03d}.tif",
                    stack, photometric="minisblack",
                )

            results.append((loc_name, readout_num, str(dax_path_out), len(interleaved_idx), None))
        except Exception as exc:
            results.append((loc_name, readout_num, "", 0, str(exc)))

    return results
