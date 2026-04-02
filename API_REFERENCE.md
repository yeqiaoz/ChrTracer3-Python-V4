# API Reference — `orca_drift.py` and `orca_fit.py`

Both modules are importable independently of the Streamlit app and can be used from scripts.
See also `compare_result_3D/run_fitting.py` in the analysis directory for a standalone batch-fitting example.

---

## `orca_drift.py`

Drift correction for ORCA/ChrTracer3 data.

---

### `read_inf(inf_path: Path) → dict`

Parse a ChrTracer3 `.inf` metadata file.

**Returns:** dict with keys `height`, `width`, `n_frames` (plus all raw key=value pairs from the file).

```python
from orca_drift import read_inf
info = read_inf(Path("Readout_001/ConvZscan_01.inf"))
# info["height"] = 1843, info["width"] = 1843, info["n_frames"] = 26
```

---

### `read_dax(dax_path, height, width, n_frames) → np.ndarray`

Read a `.dax` binary stack.

**Returns:** `(n_frames, height, width)` uint16 array.

Frames are stored in Fortran (column-major) order `(H, W, N)`; this function transposes to C order `(N, H, W)`.

```python
from orca_drift import read_inf, read_dax
info  = read_inf(path.with_suffix(".inf"))
stack = read_dax(path, info["height"], info["width"], info["n_frames"])
# stack.shape = (n_frames, H, W)
```

---

### `fiducial_maxproj(stack, fid_ch=0, n_ch=2) → np.ndarray`

Extract fiducial-channel frames (every `n_ch`-th frame starting at `fid_ch`) and max-project along Z.

**Returns:** `(H, W)` float32.

---

### `coarse_shift(ref_proj, mov_proj, ds=4) → (dy, dx, xcorr)`

Compute a coarse integer-pixel shift from downsampled cross-correlation.

- Downsample both projections by `ds`
- Cross-correlate; find peak
- Return `(dy, dx)` in full-resolution pixels and the low-res xcorr map

---

### `fine_shift(ref_proj, mov_proj, coarse_dy, coarse_dx, crop=150, spot_percentile=75) → (dy2, dx2, xcorr_zoom)`

Integer fine-shift correction.

- Apply coarse correction to `mov_proj`
- Crop 150×150 px around the `spot_percentile`-brightness bead (avoids hot-pixel artefacts)
- Cross-correlate at full resolution

---

### `correct_one_fov_stream(fov, readout_folders, output_dir, ...) → generator`

Streaming per-hyb drift correction for one FOV. Yields after each hyb:

```python
(hyb, total_hybs, row_dict, fig_path_or_None, rows_so_far)
```

Saves `fov{N:03d}_regData.csv` and `CorrAlign/CorrAlign_fov{N:04d}_h{M:04d}.png` on completion.

**Parameters:**

| Name | Default | Description |
|------|---------|-------------|
| `fov` | — | FOV index (1-based) |
| `readout_folders` | — | List of `Readout_*` Path objects in hyb order |
| `output_dir` | — | Where to write CSV and figures |
| `ref_hyb` | 1 | Which hyb is the reference (shift = 0) |
| `fid_ch` | 0 | Fiducial channel index within interleaved frames |
| `n_ch` | 2 | Number of channels (fiducial + readout) |
| `ds` | 4 | Downsampling factor for coarse correlation |
| `crop` | 150 | Half-crop size (px) for fine correlation |
| `spot_percentile` | 75 | Brightness percentile for bead selection |
| `max_fine_shift` | 5 | Discard fine shift if `|shift| > this` |
| `save_figures` | True | Write CorrAlign PNGs |

---

### `correct_one_fov(fov, readout_folders, output_dir, ...) → pd.DataFrame`

Non-streaming wrapper: runs to completion and returns the regData DataFrame.

---

### `convert_one_task(args: tuple) → tuple`

Module-level worker for Step 0 `ProcessPoolExecutor`. Converts one (location, timepoint) to a `.dax` file.

**Input tuple:** `(loc_name, loc_num, readout_num, interleaved_idx, dat_file_strs, frames_per_file_list, H, W, frame_bytes, out_path_str, write_tiff)`

**Returns:** `(loc_name, readout_num, dax_path_str, n_frames, error_or_None)`

---

## `orca_fit.py`

Spot detection and 3D Gaussian fitting.

---

### `fit_gaussian_3d(vol) → tuple`

Fit a full 3D Gaussian to a small sub-crop.

**Input:** `vol` — `(nZ, H, W)` float/uint16 array (typically 13×9×9 voxels)

**Returns:** `(x0, y0, sx, sy, z0, sz, h, bg, rmse, resnorm)`

| Return | Description |
|--------|-------------|
| `x0, y0` | Fitted centre position (column, row) within `vol` |
| `z0` | Fitted Z position (slice index) within `vol` |
| `sx, sy` | XY sigma in pixels (standard convention: `σ_python = √2 × σ_matlab`) |
| `sz` | Z sigma in slices |
| `h` | Fitted amplitude (Matlab `a`) |
| `bg` | Fitted background |
| `rmse` | Root-mean-square residual |
| `resnorm` | `sum((vol - fit)²)` — same as Matlab `lsqnonlin` resnorm; use for `fitQuality` |

Optimizer bounds: `sx, sy ∈ [0.3, 4.0]` px; `sz ∈ [0.3, 6.0]` slices.
Falls back to initial-guess values on `curve_fit` failure (no exception raised).

---

### `fit_gaussian_2d(img) → tuple`

Fit 2D Gaussian to `(H, W)` image. **Used internally; V3 uses `fit_gaussian_3d` instead.**

**Returns:** `(x0, y0, sx, sy, h, bg, residual_rms)`

---

### `fit_gaussian_1d(profile) → tuple`

Fit 1D Gaussian to a 1D array. **Legacy; not used by V3.**

**Returns:** `(center, sigma, height, background)`

---

### `crop_volume(stack, cx, cy, half, n_ch=2) → dict[int, np.ndarray]`

Crop a ±`half`-pixel box around `(cx, cy)` for each channel.

**Input:** `stack` — `(N_frames, H, W)` uint16

**Returns:** `{ch_index: (nZ, crop_H, crop_W)}` per channel

---

### `fine_align_crop(ref_crop, mov_crop, upsample=4) → (dy, dx, xcorr_peak)`

Sub-pixel XY shift via phase cross-correlation on upsampled max-projections of 3D crops.

Used inside the fitting loop to align the readout crop to the reference fiducial.

---

### `detect_spots(maxproj, threshold_pct=0.997, bg_size=50, min_dist=30, downsample=3, border=2) → pd.DataFrame`

Detect locus spots in a 2D fiducial max-projection.

**Algorithm:**
1. Downsample × `downsample`
2. Rolling-ball background subtraction (uniform filter of size `bg_size // downsample`)
3. Threshold at `threshold_pct` percentile
4. Find local maxima (pixel = neighbourhood max within `min_dist` px)
5. Map back to full resolution; remove spots within `border` px of image edge

**Returns:** DataFrame with `locusX` (column) and `locusY` (row) in full-resolution pixels.

> **Border note:** Set `border ≥ box_half` (default 15) to prevent crop extraction from going out of bounds during fitting.

---

### `fit_one_hyb(fov, hyb, test_x, test_y, reg_row, ref_crop_fid, readout_folder, params) → dict`

Fit one hyb round for a single test locus. Reads the `.dax` file internally.

Used by Step 5 (test spot) and the streaming test-spot generator. For batch fitting, use `fit_all_spots_stream` (which uses `_fit_from_crops` to avoid redundant disk reads).

**Key `params` keys:** same as quality filter parameters listed in README.

---

### `fit_all_spots_stream(fov, spots_df, readout_folders, reg_data, params, output_dir, n_workers) → generator`

Batch-fit all spots in `spots_df` across all hybs in `readout_folders`.

Yields after each spot (not each hyb) for progress reporting:
```python
(*_, rows_so_far)   # rows_so_far grows as spots complete
```

Saves `fov{N:03d}_allFits.csv` to `output_dir` on completion.

**Important:** Uses `ProcessPoolExecutor` with `n_workers` processes. On network-mounted data with many `.dax` files per folder, reduce `n_workers` to 1 to avoid I/O contention (symptom: silent hang after a few minutes).

---

### `make_fitspot_figure(ref_crop_fid, fid_crop, fid_crop_aligned, dat_maxproj, x_px, y_px, z_px, fov, hyb) → plt.Figure`

4-panel diagnostic figure showing fiducial alignment before/after correction (XY and XZ views), used in Step 5.

---

### `_check_fit_quality(h_raw, h_fit, bg, x0, y0, wx_nm, wy_nm, wz_nm, nm_xy, nm_z, params) → str`

Internal quality filter. Returns `'ok'` or a rejection reason string.

See the **Rejection status codes** table in README.md for all possible values.

---

## Standalone batch-fitting script

See `compare_result_3D/run_fitting.py` for a command-line script that re-fits already-detected spots with different parameter sets:

```bash
cd compare_result_3D

# Baseline (default params)
python3 run_fitting.py --attempt 0 --n_workers 8 --fovs 1-20

# Loosened step filter
python3 run_fitting.py --attempt 2 --max_xy_step 15 --n_workers 8 --fovs 1-5

# Loosened width filter
python3 run_fitting.py --attempt 3 --max_wx_px 4.0 --n_workers 8 --fovs 1-5
```

Compare results with:
```bash
python3 compare_attempt.py --attempt 2
```
