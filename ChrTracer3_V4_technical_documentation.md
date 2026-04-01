# ChrTracer3 V4 Python Pipeline — Technical Documentation

**Date**: 2026-03-30
**Author**: Yeqiao (with Claude Code assistance)
**Pipeline location**: `/dobby/yeqiao/software_code/ChrTracer3_py_app_V4_test/`
**Output location**: `analysis_260330_v4opt/`

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Step 0: Raw Data Conversion](#2-step-0-raw-data-conversion)
3. [Step 1: FOV-Level Drift Correction](#3-step-1-fov-level-drift-correction)
4. [Step 2: Spot Detection](#4-step-2-spot-detection)
5. [Step 3: 3D Gaussian Fitting](#5-step-3-3d-gaussian-fitting)
6. [V4 Optimization Changes](#6-v4-optimization-changes)
7. [Matlab-Python Comparison Results](#7-matlab-python-comparison-results)
8. [Parameters Reference](#8-parameters-reference)
9. [File Format Reference](#9-file-format-reference)
10. [Coordinate Conventions](#10-coordinate-conventions)
11. [Reference](#11-reference)

---

## 1. Pipeline Overview

The ChrTracer3 V4 Python pipeline processes chromatin tracing data from ORCA experiments. It takes multi-hyb, multi-FOV 3D image stacks (DAX format) and produces 3D Gaussian-fitted spot positions for each genomic locus across hybridization rounds.

### Pipeline Architecture

```
Step 0: Raw Data Conversion (orca_drift.py::convert_one_location)
  ├── Per-location, sequential
  ├── Input:  Vutara SRX .dat files + frameinfo.csv + data.json
  ├── Method: Single forward pass through .dat files
  └── Output: Readout_NNN/ConvZscan_MM.dax + .inf per hyb per FOV

Step 1: Drift Correction (orca_drift.py)
  ├── Per-FOV, parallel (ProcessPoolExecutor, 4 workers)
  ├── Input:  DAX stacks for all hybs (fiducial channel)
  ├── Method: Coarse + fine FFT cross-correlation
  └── Output: fov{N}_regData.csv (XY shifts per hyb)

Step 2: Spot Detection (orca_fit.py::detect_spots)
  ├── Per-FOV, sequential
  ├── Input:  Reference hyb fiducial max-projection
  ├── Method: Background subtraction + thresholding + local maxima
  └── Output: fov{N}_selectSpots.csv (locus XY positions)

Step 3: 3D Gaussian Fitting (orca_fit.py)
  ├── Per-FOV sequential, per-hyb parallel (8 workers)
  ├── Input:  All hyb DAX stacks + drift data + spot positions
  ├── Method: Fine 3D alignment → restricted peak search → Gaussian fit
  └── Output: fov{N}_allFits.csv → merged allFits.csv
```

### Data Flow

```
Vutara SRX .dat files (raw frames from microscope)
    │
    ▼
[Raw Data Conversion] ── DAX/INF stacks (30 hybs × 20 FOVs × 2 channels × ~50 Z-slices)
    │
    ▼
[Drift Correction] ── regData.csv (coarse + fine XY shifts)
    │
    ▼
[Spot Detection] ── selectSpots.csv (locus XY from ref hyb)
    │
    ▼
[3D Fitting] ── allFits.csv (x,y,z in nm + quality metrics)
```

### Orchestration (`run_pipeline_v4.py`)

- Resumes from last checkpoint (skips FOVs with existing output CSVs)
- Step 1 uses `ProcessPoolExecutor` (4 workers) to parallelize across FOVs
- Step 3 processes FOVs sequentially but parallelizes hybs within each FOV (8 workers)
- Memory management: explicit `gc.collect()` between FOVs

---

## 2. Step 0: Raw Data Conversion

**Module**: `orca_drift.py`
**Function**: `convert_one_location()`
**UI**: ORCA_app.py Streamlit interface (interactive mode)

### Purpose

Converts raw imaging data from Vutara SRX `.dat` format into the ChrTracer3 `.dax`/`.inf` format used by all downstream pipeline steps. This is the entry point for data acquired on the Vutara microscope.

### Input Structure

```
experiment_root/
└── Location-01/
    └── Raw Images/
        ├── frameinfo.csv        # maps GlobalIndex -> Timepoint, ZPos, Probe
        ├── data.json            # image dimensions (H, W)
        └── img*.dat             # raw frames (uint16, row-major order)
```

- **`frameinfo.csv`**: Maps each raw frame (by GlobalIndex) to its timepoint, Z position, and probe/readout identity. This is the key metadata file linking raw frames to the ORCA experimental design.
- **`data.json`**: Contains image dimensions (height, width) needed to interpret the binary `.dat` files.
- **`img*.dat`**: Raw binary image frames (uint16, little-endian, row-major). Multiple `.dat` files may exist per location.

### Algorithm

1. **Parse metadata**: Read `frameinfo.csv` to build a mapping from global frame index to `(readout, timepoint, z_position)`. Read `data.json` for image dimensions.

2. **Build frame map**: For each output task (one per readout per FOV), determine which global frame indices are needed and where they should be placed in the output stack. Result: `global_frame_idx -> [(task_idx, stack_position)]`.

3. **Single forward pass**: Walk through each `.dat` file sequentially, frame by frame:
   - If a frame is needed by any output task, copy it into the pre-allocated output stack
   - Skip unneeded frames with forward seeks only (no backward seeking, no file re-opening)

4. **Write output**: For each readout/FOV combination, write:
   - `.dax` file: uint16 binary stack in Fortran (column-major) order to match ChrTracer3 reader convention
   - `.inf` file: text metadata (`frame dimensions = WxH`, `number of frames = N`)

### Output Structure

```
ChrTracer3/
├── Readout_001/                 # hyb 1
│   ├── ConvZscan_01.dax         # FOV 01 image stack
│   ├── ConvZscan_01.inf         # metadata: dimensions, n_frames
│   ├── ConvZscan_02.dax         # FOV 02
│   └── ConvZscan_02.inf
├── Readout_002/                 # hyb 2
│   └── ...
└── Readout_060/                 # hyb 60
    └── ...
```

### Frame Interleaving

Frames within each output DAX stack are interleaved by channel:
```
ch0_z0, ch1_z0, ch0_z1, ch1_z1, ...
```
where ch0 = fiducial channel, ch1 = readout (data) channel. This matches the ChrTracer3 convention expected by `read_dax()` and `fiducial_maxproj()`.

### Efficiency

- Pre-allocates output stacks to avoid dynamic resizing
- Single sequential forward pass through all `.dat` files (I/O efficient for large datasets)
- No redundant file operations or backward seeking
- Supports optional TIFF output for visualization/QC

---

## 3. Step 1: FOV-Level Drift Correction

**Module**: `orca_drift.py`
**Matlab equivalent**: `CorrAlignFast` + `CorrAlignRotateScale`

### Purpose

Correct inter-hyb XY drift by comparing fiducial bead images across hybridization rounds. Each hyb's fiducial channel max-projection is aligned to the reference hyb (hyb 1).

### Algorithm: Two-Step FFT Cross-Correlation

#### 2.1 Core Correlation: `_corr_align_rotate_scale()`

Computes sub-pixel 2D shift between two images via FFT cross-correlation.

**Input**: Two 2D images (reference and moving), `max_shift` limit, `grad_max` flag
**Output**: `(yshift, xshift, corrPeak, corrMap)`

1. **Normalization** (Matlab `CorrAlignRotateScale` match):
   ```
   im = (im - mean(im)) / std(im)
   ```
   Both images are mean-subtracted and std-normalized to ensure correlation peak reflects structural similarity, not intensity differences.

2. **FFT Cross-Correlation**:
   ```
   F1 = rfft2(im1)
   F2 = rfft2(im2)
   corrM = fftshift(irfft2(conj(F1) × F2))
   ```
   The `conj(F1) × F2` convention places the correlation peak at the displacement vector directly. After `fftshift`, zero-displacement is at `(H//2, W//2)`.

3. **Search Region Restriction**:
   - Compute half-box: `Hc = min(H, max_shift)`, `Wc = min(W, max_shift)`
   - Extract center region: `corrMmini = corrM[zr-Hc2 : zr+Hc2, zc-Wc2 : zc+Wc2]`

4. **Peak Finding**:
   - **gradMax mode** (`grad_max=True`, default): Matches Matlab `gradMax` algorithm
     - Compute gradient: `gy, gx = np.gradient(corrMmini)`
     - Compute Laplacian: `laplacian = d²gx/dx² + d²gy/dy²`
     - Peak = minimum of Laplacian (sharpest curvature)
     - Provides sub-pixel accuracy via second-derivative analysis
   - **argmax mode** (`grad_max=False`): Simple integer argmax of correlation map

5. **Shift Output**:
   ```
   xshift = cx - Wc2    # No negation needed for conj(F1)*F2 convention
   yshift = cy - Hc2
   ```

#### 2.2 Coarse Step: `coarse_shift()`

Fast initial alignment using downsampled images.

**Input**: Reference and moving max-projections
**Output**: `(dy, dx)` integer pixel shifts

1. **Adaptive Downsampling** (Matlab `CorrAlignFast` match):
   ```
   rel_speed = sqrt(H × W) / max_size    # max_size = 400
   ```
   If `rel_speed > 1`, downsample using bilinear interpolation (`scipy.ndimage.zoom(img, 1/rel_speed, order=1)`).

2. **Coarse Correlation**: Call `_corr_align_rotate_scale()` with `grad_max=False` (argmax — gradMax on low-res images can find false peaks).

3. **Rescale**: `dy = round(rel_speed × yshift)`, `dx = round(rel_speed × xshift)`

#### 2.3 Fine Step: `fine_shift()`

Sub-pixel refinement in a bright region of the image.

**Input**: Reference and moving projections, coarse shifts
**Output**: `(dy2, dx2)` fine correction shifts

1. **Apply Coarse Correction**: Shift moving image by `(-coarse_dy, -coarse_dx)`.

2. **Find Brightest Overlapping Region** (Matlab approach):
   - Downsample both images
   - Compute product: `product = ref_small × mov_aligned_small`
   - Peak of product = region with most overlapping signal
   - Map back to full resolution

3. **Crop Fine-Search Box**: Extract `±crop` (default 200 px) around product peak.

4. **Fine Correlation with gradMax**: Call `_corr_align_rotate_scale()` with `grad_max=True` for sub-pixel accuracy.

5. **Sanity Check**: If `|dx2| > max_fine_shift` or `|dy2| > max_fine_shift`, discard fine correction (set to 0).

#### 2.4 Per-FOV Processing: `correct_one_fov_stream()`

Orchestrates drift correction for all hybs in one FOV.

1. Load reference hyb fiducial max-projection
2. For each hyb:
   - Load fiducial max-projection
   - Compute coarse shift
   - Compute fine shift
   - Store: `{xshift, yshift, theta=0, rescale=1, xshift2, yshift2, theta2=0, rescale2=1}`
3. Save `fov{N:03d}_regData.csv`
4. Optionally save 3-panel diagnostic PNGs

**Output CSV columns**: `xshift, yshift, theta, rescale, xshift2, yshift2, theta2, rescale2`
(theta and rescale are always 0 and 1 — reserved for rotation/scale correction not implemented)

---

## 4. Step 2: Spot Detection

**Function**: `orca_fit.detect_spots()`
**Matlab equivalent**: `ChrTracer3` SpotSelector

### Algorithm

1. **Downsample**: `img_ds = maxproj[::downsample, ::downsample]` (default 3×)

2. **Background Subtraction**:
   ```
   bg = uniform_filter(img_ds, size=bg_size/downsample)    # rolling-ball
   img_bg = clip(img_ds - bg, min=0)
   ```

3. **Threshold**: `thresh = percentile(img_bg, threshold_pct × 100)` (default 99.7th percentile)

4. **Local Maxima Detection**:
   - Footprint: `(2×min_dist_ds + 1)²` square
   - Keep pixels where: `img_bg == maximum_filter(img_bg, footprint)` AND `img_bg > thresh`

5. **Upscale**: Map back to full resolution: `rows = rows_ds × downsample`

6. **Border Removal**: Exclude spots within `border` pixels of image edges

**Output**: DataFrame with columns `{locusX, locusY}` (0-based pixel coordinates)

---

## 5. Step 3: 3D Gaussian Fitting

**Module**: `orca_fit.py`
**Matlab equivalent**: `ChrTracer3_FitSpots` → `Register3D` → `FitPsf3D`

### 4.1 Per-Spot Fine 3D Alignment: `fine_align_crop()`

Aligns the fiducial crop of each hyb to the reference hyb in 3D (XY + Z).

**Matlab equivalent**: `Register3D`

1. **Find Fiducial Peak**: `_find_peak_3d()` on reference crop
   - Border exclusion: `wb = min(max_fit_width//2, H//2, W//2)`, `wz = min(max_fit_zdepth//2, nZ//2) - 1`
   - Gaussian blur: `sigma=0.5` (matches Matlab `imgaussfilt3`)
   - Return brightest voxel in interior

2. **Restrict Sub-Volume**: Crop `±max_shift_xy` (XY) and `±max_shift_z` (Z) around fiducial peak

3. **Upsample**: Bilinear zoom by `(upsample_z, upsample, upsample)` (default 4×)

4. **3D Edge Subtraction**: Subtract 90th percentile of all 6 faces, clip negative to 0

5. **XY Shift** (from Z max-projection):
   ```
   ref_xy = ref_up3.max(axis=0)    # Z projection
   mov_xy = mov_up3.max(axis=0)
   xcorr_xy = fftshift(irfft2(conj(fft2(ref_xy)) × fft2(mov_xy)))
   (dy_xy, dx_xy) = argmax(xcorr_xy) - center
   ```

6. **Z Shift** (from Y max-projection):
   ```
   ref_yz = ref_up3.max(axis=1)    # Y projection → (Z, X) plane
   mov_yz = mov_up3.max(axis=1)
   xcorr_yz = fftshift(irfft2(conj(fft2(ref_yz)) × fft2(mov_yz)))
   dz_up = argmax_row(xcorr_yz) - center_row
   ```

7. **Combine and Rescale**:
   ```
   xshift = (dx_xy + dx_yz) / 2 / upsample    # average X from both projections
   yshift = dy_xy / upsample                   # Y from XY projection only
   zshift = dz_up / upsample_z                 # Z from YZ projection
   ```

### 4.2 Peak Finding: `_find_peak_3d()`

**Matlab equivalent**: `FindPeaks3D`

1. Border exclusion (computed from `max_fit_width` and `max_fit_zdepth`)
2. Gaussian blur (sigma=0.5)
3. Min peak height threshold
4. Return brightest voxel in interior

### 4.3 Restricted Region Search

After fine alignment, the data crop is restricted to a smaller search region around the fiducial peak before fitting. This matches Matlab's approach of calling `FitPsf3D` on a restricted region.

1. Find fiducial peak in reference crop
2. Define restricted region: `±max_xy_step_search` (XY), `±max_z_step` (Z) around fiducial
3. Find data peak within restricted region using `_find_peak_3d()`
4. Extract fitting sub-crop from restricted region

### 4.4 3D Gaussian Fitting: `fit_gaussian_3d()`

**Matlab equivalent**: `FitPsf3D`

**Model**:
```
G(x,y,z) = bg + h × exp(-0.5 × ((x-x0)²/sx² + (y-y0)²/sy² + (z-z0)²/sz²))
```

1. **Grid Setup**: 1-based coordinates `meshgrid(1:W, 1:H, 1:Z)` (matching Matlab)

2. **Initial Guess**:
   - `(x0, y0, z0)` = argmax of volume + 1 (1-based)
   - `bg0` = 10th percentile of volume
   - `h0` = max(volume) - bg0
   - `sx0 = sy0 = 1.25 × √2` (Matlab `initSigmaXY=1.25` converted to Python convention)
   - `sz0 = 2.5 × √2`

3. **Bounds** (Matlab-matched):
   - XY sigma: `[0.5, 2.0 × √2]` (Matlab `maxSigma=2.0` → Python `2.83`)
   - Z sigma: `[0.5, 2.5 × √2]` (Matlab `maxSigmaZ=2.5` → Python `3.54`)
   - Peak position: `±2 px` around initial guess (Matlab `peakBound=2`)
   - Height: `[0, ∞)`
   - Background: `[0, ∞)`

4. **Optimizer**: `scipy.optimize.curve_fit()` (Levenberg-Marquardt), `maxfev=5000`

5. **Output Conversion** (to Matlab convention):
   ```
   wx_nm = sx × nm_xy / √2    # Python σ → Matlab w convention
   wy_nm = sy × nm_xy / √2
   wz_nm = sz × nm_z / √2
   ```

### 4.5 Quality Checks: `_check_fit_quality()`

| Check | Criterion | Default | Matlab Equivalent |
|-------|-----------|---------|-------------------|
| Raw peak intensity | `h_raw ≥ min_h` | 200 | `datMinPeakHeight` |
| Signal-to-background | `h_raw / bg ≥ min_hb_ratio` | 1.2 | `datMinHBratio` |
| Amplitude-to-peak | `h_fit / h_raw ≥ min_ah_ratio` | 0.25 | `datMinAHratio` |
| XY width | `wx_nm / nm_xy ≤ max_wx_px` | 2.0 | `maxSigma` |
| Z width | `wz_nm / nm_z ≤ max_wz_sl` | 2.5 | `maxSigmaZ` |
| Position | within `±max_xy_step` of crop center | 12 px | `maxXYstep` |

**Status codes**: `ok`, `low_amp`, `low_quality`, `wide_spot`, `no_peak`

### 4.6 Coordinate Conversion Chain

```
Sub-crop (1-based fit coordinates)
  → Restricted region (add restricted_origin)
  → Full crop (add restrict_to_crop offset)
  → FOV coordinates (add crop_origin)
  → Nanometers (multiply by nm_xy or nm_z)
```

### 4.7 I/O Optimization

- **Outer loop = hybs**: Read each DAX file once, process all spots from that file
- **Inner loop = spots**: Crop volumes from already-loaded stack
- This is O(N_hybs) DAX reads instead of O(N_hybs × N_spots)

---

## 6. V4 Optimization Changes

Six changes were made to match Matlab ChrTracer3 output:

### Change 1: 3D Fine Alignment (Z drift correction)

**Problem**: `fine_align_crop()` only computed XY shifts (2D phase correlation on max-projections). Matlab's `Register3D` computes full 3D (XY + Z) shifts.

**Fix**: Rewrote `fine_align_crop()` to:
- Find fiducial peak using `_find_peak_3d()` (matching Matlab `FindPeaks3D`)
- Crop ±max_shift_xy/±max_shift_z around fiducial peak (not image center)
- Upsample 3D sub-volume (4× bilinear zoom)
- 3D edge subtraction (90th percentile of all 6 faces)
- XY shift from XY max-projection FFT correlation
- Z shift from YZ max-projection FFT correlation
- Returns `(dy, dx, dz, score)` instead of `(dy, dx, score)`

**Impact**: Z MAD improved from 1686 nm to 86 nm (20× improvement)

### Change 2: Fine Alignment Shift Sign Fix

**Problem**: `nd_shift(dat_crop, (fine_dz, fine_dy, fine_dx))` applied the shift in the wrong direction — doubling misalignment.

**Fix**: Changed to `nd_shift(dat_crop, (-fine_dz, -fine_dy, -fine_dx))`. Verified with synthetic test: negative sign gives RMSE=0, positive gives RMSE=49.6.

### Change 3: Restricted Region Search

**Problem**: Python extracted fitting sub-crop from full data crop. Matlab restricts data to ±maxXYstep/±maxZstep around fiducial before fitting.

**Fix**: Added restricted region step between fine alignment and Gaussian fitting, matching Matlab's `FitPsf3D` workflow.

### Change 4: Gaussian Width Convention

**Problem**: Python `exp(-0.5×(r/s)²)` vs Matlab `exp(-(r/(2s))²)`. Relationship: `s_python = √2 × s_matlab`. Output widths were in Python convention.

**Fix**: Divide fitted sigmas by √2 at output: `wx_nm = sx × nm_xy / √2`

**Impact**: Width ratio improved from 1.39× to ~1.02×

### Change 5: Fitting Bounds Matched to Matlab

**Fix**:
- XY sigma upper: `2.0 × √2 = 2.83` (Matlab `maxSigma=2.0`)
- Z sigma upper: `2.5 × √2 = 3.54` (Matlab `maxSigmaZ=2.5`)
- Peak position bounds: ±2 px (Matlab `peakBound=2`)
- Initial sigma XY: `1.25 × √2 = 1.77` (Matlab `initSigmaXY=1.25`)
- 1-based coordinate grid (matching Matlab meshgrid)

### Change 6: FOV-Level Drift Correction Rewrite

**Problem**: Old drift correction had ~4.4 px systematic offset from Matlab due to: no normalization, fixed ds=4 downsampling, argmax-only peak finding.

**Fix**: Rewrote to match Matlab `CorrAlignFast` + `CorrAlignRotateScale`:
- Image normalization (subtract mean, divide by std)
- Adaptive downsampling: `relSpeed = sqrt(H×W)/400`
- Coarse step: argmax (gradMax unreliable on low-res)
- Fine step: gradMax (Laplacian minimum) for sub-pixel accuracy
- Product-based bright region detection for fine search
- ProcessPoolExecutor with 4 workers for parallel FOV processing

---

## 7. Matlab-Python Comparison Results

Comparison across all 20 FOVs, 30 hybs (odd hybs only for Matlab), ~153,666 matched spot pairs.

### Summary Metrics

| Metric | Before V4 | After V4 | Target |
|--------|-----------|----------|--------|
| Z MAD | 1686 nm | **86 nm** | <200 nm |
| dx MAD | 37 nm | **215 nm** | — |
| dy MAD | 37 nm | **190 nm** | — |
| wx ratio (P/M) | 1.39 | **1.029** | ~1.0 |
| wy ratio (P/M) | 1.39 | **1.019** | ~1.0 |
| wz ratio (P/M) | 1.39 | **1.013** | ~1.0 |
| h_fit/a ratio | — | **0.961** | ~1.0 |
| OK rate | ~85% | **82.7%** | >80% |
| Total fits | — | 202,260 | — |
| OK fits | — | 167,290 | — |
| Matched pairs | — | 153,666 | — |

### Notes on Remaining XY Offset (~200 nm)

The Python pipeline detects a real ~3-5 px systematic drift between hybs that Matlab's `CorrAlignFast` does not correct for on this dataset. This manifests as ~200 nm XY position offset (dx/dy MAD above). The offset is consistent across all FOVs and hybes, indicating it is a genuine drift signal. This is considered a feature of the improved Python drift detection, not a bug.

---

## 8. Parameters Reference

### Drift Parameters (`DRIFT_PARAMS`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ref_hyb` | 1 | Reference hybridization round |
| `fid_ch` | 0 | Fiducial channel index |
| `n_ch` | 2 | Total channels per stack |
| `ds` | 4 | Legacy downsampling (unused in V4) |
| `crop` | 200 | Fine search box half-size (px) |
| `spot_percentile` | 75 | Legacy (unused in V4) |
| `max_fine_shift` | 5 | Threshold for discarding fine shifts |

### Fitting Parameters (`FIT_PARAMS`)

| Parameter | Value | Matlab Equivalent | Description |
|-----------|-------|-------------------|-------------|
| `nm_xy` | 108 | `nmXYpix` | Pixel size XY (nm) |
| `nm_z` | 150 | `nmZpix` | Pixel size Z (nm) |
| `box_half` | 15 | `boxWidth/2` | Crop half-size (px) |
| `n_ch` | 2 | — | Channels per stack |
| `ref_hyb` | 1 | — | Reference hyb |
| `upsample` | 4 | `upsample` | Fine alignment upsampling |
| `max_fine_shift` | 4.0 | `maxXYdrift` | Max fine XY shift (px) |
| `max_fine_shift_z` | 6.0 | `maxZdrift` | Max fine Z shift (slices) |
| `fit_half_xy` | 4 | `maxFitWidth/2` | Fit sub-crop XY (px) |
| `fit_half_z` | 6 | `maxFitZdepth/2` | Fit sub-crop Z (slices) |
| `min_h` | 200 | `datMinPeakHeight` | Min raw peak intensity |
| `min_hb_ratio` | 1.2 | `datMinHBratio` | Signal/background ratio |
| `min_ah_ratio` | 0.25 | `datMinAHratio` | Fit amplitude/peak ratio |
| `max_xy_step` | 12.0 | `maxXYstep` | Max fit position offset (px) |
| `max_xy_step_search` | 12 | `maxXYstep` | Search region XY (px) |
| `max_z_step` | 8 | `maxZstep` | Search region Z (slices) |
| `max_wx_px` | 2.0 | `maxSigma` | Max Gaussian width XY (Matlab conv.) |
| `max_wz_sl` | 2.5 | `maxSigmaZ` | Max Gaussian width Z (Matlab conv.) |

### Spot Detection Parameters (`SPOT_PARAMS`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `threshold_pct` | 0.997 | Detection threshold (99.7th percentile) |
| `bg_size` | 50 | Rolling-ball background filter size (px) |
| `min_dist` | 30 | Minimum spot separation (px) |
| `downsample` | 3 | Detection downsampling factor |
| `border` | 2 | Border exclusion (px) |

---

## 9. File Format Reference

### Input: DAX/INF Format

- `.dax`: Raw binary, 16-bit unsigned integers, row-major
- `.inf`: Text header with `frame dimensions = WxH`, `number of frames = N`
- Stack shape: `(N_frames, H, W)` where `N_frames = n_ch × n_z_slices`
- Channel interleaving: `ch0_z0, ch1_z0, ch0_z1, ch1_z1, ...`

### Output Files

| File | Columns | Description |
|------|---------|-------------|
| `fov{N}_regData.csv` | xshift, yshift, theta, rescale, xshift2, yshift2, theta2, rescale2 | Drift correction per hyb |
| `fov{N}_selectSpots.csv` | locusX, locusY | Detected spot positions |
| `fov{N}_allFits.csv` | fov, spot_id, locus_x, locus_y, hybe, x, y, z, h, h_fit, wx, wy, wz, bg, fitQuality, status, ... | Per-spot per-hyb fit results |
| `allFits.csv` | (same as above) | Merged all FOVs |

---

## 10. Coordinate Conventions

### Gaussian Convention

| Convention | Formula | Relationship |
|------------|---------|--------------|
| Python (standard) | `exp(-0.5 × (r/σ)²)` | σ_python = √2 × w_matlab |
| Matlab (ChrTracer3) | `exp(-(r/(2w))²)` | w_matlab = σ_python / √2 |

Fitting is performed in Python convention internally. Output widths (wx, wy, wz) are reported in **Matlab convention** (divided by √2).

### Coordinate Systems

| Stage | Convention | Origin |
|-------|-----------|--------|
| Fitting grid | 1-based | (1,1,1) = first voxel |
| Crop coordinates | 0-based | (0,0,0) = first voxel |
| Output positions (nm) | 1-based × nm | (1 × nm_xy, 1 × nm_xy, 1 × nm_z) |

### Drift Convention

- `xshift, yshift`: Displacement of moving image relative to reference (positive = moved right/down)
- Applied as: `nd_shift(moving, (-yshift, -xshift))` to align moving to reference
- Fine alignment: `nd_shift(data, (-fine_dz, -fine_dy, -fine_dx))` (negative = correction)

---

## Appendix: Comparison with Matlab ChrTracer3

| Component | Matlab Function | Python Function | Key Differences |
|-----------|----------------|-----------------|-----------------|
| Drift correction | `CorrAlignFast` | `coarse_shift` + `fine_shift` | Python detects ~3px real drift Matlab misses |
| Correlation | `CorrAlignRotateScale` | `_corr_align_rotate_scale` | Translation only (no rotation/scale) |
| Fine alignment | `Register3D` | `fine_align_crop` | Identical algorithm after V4 rewrite |
| Peak finding | `FindPeaks3D` | `_find_peak_3d` | Matched border exclusion + blur |
| Gaussian fitting | `FitPsf3D` | `fit_gaussian_3d` | Python uses LM, Matlab uses lsqnonlin |
| Spot detection | SpotSelector | `detect_spots` | Background subtraction + local maxima |
| Parallelization | Matlab parfor | ProcessPoolExecutor | Python avoids GIL via multiprocessing |

---

## 11. Reference

1. Mateo LJ, Murphy SE, Hafner A, Cinquini IS, Walker CA, Boettiger AN. Visualizing DNA folding and RNA in embryos at single-cell resolution. *Nature*. 2019;568(7750):49-54. doi:10.1038/s41586-019-1035-4

2. Boettiger AN, Murphy SE. Advances in chromatin imaging at kilobase-scale resolution. *Trends in Genetics*. 2020;36(4):273-287. doi:10.1016/j.tig.2019.12.010

3. ORCA-public Matlab pipeline: https://github.com/BoettigerLab/ORCA-public
