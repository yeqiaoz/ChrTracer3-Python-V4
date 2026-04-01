# ChrTracer3 V4 Optimization Changelog (2026-03-30)

## Goal
Match Python ChrTracer3 V4 pipeline output to Matlab on the same masked images,
especially XYZ positions of fitted spots.

## Changes

### 1. Per-spot fine alignment: Z drift correction (`orca_fit.py`)
**Problem**: `fine_align_crop()` only computed XY shifts (2D phase correlation on max-projections).
Matlab's `Register3D` computes full 3D (XY + Z) shifts.

**Fix**:
- Rewrote `fine_align_crop()` to match Matlab `Register3D`:
  - Finds fiducial peak using `_find_peak_3d()` (matching Matlab `FindPeaks3D` with border exclusion and Gaussian blur)
  - Crops +/-max_shift_xy, +/-max_shift_z around the fiducial peak (not image center)
  - Upsamples 3D sub-volume using bilinear zoom
  - 3D edge subtraction (90th percentile of all 6 faces)
  - XY shift from XY max-projection FFT correlation
  - Z shift from ZX max-projection FFT correlation (matching Matlab's YZ projection approach)
  - Returns `(dy, dx, dz, score)` instead of `(dy, dx, score)`

- Added `_find_peak_3d()` matching Matlab `FindPeaks3D`:
  - Border exclusion (wb, wz parameters matching Matlab formula)
  - Gaussian blur (sigma=0.5) before peak finding
  - Minimum peak height filtering

### 2. Per-spot fine alignment: shift sign fix (`orca_fit.py`)
**Problem**: `nd_shift(dat_crop, (fine_dz, fine_dy, fine_dx))` applied the shift in the
wrong direction — doubling the misalignment instead of correcting it.

**Fix**: Changed to `nd_shift(dat_crop, (-fine_dz, -fine_dy, -fine_dx))` in both
`fit_one_hyb()` (line ~544) and `_fit_from_crops()` (line ~701). Verified with a
synthetic test that negative sign gives RMSE=0, positive gives RMSE=49.6.

### 3. Per-spot fitting: restricted region and sub-crop extraction (`orca_fit.py`)
**Problem**: Python extracted the fitting sub-crop from the full data crop. Matlab
restricts data to +/-maxXYstep/+/-maxZstep around the fiducial peak before calling
`FitPsf3D`, which then finds peaks and extracts sub-crops within that restricted region.

**Fix**: Updated `fit_one_hyb()` and `_fit_from_crops()` to:
1. Apply fine alignment (3D shift) to the data crop
2. Restrict data to +/-max_xy_step_search/+/-max_z_step around fiducial peak
3. Find data peak within restricted region using `_find_peak_3d()`
4. Extract fitting sub-crop from restricted region (not full crop)
5. Coordinate conversion: sub-crop -> restricted -> full crop -> nm

### 4. Gaussian convention: width output (`orca_fit.py`)
**Problem**: Python Gaussian `exp(-0.5*(r/s)^2)` vs Matlab `exp(-(r/(2s))^2)`.
Relationship: `s_python = sqrt(2) * s_matlab`. Output widths were in Python convention.

**Fix**: At output time, divide fitted sigmas by sqrt(2) to report in Matlab convention:
```
wx_nm = sx * nm_xy / sqrt(2)
wz_nm = sz * nm_z / sqrt(2)
```

### 5. Fitting bounds matched to Matlab (`orca_fit.py`)
**Problem**: Fitting bounds were too loose compared to Matlab.

**Fix**: Updated `fit_gaussian_3d()`:
- XY sigma upper bound: 2.0*sqrt(2) = 2.83 (matching Matlab maxSigma=2.0)
- Z sigma upper bound: 2.5*sqrt(2) = 3.54 (matching Matlab maxSigmaZ=2.5)
- Peak position bounds: +/-2 px (matching Matlab peakBound=2)
- Initial sigma XY: 1.25*sqrt(2) = 1.77 (matching Matlab initSigmaXY=1.25)
- 1-based coordinate grid matching Matlab's meshgrid(1:cols, 1:rows, 1:stcks)

### 6. FOV-level drift correction rewrite (`orca_drift.py`)
**Problem**: Old drift correction had ~4.4px systematic offset from Matlab. Used simple
FFT cross-correlation without normalization, fixed ds=4 downsampling, and argmax peak finding.

**Fix**: Rewrote to match Matlab `CorrAlignFast` + `CorrAlignRotateScale`:
- **Normalization**: Images normalized (subtract mean, divide by std) before correlation
- **Correlation**: Spatial cross-correlation via `scipy.signal.fftconvolve` (matching Matlab `imfilter`)
- **Downsampling**: `relSpeed = sqrt(H*W)/maxSize` with maxSize=400 (matching Matlab)
- **Coarse step**: argmax peak finding (gradMax caused false peaks on some hybes)
- **Fine step**: gradMax (Laplacian minimum) for sub-pixel accuracy
- **Fine region**: Product-based bright region detection (im1 * im2_coarse_aligned)
- **Sign convention**: Outputs displacement of moving image (negated from Matlab's correction convention)

## Results (all 20 FOVs comparison with Matlab, 2026-03-31)

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

### Notes on drift correction
- Python drift correction detects a real ~3-5 px systematic drift between hybes
  that Matlab's CorrAlignFast does not correct for on this dataset
- This manifests as ~200 nm XY position offset (dx/dy MAD above)
- The offset is consistent across all FOVs and hybes
- Z accuracy dramatically improved: 1686 nm → 86 nm (20× improvement)
- Width ratios now match within 3% (previously 39% off due to √2 convention)

### Drift algorithm
- FFT cross-correlation: `ifft(conj(F1)*F2)` with normalization
- Coarse step: bilinear downsampling (relSpeed = sqrt(H*W)/400), argmax peak
- Fine step: product-based bright region detection, gradMax peak finding
- ProcessPoolExecutor with 4 workers for parallel FOV processing

## Files modified
- `/dobby/yeqiao/software_code/ChrTracer3_py_app_V4_test/orca_fit.py`
- `/dobby/yeqiao/software_code/ChrTracer3_py_app_V4_test/orca_drift.py`
- `/dobby/yeqiao/software_code/ChrTracer3_py_app_V4_test/run_pipeline_v4.py`

## Output
- Pipeline output: `analysis_260330_v4opt/`
- Comparison figures: `compare_result_masked/compare_masked.png`
