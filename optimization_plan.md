# Plan: Match Python ChrTracer3 V4 output to Matlab

## Context

Comparison of Matlab (`analysis_240410_masked`) vs Python V4 (`analysis_260328_v4test`) on the same masked images shows:
- **Z positions**: 1686 nm MAD — unacceptable
- **XY drift**: systematic ~9 px offset between drift corrections
- **XY positions**: 37 nm MAD — decent but improvable
- **Width ratios**: ~1.39x (expected sqrt(2) from convention difference)
- **OK rate**: Python 85% vs Matlab 94% (odd hybes)

Root causes identified from reading the source code of both pipelines.

## Critical files

- **Python pipeline**: `/dobby/yeqiao/software_code/ChrTracer3_py_app_V4_test/orca_drift.py`, `orca_fit.py`, `run_pipeline_v4.py`
- **Matlab pipeline**: `/dobby/yeqiao/software_code/ORCA-public/matlab-functions/ChrTracingLib/ChrTracer3_FitSpots.m`, `Register3D.m`, `FitPsf3D.m`, `CorrAlignFast.m`

## Changes (ordered by impact)

### 1. Add Z drift correction to per-spot fine alignment (fixes Z position — biggest issue)

**Problem**: Python's `fine_align_crop()` in `orca_fit.py` only computes XY shifts via 2D phase correlation on max-projections. Matlab's `Register3D` computes XYZ shifts by:
1. Crop 3D fiducial region around fiducial peak (+/-4 XY, +/-6 Z, upsampled 8x)
2. Compute XY shift from XY max-projection correlation
3. Compute Z shift from YZ max-projection correlation (the `yshift` of the YZ CorrAlignFast)
4. Apply all three shifts to the data crop before fitting

**Fix**: Extend `fine_align_crop()` to also compute Z shift from XZ or YZ projection correlation. Then apply the Z shift to each XY slice of the data crop before fitting.

Specifically in `orca_fit.py`:
- `fine_align_crop()` -> return `(dy, dx, dz, score)` instead of `(dy, dx, score)`
- Inside `_fit_from_crops()` and `fit_one_hyb()`: apply Z shift to data crop using `scipy.ndimage.shift` along axis 0

### 2. Match FOV-level drift correction algorithm (fixes ~9px XY offset)

**Problem**: Python uses simple array slicing `[::ds, ::ds]` for downsampling and bare `argmax` on the correlation peak. Matlab uses `imwarp` (bilinear affine) for downsampling and `gradMax` (gradient-based peak finding) for sub-pixel accuracy in the coarse step.

The fine step also differs: Python picks a fiducial bead at the 75th percentile brightness and correlates locally. Matlab uses the product `im1.*im2_coarse_aligned` to find the region with the most overlapping signal, then correlates that box.

**Fix in `orca_drift.py`**:
- `coarse_shift()`: Replace `ref_ds = ref_proj[::ds, ::ds]` with proper bilinear downsampling using `scipy.ndimage.zoom(ref_proj, 1/ds, order=1)`. Use gradient-based peak refinement on the correlation map (parabolic interpolation around the integer peak).
- `fine_shift()`: Instead of `_pick_ref_peak` (75th percentile bead), use the Matlab approach: multiply the coarse-aligned moving image with reference, find the brightest region in the product, and crop around that.

### 3. Match Matlab Gaussian convention in output (fixes sqrt(2) width ratio)

**Problem**: Python Gaussian: `exp(-0.5*(r/s)^2)`, Matlab: `exp(-(r/(2s))^2)`. Relationship: `s_python = sqrt(2) * s_matlab`.

**Fix in `orca_fit.py`**: At output time (where wx_nm, wy_nm, wz_nm are computed), divide by sqrt(2) to report in Matlab convention:
```python
wx_nm = sx * nm_xy / np.sqrt(2)
wy_nm = sy * nm_xy / np.sqrt(2)
wz_nm = sz * nm_z / np.sqrt(2)
```

Also update `_check_fit_quality` bounds accordingly (divide thresholds by sqrt(2) or adjust the comparison to use the pre-conversion values).

### 4. Match fitting bounds and sub-crop sizes

**Problem**: Minor differences in fitting parameters:
| Parameter | Matlab | Python |
|-----------|--------|--------|
| maxSigma (XY) | 2.0 | 4.0 (via bounds) |
| maxSigmaZ | 2.5 | 6.0 (via bounds) |
| peakBound | +/-2 px | no constraint (full sub-crop) |
| Z sub-crop half | 6 (datMaxFitZdepth=12/2) | 6 (already matches) |
| Meshgrid start | 1-based | 0-based |
| Initial sigma XY | 1.25 | 1.5 |
| Initial sigma Z | 2.5 | 2.5 |
| Optimizer | lsqnonlin | curve_fit (LM) |

**Fix in `orca_fit.py`**:
- `fit_gaussian_3d()` bounds: change XY sigma upper from 4.0 to 2.83 (= 2.0 * sqrt(2), converting Matlab bound to Python convention). Z sigma upper from 6.0 to 3.54 (= 2.5 * sqrt(2)).
- Add peak position bounds: fit center must stay within +/-2 px (peakBound) of the initial guess, matching Matlab's lb/ub on mu_x, mu_y, mu_z.
- Initial sigma XY: 1.25 * sqrt(2) ~ 1.77 (Matlab convention -> Python convention)
- Use 1-based coordinate grid inside `fit_gaussian_3d` to match Matlab's meshgrid, or account for the 1-unit offset in coordinate conversion.

### 5. Match QC filtering to increase OK rate

**Problem**: Python rejects more fits than Matlab. Key differences:
- Matlab filters by confidence interval width (`maxUncert=2px`), Python filters by `max_xy_step=12px` (position offset from crop center). These are not equivalent.
- Matlab uses `maxSigma=2` as a hard fitting bound (optimizer cannot exceed it), while Python has `max_wx_px=3.0` as a post-fit QC check with a looser fitting bound of 4.0. Spots hitting the Matlab bound at exactly 2.0 would pass QC, but in Python they'd fit to a larger value (up to 4.0) and fail the QC check.
- Python's `min_h=200` may be too restrictive for data channel (Matlab `datMinPeakHeight=500` is higher, but Matlab also doesn't flag low_amp separately -- it just excludes those peaks in FindPeaks3D before fitting).

**Fix**: Once fitting bounds are tightened (step 4), many fits that currently fail `wide_spot` will succeed. The `no_peak` rate should also decrease once drift correction is improved. Adjust `min_h` to match Matlab's `datMinPeakHeight=500` (currently 200) -- this is less restrictive because Matlab removes sub-500 peaks before fitting rather than flagging them after.

## Verification

After each change:
1. Re-run the V4 pipeline on the same masked images
2. Re-run `compare_masked.py` to measure improvement
3. Target metrics:
   - Z position MAD: < 200 nm (from 1686 nm)
   - XY position MAD: < 20 nm (from 37 nm)
   - Drift diff RMS: < 2 px (from ~9 px)
   - Width ratio: ~1.0 (from ~1.4)
   - OK rate (odd hybes): > 90% (from 85%)
