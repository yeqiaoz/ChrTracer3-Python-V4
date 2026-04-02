# ChrTracer3 Python V3 — 3D Gaussian Fitting Implementation Plan

**Created:** 2026-03-26
**Author:** Yeqiao Zhou, Faryabi Lab
**Base:** ChrTracer3_py_app_V2 (2D max-projection + 1D Z-profile fitting)
**Goal:** Replace V2's 2D+1D fitting with full 3D Gaussian fitting to match Matlab FitPsf3D behaviour.

---

## Background: Why 3D Fitting?

V2 fits a 2D Gaussian on the XY max-projection of the full 31×31 crop, then a separate 1D Gaussian on the Z profile at the fitted XY centre.
This approach causes:

1. **wx ~1.73× (≈√3) broader than Matlab** — two compounding factors:
   - Factor 1 (×√2): Python uses standard `exp(-½(r/σ)²)` vs Matlab's `exp(-(r/(2σ))²)` convention.
   - Factor 2 (×1.22 empirical): 2D fitting on the 31×31 max-projection finds a broader, shallower minimum compared to 3D fitting on a small 9×9 crop.
2. **wz ~√2 broader than Matlab** — Factor 1 alone applies (1D Z-fit uses same convention).
3. **fitQuality not comparable to Matlab** — V2 uses `mean(2D residuals²)/h²`; Matlab uses `sum(3D residuals²)/a²`.
4. **h_fit agrees to 5–15%** — the 2D max-projection picks the best z-slice, inflating the amplitude slightly.

V3 will fit a full 3D Gaussian on a small 3D sub-crop (matching Matlab's 9×9×13 bounding box), eliminating Factor 2 and making fitQuality directly comparable to Matlab.

---

## Matlab Reference: FitPsf3D.m

**File:** `~/Desktop/Aries/software/ORCA-public/matlab-functions/ChrTracingLib/FitPsf3D.m`

Key parameters:
| Matlab variable | Value | Meaning |
|---|---|---|
| `bw` (`maxFitWidth`) | 4 px | XY half-width of fitting sub-crop |
| `bz` (`maxFitZdepth`) | 6 slices | Z half-depth of fitting sub-crop |
| `maxSigma` | 2 (Matlab convention) | Optimizer upper bound on XY sigma |

Matlab Gaussian model (line 199):
```matlab
exp(-((Y-p(1))/(2*p(2))).^2 - ((X-p(3))/(2*p(2))).^2 - ((Z-p(5))/(2*p(4))).^2)
```
This is `exp(-(r/(2σ))²)`, so for the same PSF:
`σ_python_standard = √2 × σ_matlab`

Matlab fitQuality (line 236):
```matlab
resRatio(i) = resnorm / a(i)^2;
```
where `resnorm = sum((fit - data).^2)` over the 9×9×13 sub-crop.

---

## V3 Fitting Strategy

### Replace `fit_gaussian_2d` + `fit_gaussian_1d` with `fit_gaussian_3d`

**Sub-crop extraction** (matches Matlab):
```
bw = 4 (fit_half_xy)
bz = 6 (fit_half_z)
→ 9×9×13 voxel sub-crop centred on the max-intensity voxel
```

**Python Gaussian model** (standard convention):
```python
G = bg + h * exp(-½ * ((dx/sx)² + (dy/sy)² + (dz/sz)²))
```

**Sigma convention mapping** (for quality filters and display):
```
wx_matlab = σ_python / √2
→ Matlab maxSigma=2 ↔ Python σ_upper = 2√2 ≈ 2.83 px
```

**fitQuality** (directly matches Matlab):
```python
fit_quality = resnorm / h_fit**2   # resnorm = sum(residuals²) over sub-crop
```

### Updated quality filter defaults

| Parameter | V2 (2D) | V3 (3D) | Rationale |
|---|---|---|---|
| `max_wx_px` | 4.0 | **2.83** | Matlab `maxSigma=2` × √2; or keep 4 as a loose upper bound |
| `max_wz_sl` | 12.0 | **8.5** | Matlab `maxSigmaZ=6` × √2 ≈ 8.5 |
| `max_fitQuality` | (calibrated) | **calibrated** | Now comparable to Matlab (default `inf` in Matlab) |

### Updated ORCA_app.py defaults

```python
"fit_all_maxwx":    3,    # σ in px (Python standard), ≈ Matlab maxSigma=2 × √2
"fit_all_maxwz":    9,    # σ in slices (Python standard)
```

---

## Implementation Plan

### 1. `orca_fit.py` changes

#### 1a. Add `_gauss3d_flat` (private helper)
```python
def _gauss3d_flat(xyz, x0, y0, sx, sy, z0, sz, h, bg):
    """Standard 3D Gaussian: G = bg + h·exp(-½·((dx/sx)²+(dy/sy)²+(dz/sz)²))."""
    x, y, z = xyz
    return (bg + h * np.exp(-0.5 * (
        ((x - x0) / sx) ** 2 +
        ((y - y0) / sy) ** 2 +
        ((z - z0) / sz) ** 2
    ))).ravel()
```

#### 1b. Add `fit_gaussian_3d`
```python
def fit_gaussian_3d(vol):
    """Fit 3D Gaussian to volume (nZ, H, W). Returns (x0, y0, sx, sy, z0, sz, h, bg, rmse, resnorm).

    Uses standard convention: G = bg + h·exp(-½·((dx/sx)²+(dy/sy)²+(dz/sz)²))
    → sx_python = √2 · wx_matlab for same PSF.
    resnorm = sum((vol - fit)²) — directly comparable to Matlab lsqnonlin resnorm.
    """
    nZ, H, W = vol.shape
    zg, yg, xg = np.mgrid[0:nZ, 0:H, 0:W]
    data = vol.ravel().astype(float)

    bg0 = float(np.percentile(vol, 10))
    h0  = float(vol.max() - bg0)
    pk  = np.unravel_index(np.argmax(vol), vol.shape)
    z0i, y0i, x0i = float(pk[0]), float(pk[1]), float(pk[2])

    try:
        popt, _ = curve_fit(
            _gauss3d_flat,
            (xg.ravel(), yg.ravel(), zg.ravel()),
            data,
            p0=[x0i, y0i, 1.5, 1.5, z0i, 2.5, h0, bg0],
            bounds=(
                [0,  0,  0.3, 0.3, 0,   0.3, 0,      -np.inf],
                [W,  H,  4.0, 4.0, nZ,  6.0, np.inf,  np.inf],
            ),
            maxfev=3000,
        )
        fit       = _gauss3d_flat((xg.ravel(), yg.ravel(), zg.ravel()), *popt).reshape(nZ, H, W)
        residuals = vol.astype(float) - fit
        resnorm   = float(np.sum(residuals ** 2))
        rmse      = float(np.sqrt(np.mean(residuals ** 2)))
        return (*popt[:8], rmse, resnorm)
    except Exception:
        resnorm = float(np.sum((vol.astype(float) - bg0) ** 2))
        return x0i, y0i, 1.5, 1.5, z0i, 2.5, h0, bg0, np.nan, resnorm
```

**Optimizer bounds explanation:**
- XY sigma upper bound: 4.0 px (standard Python σ). Matlab maxSigma=2 in Matlab convention → 2√2 ≈ 2.83 px. Set to 4.0 px for a small margin.
- Z sigma upper bound: 6.0 slices (Python standard σ). Adjust after validation.

#### 1c. Replace fitting block in `fit_one_hyb` and `_fit_from_crops`

Replace 2D+1D block with:
```python
# Extract small 3D sub-crop centred on max-intensity voxel (matches Matlab bw=4, bz=6)
nZ, H_cr, W_cr = dat_crop.shape
dat_maxproj = dat_crop.max(axis=0)   # kept for h_raw and display
pk2d = np.unravel_index(np.argmax(dat_maxproj), dat_maxproj.shape)
peak_y, peak_x = int(pk2d[0]), int(pk2d[1])
peak_z = int(np.argmax(dat_crop[:, peak_y, peak_x]))

fhxy = int(params.get("fit_half_xy", 4))
fhz  = int(params.get("fit_half_z",  6))
xs = max(0, peak_x - fhxy);  xe = min(W_cr, peak_x + fhxy + 1)
ys = max(0, peak_y - fhxy);  ye = min(H_cr, peak_y + fhxy + 1)
zs = max(0, peak_z - fhz);   ze = min(nZ,   peak_z + fhz  + 1)
fit_sub = dat_crop[zs:ze, ys:ye, xs:xe]

# 3D Gaussian fit — standard convention: G = bg + h·exp(-½·((dx/sx)²+(dy/sy)²+(dz/sz)²))
# sx_python = √2 · wx_matlab  (Matlab uses exp(-((dx/(2σ))²+...)))
x0s, y0s, sx, sy, z0s, sz, h, bg, rmse, resnorm = fit_gaussian_3d(fit_sub)

# Convert sub-crop coords → full-crop coords
x0 = x0s + xs
y0 = y0s + ys
z0 = z0s + zs

h_raw = float(dat_maxproj.max())
# fitQuality = sum(residuals²) / h_fit² — matches Matlab FitPsf3D resRatio formula
# (Matlab: resnorm/a², same 3D crop, same sum convention → directly comparable)
fit_quality = float(resnorm / (h ** 2 + 1e-10))
wx_nm, wy_nm, wz_nm = sx * nm_xy, sy * nm_xy, sz * nm_z
fit_status = _check_fit_quality(h_raw, h, bg, x0, y0,
                                wx_nm, wy_nm, wz_nm, nm_xy, nm_z, params)
```

#### 1d. Update `_check_fit_quality` defaults
```python
max_wx_px = params.get("max_wx_px", 3.0)   # was 4.0 in V2 (now √2 smaller crop)
max_wz_sl = params.get("max_wz_sl", 9.0)   # was 12.0 in V2
```

#### 1e. Update module docstring
```
Fit 3D Gaussian to a small sub-crop (bw=4, bz=6, matching Matlab FitPsf3D)
```

### 2. `ORCA_app.py` changes

```python
_ALL_DEFAULTS = {
    ...
    "fit_all_maxwx":    3,    # Matlab maxSigma=2 × √2 ≈ 2.83; round up to 3
    "fit_all_maxwz":    9,    # Matlab maxSigmaZ × √2 (validate empirically)
    ...
}
```

Update Quality Filters caption and help texts to explain:
- 3D sub-crop fitting
- σ_python = √2 × σ_matlab convention
- fitQuality is now directly comparable to Matlab (sum/h², not mean)

---

## Validation Plan

After implementing V3:

1. Re-run `analysis_260322_py`-equivalent on the 240402 dataset (or a subset, e.g. FOV 1–3).
2. Run comparison script (extend `compare_fits3.py` to add V3 results).
3. Check:
   - `wx` ratio: target 1.0–1.41 (removing the 1.22× broadening; √2 convention difference remains)
   - `wz` ratio: target ~√2 (convention only; no crop-size broadening expected)
   - `fitQuality` ratio: target ~1 (same formula, same crop size)
   - `h_fit` / Matlab `a`: target ~0.94 (similar to V2; max-proj still used for h_raw)

---

## Open Questions

1. **h_raw definition**: V3 will still use `dat_maxproj.max()` for h_raw (peak of max-projection). Matlab uses a per-slice raw peak. This difference is expected and acceptable — both serve as detection thresholds.
2. **Speed**: 3D fitting is ~3–5× slower per spot than 2D+1D. Profiling needed; may require tighter sub-crop or Numba acceleration.
3. **Border sub-crops**: If the spot is near the crop edge, the 9×9 sub-crop may be smaller. The `fit_gaussian_3d` function should handle this gracefully via the `bounds` relative to the actual sub-crop shape.
4. **fitQuality threshold**: Matlab default is `inf` (no filtering). V3 will set a finite default, calibrated from the 240402 dataset comparison.
