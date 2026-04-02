# ChrTracer3 V4_test — Continuation Log (2026-03-28)

## Current status (updated 2026-03-28 ~20:20)

**Pipeline is RUNNING** as PID 159633 (restarted ~19:30, previous PID 1533192 died after FOV 8).

### Step 1: Drift correction — IN PROGRESS
- FOVs 1-8: DONE (regData.csv exist)
- FOVs 9-12: IN PROGRESS (~48 min in, should finish ~20 min)
- FOVs 13-16: queued (batch 2)
- FOVs 17-20: queued (batch 3)
- **Estimated drift completion:** ~22:50 (~2.5 hours remaining)

### Step 2: Spot detection — NOT STARTED
- Will run automatically after drift completes

### Step 3: 3D Gaussian fitting — NOT STARTED
- Will run automatically after spot detection

### Step 2: Spot detection — NOT STARTED
- Will run after drift completes
- Sequential (1 FOV at a time), should be fast

### Step 3: 3D Gaussian fitting — NOT STARTED
- Will run after spot detection
- 8 workers for fitting

### Log file
```
/dobby/yeqiao/analysis/ORCA_ChrTracer3_processing_optimization/260318_Convert_ChrTracer3_toPython/v4test_pipeline.log
```

---

## How to check / resume

### Check if pipeline is still running
```bash
ps -p 1533192 -o pid,etime,%cpu --no-headers 2>/dev/null || echo "Process dead"
```

### Check progress
```bash
# regData CSVs completed
ls /dobby/yeqiao/analysis/ORCA_ChrTracer3_processing_optimization/240402_Granta519cl97_6hWO_MYC5p_30minHyb_30step_analysis/masked_images/ChrTracer3/analysis_260328_v4test/fov*_regData.csv 2>/dev/null

# selectSpots CSVs
ls .../analysis_260328_v4test/fov*_selectSpots.csv 2>/dev/null

# allFits CSVs
ls .../analysis_260328_v4test/fov*_allFits.csv 2>/dev/null

# Pipeline log
tail -30 /dobby/yeqiao/analysis/ORCA_ChrTracer3_processing_optimization/260318_Convert_ChrTracer3_toPython/v4test_pipeline.log
```

### If pipeline died — resume (auto-skips completed FOVs)
```bash
cd /dobby/yeqiao/software_code/ChrTracer3_py_app_V4_test
nohup python3 -u run_pipeline_v4.py 2>&1 | tee -a /dobby/yeqiao/analysis/ORCA_ChrTracer3_processing_optimization/260318_Convert_ChrTracer3_toPython/v4test_pipeline.log &
```
The pipeline checks for existing `regData.csv` / `selectSpots.csv` / `allFits.csv` and skips completed FOVs.

### Run steps individually
```bash
cd /dobby/yeqiao/software_code/ChrTracer3_py_app_V4_test

# Step 1 only
python3 -c "from run_pipeline_v4 import step1_drift; step1_drift()"

# Step 2 only
python3 -c "from run_pipeline_v4 import step2_detect_spots; step2_detect_spots()"

# Step 3 only
python3 -c "from run_pipeline_v4 import step3_fit; step3_fit()"
```

---

## What was done (history)

### 1. Created ChrTracer3_py_app_V4_test
- Copied from V3: `/dobby/yeqiao/software_code/ChrTracer3_py_app_V4_test/`
- Core modules unchanged: `orca_drift.py`, `orca_fit.py`, `ORCA_app.py`
- Added: `run_pipeline_v4.py` — batch pipeline script

### 2. Dataset: Matlab-generated DAX files
- **Location:** `/dobby/yeqiao/analysis/ORCA_ChrTracer3_processing_optimization/240402_Granta519cl97_6hWO_MYC5p_30minHyb_30step_analysis/masked_images/ChrTracer3/`
- **DAX files:** `Readout_001/` through `Readout_060/`, each with `ConvZscan_01.dax` through `ConvZscan_20.dax`
- **Image size:** 1866 x 1843 pixels, 108 frames (54 Z-slices x 2 channels)
- **Physical params:** nm_xy=108, nm_z=150, n_ch=2, fid_ch=0
- **FOVs:** 20, **Readouts:** 60

### 3. Matlab reference results exist
- **Location:** `.../ChrTracer3/analysis_240410_masked/`
- Files: `fov001-fov020_regData.csv`, `fov001-fov020_AllFits.csv`, `fov001-fov020_selectSpots.csv`
- Also: `240410_Granta519cl97_6hWO_allFits.csv` (merged), CorrAlign PNGs, CropSpot/FitSpot figs
- Matlab AllFits format: 38 columns (x,y,z,h,wx,wy,wz,a,b,xL,xU,...,fid_x,fid_y,fid_z,fid_h)
- Note: Matlab `locusX` in AllFits is in **nm** (locusX_px * 108), selectSpots `locusX` is in **pixels**

### 4. Python V4_test output directory
- **Output:** `.../ChrTracer3/analysis_260328_v4test/`

---

## Pipeline parameters (run_pipeline_v4.py)

### Drift correction
- ref_hyb=1, fid_ch=0, n_ch=2
- ds=4 (downsample), crop=150 (fine window), spot_percentile=75, max_fine_shift=5
- DRIFT_WORKERS=4

### Spot detection
- threshold_pct=0.997, bg_size=50, min_dist=30, downsample=3, border=2
- Previous test: FOV1 detected 142 spots (Matlab detected 214 — different threshold/algorithm)

### 3D Gaussian fitting
- nm_xy=108, nm_z=150, box_half=15, fit_half_xy=4, fit_half_z=6
- Quality filters: min_h=200, min_hb_ratio=1.2, min_ah_ratio=0.25
- Width filters: max_wx_px=3.0, max_wz_sl=9.0, max_xy_step=12.0
- FIT_WORKERS=8

---

## After pipeline completes: comparison with Matlab

A comparison script is needed. Key points:
1. **Drift comparison:** Compare `analysis_260328_v4test/fov###_regData.csv` vs `analysis_240410_masked/fov###_regData.csv`
2. **Spot matching:** Match Python spots to nearest Matlab spots (locusX, locusY in pixels)
3. **Fit comparison:** For matched spots, compare x/y/z, wx/wy/wz, h, fitQuality
   - **Sigma convention:** Python wx = sqrt(2) * Matlab wx (different Gaussian parameterization)
   - Matlab outputs x/y/z in nm, wx/wy/wz in nm
   - Python outputs x/y/z in nm, wx/wy in nm, wz in nm

---

## Key files
| File | Purpose |
|------|---------|
| `run_pipeline_v4.py` | Main batch pipeline (drift + spots + fitting) |
| `orca_drift.py` | DAX I/O, drift correction |
| `orca_fit.py` | 3D Gaussian fitting, spot detection, quality filters |
| `ORCA_app.py` | Streamlit web app (interactive mode) |

## Related paths
| Path | Description |
|------|-------------|
| `/dobby/yeqiao/software_code/ChrTracer3_py_app_V4_test/` | V4 test code |
| `/dobby/yeqiao/software_code/ChrTracer3_py_app_V3/` | V3 code (base) |
| `.../ChrTracer3/analysis_260328_v4test/` | V4 test output |
| `.../ChrTracer3/analysis_240410_masked/` | Matlab reference output |
| `.../260318_Convert_ChrTracer3_toPython/v4test_pipeline.log` | Pipeline stdout log |
