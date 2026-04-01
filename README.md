# ChrTracer3 Python V4

A Python reimplementation of the Matlab ChrTracer3 chromatin tracing pipeline for ORCA (Optical Reconstruction of Chromatin Architecture) experiments. V4 is optimized to match Matlab ChrTracer3 output within measurement noise.

## Overview

The pipeline processes multi-hyb, multi-FOV 3D image stacks and produces 3D Gaussian-fitted spot positions for each genomic locus across hybridization rounds.

## Requirements

- Python 3.9+ (3.11 recommended)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) (recommended for environment management)

## Installation

### 1. Install Miniconda (if not already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts, then restart your terminal.

### 2. Create the conda environment and install dependencies

```bash
conda create -n ChrTracerPy python=3.11 -y
conda activate ChrTracerPy
```

### 3. Clone the repository and install packages

```bash
git clone https://github.com/yeqiaoz/ChrTracer3-Python-V4.git
cd ChrTracer3-Python-V4
pip install -r requirements.txt
```

## Source files

| File | Description |
|------|-------------|
| `ORCA_app.py` | Streamlit interactive UI (all pipeline steps) |
| `orca_drift.py` | Data conversion (Step 0) and drift correction (Step 1) |
| `orca_fit.py` | Spot detection (Step 2) and 3D Gaussian fitting (Step 3) |
| `run_pipeline_v4.py` | Headless pipeline orchestration script |

## Usage

**Important:** Always activate the conda environment before running:
```bash
conda activate ChrTracerPy
```

### Interactive mode (Streamlit UI)

```bash
streamlit run ORCA_app.py
```

The Streamlit interface provides interactive control over all pipeline steps, including Step 0 data conversion. The UI guides you through each step with parameter controls and live visualization.

### Headless mode (command line)

Edit `DAX_DIR`, `OUT_DIR`, and other parameters at the top of `run_pipeline_v4.py`, then run:

```bash
python run_pipeline_v4.py
```

**Key parameters to configure:**

| Variable | Description |
|----------|-------------|
| `DAX_DIR` | Path to the `ChrTracer3/` directory containing `Readout_NNN/` folders with `.dax`/`.inf` files |
| `OUT_DIR` | Output directory for results |
| `N_FOVS` | Number of FOVs to process |
| `DRIFT_WORKERS` | Parallel workers for drift correction (default: 4) |
| `FIT_WORKERS` | Parallel workers for fitting within each FOV (default: 8) |

The pipeline automatically resumes from the last checkpoint by skipping FOVs that already have output CSVs.

### Input data format

**Step 0 input** (Vutara SRX raw data):
```
experiment_root/
  Location-NN/
    Raw Images/
      frameinfo.csv       # frame -> timepoint/Z/probe mapping
      data.json            # image dimensions
      img*.dat             # raw binary frames (uint16)
```

**Step 1-3 input** (ChrTracer3 format, output of Step 0):
```
ChrTracer3/
  Readout_001/
    ConvZscan_01.dax      # image stack (uint16 binary)
    ConvZscan_01.inf      # metadata (dimensions, frame count)
  Readout_002/
  ...
```

### Output

```
output_dir/
  fov001_regData.csv      # drift correction (XY shifts per hyb)
  fov001_selectSpots.csv  # detected spot positions
  fov001_allFits.csv      # per-spot per-hyb fit results
  ...
  allFits.csv             # merged results across all FOVs
```

**`allFits.csv` columns:**

| Column | Description |
|--------|-------------|
| `fov` | Field of view index |
| `spot_id` | Spot index within FOV |
| `hybe` | Hybridization round |
| `x`, `y`, `z` | Fitted position (nm) |
| `h`, `h_fit` | Raw peak intensity and fitted amplitude |
| `wx`, `wy`, `wz` | Gaussian widths (nm, Matlab convention) |
| `bg` | Background level |
| `status` | Quality flag: `ok`, `low_amp`, `low_quality`, `wide_spot`, `no_peak` |

## Pipeline parameters

See the [Technical Documentation](ChrTracer3_V4_technical_documentation.md) for a complete parameter reference with Matlab equivalents.

Key fitting parameters (Matlab-matched defaults):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `nm_xy` | 108 nm | Pixel size XY |
| `nm_z` | 150 nm | Pixel size Z |
| `upsample` | 4 | Fine alignment upsampling factor |
| `max_fine_shift` | 4.0 px | Max per-spot XY drift |
| `max_fine_shift_z` | 6.0 slices | Max per-spot Z drift |
| `fit_half_xy` | 4 px | Fitting sub-crop half-size XY |
| `fit_half_z` | 6 slices | Fitting sub-crop half-size Z |

## V4 optimization summary

Six changes were made to match Matlab ChrTracer3 output:

1. **3D fine alignment** -- added Z drift correction to per-spot alignment (Z MAD: 1686 nm -> 86 nm)
2. **Shift sign fix** -- corrected sign of fine alignment shifts
3. **Restricted region search** -- match Matlab's restricted peak search before fitting
4. **Gaussian width convention** -- divide by sqrt(2) to match Matlab convention (width ratio: 1.39 -> 1.02)
5. **Fitting bounds** -- matched Matlab sigma bounds, peak position bounds, and initial guesses
6. **Drift correction rewrite** -- image normalization, adaptive downsampling, gradMax sub-pixel refinement

See [CHANGELOG_v4_optimization.md](CHANGELOG_v4_optimization.md) for details and [ChrTracer3_V4_technical_documentation.md](ChrTracer3_V4_technical_documentation.md) for full algorithm descriptions.

## Documentation

- [ChrTracer3_V4_technical_documentation.md](ChrTracer3_V4_technical_documentation.md) -- full technical documentation of algorithms, parameters, coordinate conventions, and Matlab comparison
- [CHANGELOG_v4_optimization.md](CHANGELOG_v4_optimization.md) -- detailed changelog of V4 optimization changes and results
- [optimization_plan.md](optimization_plan.md) -- original optimization plan with root cause analysis

## Reference

1. Mateo LJ, Murphy SE, Hafner A, Cinquini IS, Walker CA, Boettiger AN. Visualizing DNA folding and RNA in embryos at single-cell resolution. *Nature*. 2019;568(7750):49-54. doi:10.1038/s41586-019-1035-4

2. Boettiger AN, Murphy SE. Advances in chromatin imaging at kilobase-scale resolution. *Trends in Genetics*. 2020;36(4):273-287. doi:10.1016/j.tig.2019.12.010

3. Zhou Y, Jay A, Burget N, Friedrich T, Yoon S, Alsing J, Nir G, Grosschedl R, Vahedi G, Faryabi RB. Lineage-determining transcription factors constrain cohesin to drive multi-enhancer oncogene regulation. *Nat Cell Biol*. 2026;28(1):149-165. doi:10.1038/s41556-025-01827-2

4. ORCA-public Matlab pipeline: https://github.com/BoettigerLab/ORCA-public
