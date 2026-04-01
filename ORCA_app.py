#!/usr/bin/env python3
"""
ChrTracer3 Python — ORCA Analysis App
Streamlit-based replacement for the ChrTracer3 Matlab GUI.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import orca_drift
import orca_fit

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ChrTracer3 Python",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom button colour palette
#   ct-exec  = execution (teal-blue)   — triggers computation
#   ct-skip  = skip/checkbox (amber)   — loads pre-existing results
#   ct-nav   = navigate/save (emerald) — proceed or save per-FOV
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Default primary: override Streamlit's coral/red */
button[data-testid="baseButton-primary"] {
    background-color: #0284C7; border-color: #0284C7; color: white;
}
button[data-testid="baseButton-primary"]:hover {
    background-color: #0369A1; border-color: #0369A1;
}

/* Execution — teal-blue */
[data-testid="stMarkdownContainer"]:has(.ct-exec) + [data-testid="stButton"] button {
    background-color: #0284C7 !important; border-color: #0284C7 !important; color: white !important;
}
[data-testid="stMarkdownContainer"]:has(.ct-exec) + [data-testid="stButton"] button:hover {
    background-color: #0369A1 !important; border-color: #0369A1 !important;
}

/* Skip / load-existing — amber */
[data-testid="stMarkdownContainer"]:has(.ct-skip) + [data-testid="stButton"] button {
    background-color: #D97706 !important; border-color: #D97706 !important; color: white !important;
}
[data-testid="stMarkdownContainer"]:has(.ct-skip) + [data-testid="stButton"] button:hover {
    background-color: #B45309 !important; border-color: #B45309 !important;
}

/* Navigate / proceed / save — emerald */
[data-testid="stMarkdownContainer"]:has(.ct-nav) + [data-testid="stButton"] button {
    background-color: #059669 !important; border-color: #059669 !important; color: white !important;
}
[data-testid="stMarkdownContainer"]:has(.ct-nav) + [data-testid="stButton"] button:hover {
    background-color: #047857 !important; border-color: #047857 !important;
}

/* Multiselect tags — purple */
span[data-baseweb="tag"] {
    background-color: #7C3AED !important;
}
span[data-baseweb="tag"] span {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


def _exec_button(label, **kw):
    """Execution button (teal-blue): triggers computation."""
    st.markdown('<div class="ct-exec"></div>', unsafe_allow_html=True)
    kw.setdefault("type", "primary")
    return st.button(label, **kw)


def _skip_button(label, **kw):
    """Skip/load-existing button (amber): loads pre-computed results."""
    st.markdown('<div class="ct-skip"></div>', unsafe_allow_html=True)
    kw.setdefault("type", "primary")
    return st.button(label, **kw)


def _nav_button(label, **kw):
    """Navigation/save button (emerald): proceed or save per-FOV."""
    st.markdown('<div class="ct-nav"></div>', unsafe_allow_html=True)
    kw.setdefault("type", "primary")
    return st.button(label, **kw)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

STEPS = [
    "0. Convert Raw Data",
    "1. Load Files",
    "2. Drift Correction",
    "3. Validate Drift",
    "4. Select Test Spot",
    "5. Fit Test Spot",
    "6. Auto Detect All Spots",
    "7. Validate Picked Spots",
    "8. Fit All",
    "9. Export to OLIVE",
    "10. Impute & QC",
]

if "current_step" not in st.session_state:
    st.session_state.current_step = 0          # index into STEPS

if "raw_data_dir" not in st.session_state:
    st.session_state.raw_data_dir = None       # Path — Vutara experiment root (contains Location-XX/)

if "exp_layout" not in st.session_state:
    st.session_state.exp_layout = None         # DataFrame

if "chrtracer_dir" not in st.session_state:
    st.session_state.chrtracer_dir = None      # Path

if "output_dir" not in st.session_state:
    st.session_state.output_dir = None         # Path

if "dax_files" not in st.session_state:
    st.session_state.dax_files = {}            # {readout_folder: {loc: Path}}

if "missing_dax" not in st.session_state:
    st.session_state.missing_dax = []

if "reg_data" not in st.session_state:
    st.session_state.reg_data = {}    # {fov: DataFrame}

if "drift_overrides" not in st.session_state:
    st.session_state.drift_overrides = {}   # {fov: edited DataFrame}

if "test_spot" not in st.session_state:
    st.session_state.test_spot = None       # {fov, hyb, x, y}

if "maxproj_cache" not in st.session_state:
    st.session_state.maxproj_cache = {}     # {(fov, hyb, ch): ndarray}

if "test_fit_rows" not in st.session_state:
    st.session_state.test_fit_rows = []     # list of row dicts from fit_test_spot_stream

if "select_spots" not in st.session_state:
    st.session_state.select_spots = {}      # {fov: DataFrame with locusX, locusY}

if "last_spot_click" not in st.session_state:
    st.session_state.last_spot_click = None  # (fov, x, y) of last processed click

if "all_fits" not in st.session_state:
    st.session_state.all_fits = {}           # {fov: DataFrame}

# Step 8 per-FOV-per-rerun fitting state
if "fit_running" not in st.session_state:
    st.session_state.fit_running = False
if "fit_fov_idx" not in st.session_state:
    st.session_state.fit_fov_idx = 0
if "fit_stop_requested" not in st.session_state:
    st.session_state.fit_stop_requested = False
if "fit_all_rows" not in st.session_state:
    st.session_state.fit_all_rows = []       # accumulating row dicts across FOVs
if "fit_by_fov" not in st.session_state:
    st.session_state.fit_by_fov = {}         # {fov: DataFrame} partial results
if "fit_params_snapshot" not in st.session_state:
    st.session_state.fit_params_snapshot = {}  # params frozen at Run time

if "olive_keep_hybs" not in st.session_state:
    st.session_state.olive_keep_hybs = []    # list of hyb ints selected in Step 9

if "step0_done" not in st.session_state:
    st.session_state.step0_done = False
if "fov_list" not in st.session_state:
    st.session_state.fov_list = []
if "step0_running" not in st.session_state:
    st.session_state.step0_running = False

# ---------------------------------------------------------------------------
# Sidebar — step navigator
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("ChrTracer3 Python")
    st.caption("Faryabi Lab")
    _hn  = socket.gethostname().lower()
    _srv = next((s.capitalize() for s in ["dobby","aries","simurgh","plutus","argonauts"]
                 if s in _hn), socket.gethostname())
    st.caption(f"⚙ Running on **{_srv}**")
    svg_path = Path(__file__).parent / "BOTH_2.svg"
    png_path = Path(__file__).parent / "BOTH_2.png"
    if svg_path.exists():
        try:
            import base64
            svg_b64 = base64.b64encode(svg_path.read_bytes()).decode()
            st.markdown(
                f'<img src="data:image/svg+xml;base64,{svg_b64}" width="160">',
                unsafe_allow_html=True,
            )
        except Exception:
            pass
    elif png_path.exists():
        try:
            st.image(str(png_path), width=160)
        except Exception:
            pass
    st.divider()
    st.subheader("Pipeline Steps")

    for i, step in enumerate(STEPS):
        if i == st.session_state.current_step:
            st.markdown(f"**▶ {step}**")
        elif i < st.session_state.current_step:
            st.markdown(f"~~{step}~~ ✓")
        else:
            st.markdown(f"<span style='color:gray'>{step}</span>", unsafe_allow_html=True)

    st.divider()

    # Quick jump (only to completed or current steps)
    if st.session_state.current_step > 0:
        jump = st.selectbox(
            "Jump to step",
            options=list(range(st.session_state.current_step + 1)),
            format_func=lambda i: STEPS[i],
            index=st.session_state.current_step,
        )
        if jump != st.session_state.current_step:
            st.session_state.current_step = jump
            st.rerun()

    # Show loaded experiment info
    st.divider()
    st.caption("**References**")
    st.caption("[PMID 33619390](https://pubmed.ncbi.nlm.nih.gov/33619390/) · ChrTracer3")
    st.caption("[PMID 41118766](https://pubmed.ncbi.nlm.nih.gov/41118766/) · OLIVE")
    st.caption("[PMID 41331087](https://pubmed.ncbi.nlm.nih.gov/41331087/) · Faryabi Lab ORCA")

# ---------------------------------------------------------------------------
# Step router
# ---------------------------------------------------------------------------

step = st.session_state.current_step


# ============================================================
# STEP 0 — Convert Raw Data (.dat → DAX)
# ============================================================

if step == 0:
    st.header("Step 0 — Convert Raw Data")

    # --- Server / user / path badge ---
    _hostname = socket.gethostname().lower()
    _known    = ["dobby", "aries", "simurgh", "plutus", "argonauts"]
    _server   = next((s.capitalize() for s in _known if s in _hostname), _hostname)
    _colors   = {"Dobby": "#0284C7", "Aries": "#059669", "Simurgh": "#7C3AED",
                 "Plutus": "#D97706", "Argonauts": "#DC2626"}
    _color    = _colors.get(_server, "#6B7280")
    _user     = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
    _cwd      = os.getcwd()
    st.markdown(
        f'<span style="background:{_color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600;">⚙ {_server}</span>'
        f'&nbsp;&nbsp;<span style="color:#6B7280;font-size:0.85em;">{_user}</span>'
        f'&nbsp;<span style="color:#9CA3AF;font-size:0.85em;">·</span>&nbsp;'
        f'<code style="font-size:0.82em;color:#4B5563;">{_cwd}</code>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    st.markdown(
        "- Convert Vutara raw `.dat` files directly to ChrTracer3 `.dax` format, "
        "bypassing the Vutara SRX export step.\n"
        "- Reads `frameinfo.csv` to map frames to hyb rounds and channels."
    )

    # ── Inputs ───────────────────────────────────────────────
    default_raw = str(st.session_state.raw_data_dir or "")
    raw_input = st.text_input(
        "Raw data root (contains Location-XX/ folders)",
        value=default_raw,
        placeholder="/mnt/data2/yeqiao/experiment/2024-04-02-.../",
    )
    raw_path = Path(raw_input).expanduser() if raw_input.strip() else None
    if raw_path and not raw_path.exists():
        st.error(f"Folder not found: `{raw_path}`")
        raw_path = None

    default_out = str(st.session_state.chrtracer_dir or "")
    out_input = st.text_input(
        "ChrTracer3 output folder (DAX files will be written here)",
        value=default_out,
        placeholder="/mnt/data2/yeqiao/analysis/.../ChrTracer3",
    )
    out_path = Path(out_input).expanduser() if out_input.strip() else None
    if out_path and not out_path.exists():
        st.caption(f"Output folder will be created: `{out_path}`")

    # ── Auto-detect locations and hybs from raw folder ──────
    loc_dirs = []
    if raw_path and raw_path.exists():
        loc_dirs = sorted(raw_path.glob("Location-*/"))
        if loc_dirs:
            st.success(f"Found {len(loc_dirs)} location(s): "
                       f"{[d.name for d in loc_dirs[:5]]}"
                       + (" …" if len(loc_dirs) > 5 else ""))
        else:
            st.warning("No `Location-XX` folders found in this directory.")

    # Read frameinfo from first location to show hyb count
    n_hybs_detected = None
    H_detected = W_detected = None
    if loc_dirs:
        fi_path = loc_dirs[0] / "Raw Images" / "frameinfo.csv"
        dj_path = loc_dirs[0] / "Raw Images" / "data.json"
        if fi_path.exists():
            import json
            fi = pd.read_csv(fi_path)
            n_hybs_detected = fi["Timepoint"].nunique()
            n_z_detected    = fi["ZPos"].nunique()
            n_probes        = fi["Probe"].nunique()
            st.info(
                f"Detected: **{n_hybs_detected} hyb rounds** · "
                f"**{n_z_detected} Z slices** · "
                f"**{n_probes} channels** (Probe 0 = fiducial, Probe 1 = readout)"
            )
        if dj_path.exists():
            with open(dj_path) as f:
                dj = json.load(f)
            val = dj.get("value", {})
            H_detected = int(val.get("WidefieldImageDimY", 0)) or None
            W_detected = int(val.get("WidefieldImageDimX", 0)) or None

    # ── Parameters ───────────────────────────────────────────
    st.subheader("Parameters")
    c1, c2, c3 = st.columns(3)
    write_tiff = c1.checkbox("Also write multi-page TIFFs", value=False,
                              help="Writes TIFFs/Location-XX/output_TXXX.tif alongside DAX")
    resume_mode = st.checkbox(
        "Resume — skip readouts whose DAX files already exist",
        value=True,
        help="Useful after a disconnection. Each Readout_XXX/ConvZscan_YY.dax is checked individually.",
    )
    locs_select = st.multiselect(
        "Locations to convert",
        options=[d.name for d in loc_dirs],
        default=[d.name for d in loc_dirs],
    ) if loc_dirs else []

    # Show done/pending summary when resume is on and output path is set
    if resume_mode and out_path and out_path.exists() and loc_dirs and n_hybs_detected:
        done_count = pending_count = 0
        for loc_name in locs_select:
            loc_num = int(loc_name.split("-")[1])
            for tp in range(n_hybs_detected):
                dax = out_path / f"Readout_{tp+1:03d}" / f"ConvZscan_{loc_num:02d}.dax"
                if dax.exists():
                    done_count += 1
                else:
                    pending_count += 1
        total_tasks_preview = done_count + pending_count
        if done_count > 0:
            st.info(
                f"Resume scan: **{done_count}/{total_tasks_preview}** DAX files already exist "
                f"— **{pending_count}** remaining."
            )

    col_run, col_skip, _ = st.columns([1, 2, 4])
    with col_run:
        run_clicked = _exec_button("Convert", disabled=not (raw_path and out_path and loc_dirs))
    with col_skip:
        if _skip_button("⏭ Skip — DAX files already exist"):
            if out_path:
                st.session_state.chrtracer_dir = out_path
            st.session_state.current_step = 1
            st.rerun()

    st.divider()

    # ── Conversion ───────────────────────────────────────────
    if run_clicked:
        st.session_state.step0_running = True
        st.session_state.step0_params  = {
            "locs_select": locs_select,
            "raw_path": str(raw_path),
            "out_path": str(out_path),
            "resume_mode": resume_mode,
            "write_tiff": write_tiff,
        }
        st.rerun()

    if st.session_state.step0_running:
        import json

        p = st.session_state.step0_params
        raw_path  = Path(p["raw_path"])
        out_path  = Path(p["out_path"])
        resume_mode = p["resume_mode"]
        write_tiff  = p["write_tiff"]
        locs_select = p["locs_select"]

        out_path.mkdir(parents=True, exist_ok=True)
        st.session_state.raw_data_dir  = raw_path
        st.session_state.chrtracer_dir = out_path

        selected_dirs = [raw_path / n for n in locs_select]

        # ── Build task groups (one group per location) ───────────────
        loc_task_groups = []   # list[list[task_tuple]]
        skipped = 0
        for loc_dir in selected_dirs:
            loc_name = loc_dir.name
            raw_img  = loc_dir / "Raw Images"
            fi_path  = raw_img / "frameinfo.csv"
            dj_path  = raw_img / "data.json"

            if not fi_path.exists():
                st.warning(f"{loc_name}: frameinfo.csv not found, skipping.")
                continue

            fi  = pd.read_csv(fi_path)
            with open(dj_path) as f:
                dj = json.load(f)
            val         = dj.get("value", {})
            H           = int(val["WidefieldImageDimY"])
            W           = int(val["WidefieldImageDimX"])
            frame_bytes = H * W * 2
            loc_num     = int(loc_name.split("-")[1])

            dat_files            = sorted(raw_img.glob("img*.dat"))
            frames_per_file_list = [f.stat().st_size // frame_bytes for f in dat_files]
            dat_file_strs        = [str(f) for f in dat_files]

            n_hybs    = fi["Timepoint"].nunique()
            loc_tasks = []
            for tp in range(n_hybs):
                readout_num  = tp + 1
                dax_path_out = out_path / f"Readout_{readout_num:03d}" / f"ConvZscan_{loc_num:02d}.dax"

                if resume_mode and dax_path_out.exists():
                    skipped += 1
                    continue

                tp_frames       = fi[fi["Timepoint"] == tp].sort_values("GlobalIndex")
                fid_idx         = tp_frames[tp_frames["Probe"] == 0].sort_values("ZPos")["GlobalIndex"].tolist()
                read_idx        = tp_frames[tp_frames["Probe"] == 1].sort_values("ZPos")["GlobalIndex"].tolist()
                interleaved_idx = [idx for pair in zip(fid_idx, read_idx) for idx in pair]

                loc_tasks.append((
                    loc_name, loc_num, readout_num, interleaved_idx,
                    dat_file_strs, frames_per_file_list, H, W, frame_bytes,
                    str(out_path), write_tiff,
                ))

            if loc_tasks:
                loc_task_groups.append(loc_tasks)

        total_tasks = sum(len(g) for g in loc_task_groups) + skipped
        done        = [skipped]   # list so inner function can mutate it

        progress_bar = st.progress(done[0] / max(total_tasks, 1), text="Starting…")
        status_text  = st.empty()
        log_area     = st.empty()
        log_lines    = []

        def _update_ui(loc_name, readout_num, dax_path_str, n_frames, err):
            done[0] += 1
            if err:
                log_lines.append(f"✗ {loc_name} Readout_{readout_num:03d}  ERROR: {err}")
            else:
                log_lines.append(f"✓ {loc_name} Readout_{readout_num:03d}  →  {Path(dax_path_str).name}")
            if len(log_lines) > 12:
                log_lines[:] = log_lines[-12:]
            msg = f"{done[0]}/{total_tasks} — {loc_name} Readout {readout_num:03d}"
            progress_bar.progress(done[0] / max(total_tasks, 1), text=msg)
            if not err:
                status_text.markdown(f"**{msg}** — {n_frames} frames written")
            else:
                status_text.error(f"{msg} — {err}")
            log_area.code("\n".join(log_lines))

        try:
            # Sequential: one location at a time (I/O-bound, no benefit from parallelism)
            for loc_tasks in loc_task_groups:
                for result in orca_drift.convert_one_location(loc_tasks):
                    _update_ui(*result)

            progress_bar.progress(1.0, text="Done!")
            status_text.success(
                f"Conversion complete — {len(selected_dirs)} location(s) × {n_hybs_detected} hybs."
            )
            st.session_state.step0_done = True
            st.session_state.num_locs   = len(selected_dirs)
        finally:
            st.session_state.step0_running = False

    # ── Navigate ─────────────────────────────────────────────
    st.divider()
    if st.session_state.step0_done or st.session_state.chrtracer_dir is not None:
        if _nav_button("Proceed to Load Files →"):
            st.session_state.current_step = 1
            st.rerun()


# ============================================================
# STEP 1 — Load Files
# ============================================================

if step == 1:
    st.header("Step 1 — Load Files")

    # --- Server badge ---
    _hostname = socket.gethostname().lower()
    _known    = ["dobby", "aries", "simurgh", "plutus", "argonauts"]
    _server   = next((s.capitalize() for s in _known if s in _hostname), _hostname)
    _colors   = {"Dobby": "#0284C7", "Aries": "#059669", "Simurgh": "#7C3AED",
                 "Plutus": "#D97706", "Argonauts": "#DC2626"}
    _color    = _colors.get(_server, "#6B7280")
    _user = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
    _cwd  = os.getcwd()
    st.markdown(
        f'<span style="background:{_color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600;">⚙ {_server}</span>'
        f'&nbsp;&nbsp;<span style="color:#6B7280;font-size:0.85em;">{_user}</span>'
        f'&nbsp;<span style="color:#9CA3AF;font-size:0.85em;">·</span>&nbsp;'
        f'<code style="font-size:0.82em;color:#4B5563;">{_cwd}</code>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    st.markdown(
        "Set the ChrTracer3 output folder. "
        "The app will auto-discover `Readout_*` sub-folders and verify that all expected `.dax` files are present."
    )

    # --- Folder input ---
    default_dir = str(st.session_state.chrtracer_dir or "")
    folder_input = st.text_input(
        "ChrTracer3 folder path",
        value=default_dir,
        placeholder="/mnt/data2/yeqiao/analysis/.../ChrTracer3",
    )

    folder_path = Path(folder_input).expanduser() if folder_input.strip() else None

    if folder_path and not folder_path.exists():
        st.error(f"Folder not found: `{folder_path}`")
        folder_path = None

    # --- Auto-discover Readout_* folders ---
    readout_dirs = []
    if folder_path and folder_path.exists():
        readout_dirs = sorted(folder_path.glob("Readout_*/"), key=lambda p: p.name)
        if not readout_dirs:
            st.warning("No `Readout_*` folders found in this directory.")
        else:
            st.caption(
                f"Found **{len(readout_dirs)}** readout folders "
                f"(`{readout_dirs[0].name}` – `{readout_dirs[-1].name}`)"
            )

    # --- Channel layout ---
    st.markdown("**Channel layout**")
    st.radio(
        "Channel layout",
        options=["2-channel: ch0 = fiducial (565 nm), ch1 = readout (647 nm)"],
        index=0,
        label_visibility="collapsed",
    )
    st.caption("🚧 3-channel (ch0 = fiducial 565 nm, ch1 = readout 647 nm, ch2 = readout 750 nm) — under development")

    # --- Output path ---
    default_out = str(st.session_state.output_dir or "")
    output_input = st.text_input(
        "Output folder path",
        value=default_out,
        placeholder="/mnt/data2/yeqiao/analysis/.../analysis_YYYYMMDD",
    )
    output_path = Path(output_input).expanduser() if output_input.strip() else None
    if output_path and not output_path.exists():
        st.caption(f"Output folder will be created: `{output_path}`")

    # --- Number of locations (auto-detected or manual) ---
    # Auto-detect FOV numbers from first Readout folder
    auto_fovs = []
    if readout_dirs:
        import re
        for dax_file in sorted(readout_dirs[0].glob("ConvZscan_*.dax")):
            m = re.search(r"ConvZscan_(\d+)\.dax", dax_file.name)
            if m:
                auto_fovs.append(int(m.group(1)))
    if auto_fovs:
        st.caption(f"Auto-detected FOVs: {auto_fovs}")
    num_locs = st.number_input("Number of locations (FOVs)", min_value=1,
                               value=st.session_state.get("num_locs", len(auto_fovs) or 1), step=1)

    # --- Load button ---
    col1, col2 = st.columns([1, 4])
    with col1:
        load_clicked = _exec_button("Load", use_container_width=True)

    if load_clicked:
        if folder_path is None or not readout_dirs:
            st.error("Please specify a valid folder containing `Readout_*` sub-folders.")
        elif output_path is None:
            st.error("Please specify an output folder path.")
        else:
            with st.spinner("Scanning readout folders and .dax files…"):
                # Build layout DataFrame from discovered folders
                df = pd.DataFrame({"FolderName": [d.name for d in readout_dirs]})
                st.session_state.exp_layout = df
                st.session_state.chrtracer_dir = folder_path
                st.session_state.output_dir = output_path

                # Auto-detect FOV numbers from DAX files in first Readout folder
                import re as _re
                detected_fovs = set()
                for rd in readout_dirs:
                    for dax_file in rd.glob("ConvZscan_*.dax"):
                        m = _re.search(r"ConvZscan_(\d+)\.dax", dax_file.name)
                        if m:
                            detected_fovs.add(int(m.group(1)))
                fov_list = sorted(detected_fovs) if detected_fovs else list(range(1, int(num_locs) + 1))
                st.session_state.fov_list = fov_list
                st.session_state.num_locs = len(fov_list)

                # Scan for dax files
                dax_files = {}
                missing = []
                for folder_name in df["FolderName"]:
                    readout_folder = folder_path / folder_name
                    loc_daxes = {}
                    for loc in fov_list:
                        dax_path = readout_folder / f"ConvZscan_{loc:02d}.dax"
                        if dax_path.exists():
                            loc_daxes[loc] = dax_path
                        else:
                            missing.append(str(dax_path.relative_to(folder_path)))
                    dax_files[folder_name] = loc_daxes

                st.session_state.dax_files = dax_files
                st.session_state.missing_dax = missing

            st.success("Loaded successfully!")
            st.rerun()

    # --- Show results if already loaded ---
    if st.session_state.exp_layout is not None and st.session_state.chrtracer_dir == folder_path:
        df = st.session_state.exp_layout
        missing = st.session_state.missing_dax

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hyb rounds", len(df))
        col2.metric("FOVs (locations)", st.session_state.get("num_locs", "?"))
        total_dax = sum(len(v) for v in st.session_state.dax_files.values())
        col3.metric("DAX files found", total_dax)
        col4.metric("Missing DAX files", len(missing), delta=None if len(missing) == 0 else f"⚠ {len(missing)}")

        # Missing file warning
        if missing:
            with st.expander(f"⚠ {len(missing)} missing .dax files", expanded=False):
                for m in missing[:50]:
                    st.code(m)
                if len(missing) > 50:
                    st.caption(f"… and {len(missing) - 50} more")
        else:
            st.success(f"All {total_dax} .dax files present.")

        # Proceed button
        st.divider()
        col_back, col_fwd, _ = st.columns([1, 2, 5], vertical_alignment="bottom")
        if col_back.button("← Back"):
            st.session_state.current_step = 0
            st.rerun()
        with col_fwd:
            if _nav_button("Proceed to Drift Correction →"):
                st.session_state.current_step = 2
                st.rerun()


# ============================================================
# STEP 2 — Drift Correction
# ============================================================

elif step == 2:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.header("Step 2 — Drift Correction")
    st.markdown(
        "- **Fine spot selection:** The fine alignment crop is centred on the fiducial bead "
        "at the chosen brightness percentile (default 75th) rather than the single brightest pixel. "
        "The very brightest spot is unreliable due to abnormal signal aggregation "
        "or hot pixels that do not represent true fiducial beads.\n"
        "- **Fine shift validation:** After fine alignment, if either axis of the fine shift "
        "exceeds the maximum allowed value, the fine result is discarded and only the coarse shift "
        "is used. A large fine shift indicates the fine cross-correlation found a false peak "
        "(the fine step should only correct the small sub-pixel residual left by coarse alignment)."
    )

    if st.session_state.chrtracer_dir is None or st.session_state.exp_layout is None:
        st.error("Complete Step 1 first.")
        st.stop()

    df_layout     = st.session_state.exp_layout
    chrtracer_dir = st.session_state.chrtracer_dir
    output_dir    = st.session_state.output_dir
    num_locs      = st.session_state.get("num_locs", 1)

    readout_folders = [chrtracer_dir / row["FolderName"]
                       for _, row in df_layout.iterrows()]
    n_hybs = len(readout_folders)

    # ── Skip option: load existing regData ──────────────────
    if output_dir is not None:
        existing = sorted(output_dir.glob("fov*_regData.csv"))
        if existing:
            st.success(
                f"Found {len(existing)} existing regData file(s) in output folder."
            )
            if _skip_button(
                f"⏭ Load existing drift data and skip to Step 4  ({len(existing)}/{num_locs} FOVs)",
            ):
                loaded = {}
                for csv in existing:
                    fov_n = int(csv.stem.replace("fov", "").replace("_regData", ""))
                    loaded[fov_n] = pd.read_csv(csv)
                st.session_state.reg_data = loaded
                st.session_state.current_step = 4   # jump to Step 4
                st.rerun()
            st.divider()

    # ── Parameters ──────────────────────────────────────────
    st.subheader("Parameters")

    # Row 1: FOV selection + coarse/fine tuning
    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
    _fov_options = st.session_state.get("fov_list", list(range(1, num_locs + 1)))
    fov_select = col1.multiselect(
        "FOVs to process",
        _fov_options,
        default=_fov_options,
        format_func=lambda f: f"FOV {f:02d}",
    )
    ds_factor       = col2.number_input("Coarse downsample", min_value=1, value=4, step=1)
    crop_size       = col3.number_input("Fine crop (px)", min_value=20, value=150, step=10)
    spot_percentile = col4.number_input(
        "Fine spot percentile",
        min_value=0, max_value=100, value=75, step=5,
        help="Brightness percentile of the fiducial spot used for fine alignment. "
             "100 = brightest (fragile to signal aggregation / hot pixels). "
             "75 = default, avoids outliers while staying bright.",
    )
    max_fine_shift  = col5.number_input(
        "Max fine shift (px)",
        min_value=1, value=5, step=1,
        help="If the fine shift on either axis exceeds this value the fine result "
             "is discarded and only the coarse shift is used.",
    )

    # Row 2: checkboxes
    col_savefig, col_liveimg, _ = st.columns([1, 1, 5])
    save_figs  = col_savefig.checkbox("Save alignment figures", value=True)
    show_live  = col_liveimg.checkbox("Show live image update", value=True)

    # Row 3: action buttons aligned at the bottom
    col_run, col_back, _ = st.columns([1, 1, 6], vertical_alignment="bottom")
    with col_run:
        run_clicked = _exec_button("Run")
    if col_back.button("← Back"):
        st.session_state.current_step = 1
        st.rerun()

    st.divider()

    # ── Processing ───────────────────────────────────────────
    if run_clicked:
        if output_dir is None:
            st.error("No output folder set — go back to Step 1.")
            st.stop()
        output_dir.mkdir(parents=True, exist_ok=True)

        if not fov_select:
            st.warning("Select at least one FOV.")
            st.stop()

        _drift_kwargs = dict(
            readout_folders=readout_folders,
            output_dir=output_dir,
            fid_ch=0, n_ch=2,
            ds=int(ds_factor),
            crop=int(crop_size),
            spot_percentile=float(spot_percentile),
            max_fine_shift=int(max_fine_shift),
            save_figures=bool(save_figs),
        )

        progress_bar = st.progress(0, text="Starting…")
        status_text  = st.empty()
        live_img     = st.empty() if show_live else None
        live_plot    = st.empty()

        for fov_i, fov in enumerate(fov_select):
            fov = int(fov)
            status_text.subheader(f"Running — FOV {fov:02d} ({fov_i+1}/{len(fov_select)})")
            rows_so_far = []

            try:
                gen = orca_drift.correct_one_fov_stream(fov=fov, **_drift_kwargs)

                for hyb, n_total, row, fig_path, rows_so_far in gen:
                    pct = (fov_i + hyb / n_total) / len(fov_select)
                    progress_bar.progress(pct,
                        text=f"FOV {fov:02d} ({fov_i+1}/{len(fov_select)}) — hyb {hyb}/{n_total}")

                    if live_img is not None and fig_path and Path(fig_path).exists():
                        live_img.image(str(fig_path), use_container_width=True)

                    df_so_far = pd.DataFrame(rows_so_far)
                    hybs_done = np.arange(1, len(df_so_far) + 1)
                    tx = df_so_far["xshift"].fillna(0) + df_so_far["xshift2"].fillna(0)
                    ty = df_so_far["yshift"].fillna(0) + df_so_far["yshift2"].fillna(0)

                    fig_drift, ax = plt.subplots(figsize=(10, 2.5))
                    ax.plot(hybs_done, tx, "o-", markersize=3, label="X drift (px)")
                    ax.plot(hybs_done, ty, "s-", markersize=3, label="Y drift (px)")
                    ax.axhline(0, color="gray", linewidth=0.5)
                    ax.set_xlim(1, n_total)
                    ax.set_xlabel("Hyb round")
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    ax.set_ylabel("Shift (px)")
                    ax.set_title(f"FOV {fov:02d} drift — {hyb}/{n_total} hybs")
                    ax.legend(fontsize=8)
                    fig_drift.tight_layout()
                    live_plot.pyplot(fig_drift)
                    plt.close(fig_drift)

            except Exception as exc:
                st.error(f"Error on FOV {fov}: {exc}")
                continue

            if rows_so_far:
                st.session_state.reg_data[fov] = pd.DataFrame(rows_so_far)

        progress_bar.progress(1.0, text="All done!")
        status_text.success(f"Drift correction complete — {len(fov_select)} FOV(s) processed.")

    # ── Results browser (previously processed FOVs) ──────────
    if st.session_state.reg_data:
        reg_data  = st.session_state.reg_data
        fovs_done = sorted(reg_data.keys())

        st.subheader("Results")
        fov_view = st.selectbox(
            "Inspect FOV",
            fovs_done,
            format_func=lambda f: f"FOV {f:02d}",
            key="drift_inspect_fov",
        )
        df_reg = reg_data[fov_view]

        # Drift plot
        hybs    = np.arange(1, len(df_reg) + 1)
        total_x = df_reg["xshift"].fillna(0) + df_reg["xshift2"].fillna(0)
        total_y = df_reg["yshift"].fillna(0) + df_reg["yshift2"].fillna(0)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(hybs, total_x, "o-", markersize=4, label="X drift (px)")
        ax.plot(hybs, total_y, "s-", markersize=4, label="Y drift (px)")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Hyb round")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylabel("Shift (pixels)")
        ax.set_title(f"FOV {fov_view:02d} — total drift per hyb")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("regData table"):
            st.dataframe(df_reg, use_container_width=True)

        # CorrAlign image gallery
        if output_dir is not None:
            imgs = sorted((output_dir / "CorrAlign").glob(
                f"CorrAlign_fov{fov_view:04d}_*.png"
            ))
            if imgs:
                st.subheader(f"CorrAlign images — FOV {fov_view:02d} ({len(imgs)} hybs)")
                cols_per_row = 3
                for i in range(0, len(imgs), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, p in enumerate(imgs[i:i + cols_per_row]):
                        cols[j].image(str(p), caption=p.stem, use_container_width=True)

        st.divider()
        if _nav_button("Proceed to Validate Drift →"):
            st.session_state.current_step = 3
            st.rerun()


# ============================================================
# STEP 3 — Validate Drift
# ============================================================

elif step == 3:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.header("Step 3 — Validate Drift")
    st.markdown(
        "- **Inspecting results:** Drift traces and CorrAlign images are shown here for review. "
        "If a FOV looks incorrect, go back to Step 2, select only that FOV, adjust parameters "
        "(e.g. fine spot percentile or max fine shift), and re-run — all other FOVs are unaffected.\n"
        "- **Manual editing:** Individual hyb shifts can be overridden in the Edit Shifts tab. "
        "Changes are saved to the regData CSV for that FOV and will be used in all downstream steps."
    )

    output_dir = st.session_state.output_dir
    num_locs   = st.session_state.get("num_locs", 1)

    # ── Load regData (session state OR from disk) ────────────
    reg_data = dict(st.session_state.reg_data)   # copy

    if output_dir is not None:
        for fov in st.session_state.get("fov_list", list(range(1, num_locs + 1))):
            if fov not in reg_data:
                csv = output_dir / f"fov{fov:03d}_regData.csv"
                if csv.exists():
                    reg_data[fov] = pd.read_csv(csv)
        if reg_data and not st.session_state.reg_data:
            st.session_state.reg_data = reg_data

    if not reg_data:
        st.warning("No regData found. Run Step 2 first, or check your output folder.")
        if st.button("← Back"):
            st.session_state.current_step = 2
            st.rerun()
        st.stop()

    # Merge with any saved overrides
    for fov, df_edit in st.session_state.drift_overrides.items():
        reg_data[fov] = df_edit

    # ── Overview table ───────────────────────────────────────
    st.subheader("Drift Overview — All FOVs")

    overview_rows = []
    for fov in sorted(reg_data.keys()):
        df = reg_data[fov]
        tx = df["xshift"].fillna(0) + df["xshift2"].fillna(0)
        ty = df["yshift"].fillna(0) + df["yshift2"].fillna(0)
        overview_rows.append({
            "FOV": fov,
            "Max |X| (px)": round(tx.abs().max(), 1),
            "Max |Y| (px)": round(ty.abs().max(), 1),
            "RMS X (px)":   round(float(np.sqrt((tx**2).mean())), 1),
            "RMS Y (px)":   round(float(np.sqrt((ty**2).mean())), 1),
            "Missing hybs": int(df["xshift"].isna().sum()),
        })

    df_overview = pd.DataFrame(overview_rows).set_index("FOV")

    # Color-code max drift: green < 5px, yellow < 15px, red >= 15px
    def _drift_color(val):
        if pd.isna(val): return ""
        if val < 5:  return "background-color: #d4edda"
        if val < 15: return "background-color: #fff3cd"
        return "background-color: #f8d7da"

    _style_func = getattr(df_overview.style, "map", None) or df_overview.style.applymap
    styled = _style_func(_drift_color, subset=["Max |X| (px)", "Max |Y| (px)"])
    st.dataframe(styled, use_container_width=True)

    # ── Per-FOV inspector ────────────────────────────────────
    st.divider()
    st.subheader("Per-FOV Inspector")

    fov_view = st.selectbox(
        "Select FOV",
        sorted(reg_data.keys()),
        format_func=lambda f: f"FOV {f:02d}",
    )
    df_reg = reg_data[fov_view].copy()
    n_hybs = len(df_reg)
    hybs   = np.arange(1, n_hybs + 1)
    tx     = df_reg["xshift"].fillna(0) + df_reg["xshift2"].fillna(0)
    ty     = df_reg["yshift"].fillna(0) + df_reg["yshift2"].fillna(0)

    tab_plot, tab_imgs, tab_edit = st.tabs(["Drift Trace", "CorrAlign Images", "Edit Shifts"])

    # ── Tab 1: Drift trace ───────────────────────────────────
    with tab_plot:
        drift_thresh = st.number_input("Flag threshold (px)", min_value=1, value=15, step=1,
                                        help="Hybs exceeding this drift are highlighted in red.")
        flagged = (tx.abs() > drift_thresh) | (ty.abs() > drift_thresh)

        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        for ax, vals, label, color in zip(
            axes, [tx, ty], ["X drift (px)", "Y drift (px)"], ["steelblue", "darkorange"]
        ):
            ax.plot(hybs, vals, "o-", color=color, markersize=4, linewidth=1)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axhline( drift_thresh, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axhline(-drift_thresh, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
            # Highlight flagged hybs
            for h in hybs[flagged.values]:
                ax.axvspan(h - 0.5, h + 0.5, color="red", alpha=0.12)
            ax.set_ylabel(label)
        axes[-1].set_xlabel("Hyb round")
        axes[-1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axes[0].set_title(f"FOV {fov_view:02d} drift  —  {int(flagged.sum())} hybs exceed {drift_thresh}px")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if flagged.any():
            st.warning(f"Flagged hybs: {[int(h) for h in hybs[flagged.values]]}")

    # ── Tab 2: CorrAlign image gallery ───────────────────────
    with tab_imgs:
        if output_dir is None:
            st.info("No output folder set.")
        else:
            corralign_dir = output_dir / "CorrAlign"
            imgs = sorted(corralign_dir.glob(f"CorrAlign_fov{fov_view:04d}_*.png")) \
                   if corralign_dir.exists() else []

            if not imgs:
                st.info("No CorrAlign images found for this FOV. Run Step 2 with 'Save figures' enabled.")
            else:
                # Filter to flagged-only if requested
                show_flagged_only = st.checkbox("Show flagged hybs only", value=False)
                if show_flagged_only:
                    flagged_hybs = set(int(h) for h in hybs[flagged.values])
                    imgs = [p for p in imgs
                            if int(p.stem.split("_h")[-1]) in flagged_hybs]

                st.caption(f"Showing {len(imgs)} images")
                cols_per_row = 2
                for i in range(0, len(imgs), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j, p in enumerate(imgs[i:i + cols_per_row]):
                        hyb_num = int(p.stem.split("_h")[-1])
                        caption = f"Hyb {hyb_num}"
                        if flagged.iloc[hyb_num - 1]:
                            caption += " ⚠ flagged"
                        row_cols[j].image(str(p), caption=caption, use_container_width=True)

    # ── Tab 3: Edit shifts ───────────────────────────────────
    with tab_edit:
        st.markdown("Edit shifts directly. Changes are applied immediately to downstream steps.")
        df_edit = st.data_editor(
            df_reg[["xshift", "yshift", "xshift2", "yshift2"]],
            use_container_width=True,
            num_rows="fixed",
            key=f"edit_fov{fov_view}",
        )
        if _nav_button("Save edits for this FOV"):
            # Merge edited columns back into full df
            df_merged = df_reg.copy()
            df_merged[["xshift", "yshift", "xshift2", "yshift2"]] = df_edit
            st.session_state.drift_overrides[fov_view] = df_merged
            if output_dir is not None:
                df_merged.to_csv(output_dir / f"fov{fov_view:03d}_regData.csv", index=False)
                st.success(f"Saved fov{fov_view:03d}_regData.csv")
            else:
                st.success("Edits saved to session (set output folder to persist to disk).")

    # ── Navigation ───────────────────────────────────────────
    st.divider()
    col_back, col_fwd, _ = st.columns([1, 2, 5], vertical_alignment="bottom")
    if col_back.button("← Back"):
        st.session_state.current_step = 2
        st.rerun()
    with col_fwd:
        if _nav_button("Proceed to Select Test Spot →"):
            st.session_state.current_step = 4
            st.rerun()


# ============================================================
# STEP 4 — Select Test Spot
# ============================================================

elif step == 4:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.header("Step 4 — Select Test Spot")
    st.markdown(
        "Load a readout channel max-projection, then **click on the image** to pick a "
        "single test locus. This spot will be used in Step 5 to verify fitting parameters."
    )

    chrtracer_dir = st.session_state.chrtracer_dir
    df_layout     = st.session_state.exp_layout
    output_dir    = st.session_state.output_dir
    num_locs      = st.session_state.get("num_locs", 1)

    if chrtracer_dir is None or df_layout is None:
        st.error("Complete Step 1 first.")
        st.stop()

    # ── Skip if test fit outputs already exist ───────────────
    if output_dir is not None:
        existing_fits = sorted(output_dir.glob("fov*_testSpot_fits.csv"))
        if existing_fits:
            st.success(f"Found {len(existing_fits)} existing test-spot fit file(s).")
            if _skip_button("⏭ Skip Steps 4 & 5 — proceed to Auto Detect All Spots"):
                # Load the first fit CSV to populate test_fit_rows
                df_fits = pd.read_csv(existing_fits[0])
                st.session_state.test_fit_rows = df_fits.to_dict("records")
                st.session_state.current_step = 6   # jump to Step 6
                st.rerun()
            st.divider()

    readout_folders = [chrtracer_dir / row["FolderName"]
                       for _, row in df_layout.iterrows()]

    # ── Controls ─────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    fov_sel = col1.selectbox("FOV", st.session_state.get("fov_list", list(range(1, num_locs + 1))),
                              format_func=lambda f: f"FOV {f:02d}")
    hyb_sel = col2.selectbox("Hyb (readout round)", list(range(1, len(readout_folders) + 1)),
                              index=0, format_func=lambda h: f"Hyb {h:02d}")
    ch_sel  = col3.radio("Channel to display", ["Readout (647)", "Fiducial (565)"],
                          index=0, horizontal=True)
    ch_idx  = 1 if ch_sel.startswith("Readout") else 0   # 0=fid, 1=readout in interleaved frames

    load_clicked = _exec_button("Load max-projection")

    cache_key = (fov_sel, hyb_sel, ch_idx)

    if load_clicked:
        dax_name = f"ConvZscan_{fov_sel:02d}.dax"
        inf_name = f"ConvZscan_{fov_sel:02d}.inf"
        folder   = readout_folders[hyb_sel - 1]
        dax_path = folder / dax_name

        if not dax_path.exists():
            st.error(f"DAX not found: {dax_path}")
            st.stop()

        with st.spinner(f"Reading {dax_path.name}…"):
            inf   = orca_drift.read_inf(folder / inf_name)
            H, W, N = inf["height"], inf["width"], inf["n_frames"]
            stack = orca_drift.read_dax(dax_path, H, W, N)
            # Extract channel and max-project
            ch_frames = stack[ch_idx::2]
            maxproj   = ch_frames.max(axis=0).astype(np.float32)
            st.session_state.maxproj_cache[cache_key] = maxproj

    # ── Image display ─────────────────────────────────────────
    maxproj = st.session_state.maxproj_cache.get(cache_key)

    if maxproj is None:
        st.info("Select FOV, hyb and channel above, then click **Load max-projection**.")
    else:
        H, W = maxproj.shape

        from streamlit_image_coordinates import streamlit_image_coordinates
        from PIL import Image as PILImage

        # Auto-contrast (0.5 – 99.5 percentile) → 8-bit PIL image
        lo, hi = np.percentile(maxproj, 0.5), np.percentile(maxproj, 99.5)
        disp = np.clip((maxproj - lo) / (hi - lo + 1e-10), 0, 1)

        # Downsample 4× for display speed; keep ds factor for coordinate mapping
        ds = 4
        disp_ds = (disp[::ds, ::ds] * 255).astype(np.uint8)

        # Retrieve existing selection
        ts    = st.session_state.test_spot or {}
        sel_x = ts.get("x")    # full-res pixel X (column)
        sel_y = ts.get("y")    # full-res pixel Y (row)

        # Draw crosshair on display image if spot already selected
        disp_rgb = np.stack([disp_ds, disp_ds, disp_ds], axis=-1)
        if sel_x is not None and sel_y is not None:
            sx_ds = sel_x // ds
            sy_ds = sel_y // ds
            disp_rgb[max(0, sy_ds-1):sy_ds+2, :] = [255, 0, 0]   # horizontal
            disp_rgb[:, max(0, sx_ds-1):sx_ds+2] = [255, 0, 0]   # vertical

        pil_img = PILImage.fromarray(disp_rgb)

        st.caption(f"FOV {fov_sel:02d}  Hyb {hyb_sel:02d}  {ch_sel} — click to select test spot")
        coords = streamlit_image_coordinates(pil_img, key="spot_picker")

        # Handle click — coords are in downsampled space, map back to full-res
        if coords is not None:
            clicked_x = int(coords["x"]) * ds
            clicked_y = int(coords["y"]) * ds
            st.session_state.test_spot = {
                "fov": fov_sel, "hyb": hyb_sel,
                "x": clicked_x, "y": clicked_y,
            }
            sel_x, sel_y = clicked_x, clicked_y

        # ── Info + zoom ───────────────────────────────────────
        if sel_x is not None and sel_y is not None:
            ts = st.session_state.test_spot
            st.success(
                f"Test spot — FOV {ts['fov']:02d}  |  "
                f"X = {ts['x']}  Y = {ts['y']}  |  "
                f"Hyb ref: {ts['hyb']:02d}"
            )

            # Zoom crop (60×60 px)
            crop = 30
            r0, r1 = max(0, sel_y - crop), min(H, sel_y + crop)
            c0, c1 = max(0, sel_x - crop), min(W, sel_x + crop)
            patch  = maxproj[r0:r1, c0:c1]

            fig_z, ax = plt.subplots(figsize=(4, 4))
            lo_p, hi_p = np.percentile(patch, 1), np.percentile(patch, 99)
            ax.imshow(patch, cmap="gray", vmin=lo_p, vmax=hi_p, origin="upper")
            ax.axhline(sel_y - r0, color="red", linewidth=0.8)
            ax.axvline(sel_x - c0, color="red", linewidth=0.8)
            ax.set_title(f"Zoom ±{crop}px around ({sel_x}, {sel_y})", fontsize=9)
            ax.axis("off")
            fig_z.tight_layout()
            st.pyplot(fig_z)
            plt.close(fig_z)

    # ── Navigation ────────────────────────────────────────────
    st.divider()
    col_back, col_fwd, _ = st.columns([1, 2, 5], vertical_alignment="bottom")
    if col_back.button("← Back"):
        st.session_state.current_step = 3
        st.rerun()
    ts = st.session_state.test_spot
    if ts is not None:
        with col_fwd:
            if _nav_button("Proceed to Fit Test Spot →"):
                st.session_state.current_step = 5
                st.rerun()
    else:
        col_fwd.caption("Select a spot to proceed.")


# ============================================================
# STEP 5 — Fit Test Spot
# ============================================================

elif step == 5:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.header("Step 5 — Fit Test Spot")
    st.markdown(
        "- **Fit status:** Each hyb is marked `ok` (DAX found, readout crop non-empty, "
        "3D Gaussian fit on 9×9×13 sub-crop completed), `missing` (DAX file not found on disk), "
        "or `no_data` (DAX exists but the readout crop is empty, e.g. spot too close to image edge). "
        "Only `ok` hybs are plotted and included in downstream analysis.\n"
        "- **Fit quality:** `fitQuality` = sum(residuals²)/h_fit² over the 3D sub-crop — "
        "directly comparable to Matlab's resRatio. Lower values indicate a better fit. "
        "No automatic threshold is applied here; use the Quality Filters in Step 7 to exclude poor fits."
    )

    ts = st.session_state.test_spot
    if ts is None:
        st.error("No test spot selected. Go back to Step 4.")
        if st.button("← Back"):
            st.session_state.current_step = 4
            st.rerun()
        st.stop()

    chrtracer_dir = st.session_state.chrtracer_dir
    df_layout     = st.session_state.exp_layout
    output_dir    = st.session_state.output_dir
    reg_data      = st.session_state.reg_data

    readout_folders = [chrtracer_dir / row["FolderName"]
                       for _, row in df_layout.iterrows()]
    fov = ts["fov"]

    # Load regData for this FOV
    df_reg = reg_data.get(fov)
    if df_reg is None and output_dir is not None:
        csv = output_dir / f"fov{fov:03d}_regData.csv"
        if csv.exists():
            df_reg = pd.read_csv(csv)
    if df_reg is None:
        st.warning(f"No regData for FOV {fov}. Run Step 2 first.")
        df_reg = pd.DataFrame([{"xshift":0,"yshift":0,"xshift2":0,"yshift2":0}]
                               * len(readout_folders))

    st.info(f"Test spot — FOV {fov:02d}  X={ts['x']}  Y={ts['y']}")

    # ── Skip if outputs already exist ───────────────────────
    if output_dir is not None:
        existing_fits = sorted(output_dir.glob("fov*_testSpot_fits.csv"))
        if existing_fits:
            st.success(f"Found {len(existing_fits)} existing test-spot fit file(s).")
            if _skip_button("⏭ Skip Step 5 — proceed to Auto Detect All Spots"):
                df_fits = pd.read_csv(existing_fits[0])
                st.session_state.test_fit_rows = df_fits.to_dict("records")
                st.session_state.current_step = 6
                st.rerun()
            st.divider()

    # ── Parameters ──────────────────────────────────────────
    st.subheader("Fitting Parameters")
    c1, c2, c3, c4, c5 = st.columns(5)
    box_width = c1.number_input("Box width (px)", min_value=10, value=30, step=2)
    nm_xy     = c2.number_input("nm / XY pixel", min_value=1, value=108, step=1)
    nm_z      = c3.number_input("nm / Z slice",  min_value=1, value=150, step=1)
    upsample  = c4.number_input("Upsample factor", min_value=1, value=4, step=1)
    ref_hyb   = c5.number_input("Reference hyb", min_value=1,
                                 max_value=len(readout_folders), value=1, step=1)

    params = {
        "box_half": int(box_width) // 2,
        "nm_xy": int(nm_xy), "nm_z": int(nm_z),
        "upsample": int(upsample), "ref_hyb": int(ref_hyb), "n_ch": 2,
        "max_fine_shift": 5.0,
    }

    col_run, col_back, _ = st.columns([1, 1, 5], vertical_alignment="bottom")
    with col_run:
        run_clicked = _exec_button("Run Fit")
    if col_back.button("← Back"):
        st.session_state.current_step = 4
        st.rerun()

    st.divider()

    # ── Live fitting ─────────────────────────────────────────
    if run_clicked:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = st.progress(0, text="Loading reference crop…")
        status_text  = st.empty()
        live_fig     = st.empty()
        live_plot    = st.empty()

        rows_so_far = []

        try:
            gen = orca_fit.fit_test_spot_stream(
                fov=fov,
                test_x=ts["x"], test_y=ts["y"],
                readout_folders=readout_folders,
                reg_data=df_reg,
                params=params,
                output_dir=output_dir,
            )

            for hyb, n_total, result, fig, rows_so_far in gen:
                progress_bar.progress(hyb / n_total,
                                       text=f"Hyb {hyb}/{n_total} — {result.get('status','')}")

                status_text.markdown(
                    f"**Hyb {hyb}** — "
                    f"x={result.get('x', float('nan')):.0f} nm  "
                    f"y={result.get('y', float('nan')):.0f} nm  "
                    f"z={result.get('z', float('nan')):.0f} nm  "
                    f"h={result.get('h', float('nan')):.0f}  "
                    f"fitQ={result.get('fitQuality', float('nan')):.2f}"
                )

                # FitSpot figure
                if fig is not None:
                    live_fig.pyplot(fig)
                    plt.close(fig)

                # Live trajectory plot
                df_so = pd.DataFrame(rows_so_far)
                ok    = df_so[df_so["status"] == "ok"]
                if len(ok) >= 2:
                    fig_t, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
                    for ax, col_name, label in zip(
                        axes,
                        ["x", "y", "z"],
                        ["X (nm)", "Y (nm)", "Z (nm)"],
                    ):
                        ax.plot(ok["hybe"], ok[col_name], "o-", markersize=3)
                        ax.set_ylabel(label, fontsize=8)
                        ax.axhline(ok[col_name].mean(), color="gray",
                                   linewidth=0.7, linestyle="--")
                    axes[-1].set_xlabel("Hyb round")
                    axes[-1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    axes[0].set_title(f"Test spot trajectory — FOV {fov:02d} "
                                      f"({len(ok)}/{n_total} hybs fit)")
                    fig_t.tight_layout()
                    live_plot.pyplot(fig_t)
                    plt.close(fig_t)

        except Exception as exc:
            st.error(f"Error: {exc}")
            raise

        st.session_state.test_fit_rows = rows_so_far
        progress_bar.progress(1.0, text="Done!")
        status_text.success(f"Fit complete — {sum(1 for r in rows_so_far if r['status']=='ok')} "
                             f"/ {len(rows_so_far)} hybs fit successfully.")

    # ── Results table ────────────────────────────────────────
    if st.session_state.test_fit_rows:
        df_fits = pd.DataFrame(st.session_state.test_fit_rows)
        with st.expander("Fit results table"):
            st.dataframe(df_fits[["hybe","x","y","z","h","wx","wy","wz",
                                   "fitQuality","status"]].round(1),
                         use_container_width=True)

    # ── Navigation ───────────────────────────────────────────
    st.divider()
    col_b, col_f, _ = st.columns([1, 2, 5], vertical_alignment="bottom")
    if col_b.button("← Back ", key="step5_back"):
        st.session_state.current_step = 4
        st.rerun()
    if st.session_state.test_fit_rows:
        with col_f:
            if _nav_button("Proceed to Auto Detect All Spots →"):
                st.session_state.current_step = 6
                st.rerun()


# ============================================================
# STEP 6 — Auto Detect All Spots
# ============================================================

elif step == 6:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    from streamlit_image_coordinates import streamlit_image_coordinates

    st.header("Step 6 — Auto Detect All Spots")

    chrtracer_dir = st.session_state.chrtracer_dir
    df_layout     = st.session_state.exp_layout
    output_dir    = st.session_state.output_dir
    num_locs      = st.session_state.get("num_locs", 1)

    if chrtracer_dir is None or df_layout is None:
        st.error("Complete Step 1 first.")
        st.stop()

    readout_folders = [chrtracer_dir / row["FolderName"]
                       for _, row in df_layout.iterrows()]

    # ── Skip if selectSpots already on disk ─────────────────
    if output_dir is not None:
        existing = sorted(output_dir.glob("fov*_selectSpots.csv"))
        if existing:
            st.success(f"Found {len(existing)} existing selectSpots file(s).")
            if _skip_button(f"⏭ Load existing spots and skip to Step 7  ({len(existing)}/{num_locs} FOVs)"):
                loaded = {}
                for csv in existing:
                    fov_n = int(csv.stem.replace("fov", "").replace("_selectSpots", ""))
                    loaded[fov_n] = pd.read_csv(csv)
                st.session_state.select_spots = loaded
                st.session_state.current_step = 7
                st.rerun()
            st.divider()

    # ── Parameters ──────────────────────────────────────────
    st.subheader("Detection Parameters")
    c1, c2, c3, c4, c5 = st.columns(5)
    thresh_pct = c1.number_input("Threshold percentile", 0.90, 0.9999,
                                  value=0.997, step=0.001, format="%.3f")
    bg_size    = c2.number_input("Background size (px)", 10, 200, value=50, step=5)
    min_dist   = c3.number_input("Min spot separation (px)", 5, 100, value=5, step=5)
    ds_det     = c4.number_input("Detection downsample", 1, 8, value=3, step=1)
    border     = c5.number_input("Border exclusion (px)", 0, 50, value=15, step=1,
                                   help="Should be ≥ crop half-width (default 15) to avoid edge artifacts")

    _fov_options = st.session_state.get("fov_list", list(range(1, num_locs + 1)))
    fov_select = st.multiselect("FOVs to process",
                                 _fov_options,
                                 default=_fov_options,
                                 format_func=lambda f: f"FOV {f:02d}")

    col_run, col_back, _ = st.columns([1, 1, 5], vertical_alignment="bottom")
    with col_run:
        run_clicked = _exec_button("Detect Spots")
    if col_back.button("← Back"):
        st.session_state.current_step = 5
        st.rerun()

    st.divider()

    if run_clicked:
        if not fov_select:
            st.warning("Select at least one FOV.")
            st.stop()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = st.progress(0, text="Starting…")
        status_text  = st.empty()
        live_img     = st.empty()

        folder_ref = readout_folders[0]

        for i, fov in enumerate(fov_select):
            status_text.text(f"FOV {fov:02d} — loading max-projection…")
            dax_name = f"ConvZscan_{fov:02d}.dax"
            inf_name = f"ConvZscan_{fov:02d}.inf"
            dax_path = folder_ref / dax_name

            if not dax_path.exists():
                st.warning(f"FOV {fov}: DAX not found, skipping.")
                continue

            inf   = orca_drift.read_inf(folder_ref / inf_name)
            H, W, N = inf["height"], inf["width"], inf["n_frames"]
            stack = orca_drift.read_dax(dax_path, H, W, N)
            # Fiducial channel = ch0 (even frames)
            maxproj = stack[0::2].max(axis=0).astype(np.float32)
            del stack

            status_text.text(f"FOV {fov:02d} — detecting spots…")
            spots_df = orca_fit.detect_spots(
                maxproj,
                threshold_pct=float(thresh_pct),
                bg_size=int(bg_size),
                min_dist=int(min_dist),
                downsample=int(ds_det),
                border=int(border),
            )
            st.session_state.select_spots[fov] = spots_df

            if output_dir is not None:
                spots_df.to_csv(output_dir / f"fov{fov:03d}_selectSpots.csv", index=False)

            # Show overlay
            rgb = orca_fit.detect_spots_overlay(maxproj, spots_df, ds=4)
            live_img.image(PILImage.fromarray(rgb),
                           caption=f"FOV {fov:02d} — {len(spots_df)} spots detected",
                           use_container_width=True)

            progress_bar.progress((i + 1) / len(fov_select),
                                   text=f"FOV {fov:02d}: {len(spots_df)} spots")

        status_text.success(f"Detection complete — {len(fov_select)} FOV(s) processed.")

    # ── Summary ──────────────────────────────────────────────
    if st.session_state.select_spots:
        spots_all = st.session_state.select_spots
        st.subheader("Detection Summary")
        summary = pd.DataFrame([
            {"FOV": fov, "Spots detected": len(df)}
            for fov, df in sorted(spots_all.items())
        ])
        st.dataframe(summary.set_index("FOV"), use_container_width=True)
        st.metric("Total spots", summary["Spots detected"].sum())

        st.divider()
        if _nav_button("Proceed to Validate Spots →"):
            st.session_state.current_step = 7
            st.rerun()


# ============================================================
# STEP 7 — Validate Picked Spots
# ============================================================

elif step == 7:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    from streamlit_image_coordinates import streamlit_image_coordinates

    st.header("Step 7 — Validate Picked Spots")
    st.markdown(
        "Review detected spots per FOV. **Click near a spot** to remove it, "
        "or use the table editor to delete rows directly."
    )

    chrtracer_dir = st.session_state.chrtracer_dir
    df_layout     = st.session_state.exp_layout
    output_dir    = st.session_state.output_dir
    num_locs      = st.session_state.get("num_locs", 1)
    select_spots  = st.session_state.select_spots

    if not select_spots:
        st.warning("No spots detected yet. Run Step 6 first.")
        if st.button("← Back"):
            st.session_state.current_step = 6
            st.rerun()
        st.stop()

    readout_folders = [chrtracer_dir / row["FolderName"]
                       for _, row in df_layout.iterrows()]

    _fov_list = sorted(select_spots.keys())
    if "validate_fov_idx" not in st.session_state:
        st.session_state.validate_fov_idx = 0
    st.session_state.validate_fov_idx = min(
        st.session_state.validate_fov_idx, len(_fov_list) - 1
    )
    # Drive the selectbox via its key so programmatic changes (Next FOV) take effect
    st.session_state.validate_fov_select = _fov_list[st.session_state.validate_fov_idx]
    fov_view = st.selectbox(
        "Select FOV to validate",
        _fov_list,
        format_func=lambda f: f"FOV {f:02d}",
        key="validate_fov_select",
    )
    st.session_state.validate_fov_idx = _fov_list.index(fov_view)

    spots_df = select_spots[fov_view].copy().reset_index(drop=True)

    # Load / cache fiducial max-projection for display (hyb 1, ch 0)
    cache_key = (fov_view, 1, 0)
    if cache_key not in st.session_state.maxproj_cache:
        folder_ref = readout_folders[0]
        dax_path   = folder_ref / f"ConvZscan_{fov_view:02d}.dax"
        if dax_path.exists():
            with st.spinner("Loading image…"):
                inf   = orca_drift.read_inf(folder_ref / f"ConvZscan_{fov_view:02d}.inf")
                H, W, N = inf["height"], inf["width"], inf["n_frames"]
                stack = orca_drift.read_dax(dax_path, H, W, N)
                mp    = stack[0::2].max(axis=0).astype(np.float32)
                del stack
                st.session_state.maxproj_cache[cache_key] = mp

    maxproj = st.session_state.maxproj_cache.get(cache_key)
    H, W    = (maxproj.shape if maxproj is not None else (1843, 1843))
    ds      = 2   # smaller downsample → larger natural image → no display-scaling coordinate errors

    # ── Image with spots overlay ─────────────────────────────
    st.subheader(f"FOV {fov_view:02d} — {len(spots_df)} spots")

    if maxproj is not None:
        rgb = orca_fit.detect_spots_overlay(maxproj, spots_df, ds=ds)
        pil = PILImage.fromarray(rgb)

        click_radius = st.slider("Click radius (full-res px)", 5, 200, 15, key="click_radius")
        st.caption(
            "**Click near an existing spot** to remove it.  "
            "**Click on empty space** to add a new spot.  "
            f"(radius = {click_radius} full-res px ≈ {click_radius * 0.108:.0f} µm)"
        )
        coords = streamlit_image_coordinates(pil, key=f"validate_fov{fov_view}")

        if coords is not None:
            cx_full = int(coords["x"]) * ds
            cy_full = int(coords["y"]) * ds
            click_id = (fov_view, cx_full, cy_full)

            if click_id != st.session_state.last_spot_click:
                st.session_state.last_spot_click = click_id
                radius  = click_radius

                removed = False
                if len(spots_df) > 0:
                    dists   = np.sqrt((spots_df["locusX"] - cx_full) ** 2 +
                                       (spots_df["locusY"] - cy_full) ** 2)
                    closest = dists.idxmin()
                    if dists[closest] <= radius:
                        # Remove closest spot
                        spots_df = spots_df.drop(index=closest).reset_index(drop=True)
                        st.session_state.select_spots[fov_view] = spots_df
                        removed = True

                if not removed:
                    # No nearby spot found → add new spot
                    new_row  = pd.DataFrame({"locusX": [cx_full], "locusY": [cy_full]})
                    spots_df = pd.concat([spots_df, new_row], ignore_index=True)
                    st.session_state.select_spots[fov_view] = spots_df

                st.rerun()

    # ── Table editor ─────────────────────────────────────────
    with st.expander("Edit spots table (delete rows to remove spots)"):
        edited = st.data_editor(spots_df, use_container_width=True,
                                num_rows="dynamic", key=f"table_fov{fov_view}")
        if _nav_button("Apply table edits", key=f"apply_fov{fov_view}"):
            st.session_state.select_spots[fov_view] = edited.reset_index(drop=True)
            if output_dir is not None:
                edited.reset_index(drop=True).to_csv(
                    output_dir / f"fov{fov_view:03d}_selectSpots.csv", index=False)
                st.success("Saved.")
            st.rerun()

    # ── Save validated spots ─────────────────────────────────
    col_save, col_next, col_back, col_fwd, _ = st.columns([1, 1, 1, 2, 2], vertical_alignment="bottom")
    with col_save:
        if _nav_button("Save this FOV"):
            if output_dir is not None:
                spots_df.to_csv(output_dir / f"fov{fov_view:03d}_selectSpots.csv", index=False)
                st.success(f"Saved {len(spots_df)} spots for FOV {fov_view:02d}.")
    with col_next:
        _next_disabled = st.session_state.validate_fov_idx >= len(_fov_list) - 1
        if _nav_button("Next FOV →", disabled=_next_disabled):
            st.session_state.validate_fov_idx += 1
            st.rerun()
    if col_back.button("← Back"):
        st.session_state.current_step = 6
        st.rerun()
    with col_fwd:
        if _nav_button("Proceed to Fit All →"):
            st.session_state.current_step = 8
            st.rerun()


# ============================================================
# STEP 8 — Fit All
# ============================================================

elif step == 8:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.header("Step 8 — Fit All")

    chrtracer_dir = st.session_state.chrtracer_dir
    df_layout     = st.session_state.exp_layout
    output_dir    = st.session_state.output_dir
    select_spots  = st.session_state.select_spots
    reg_data_all  = st.session_state.reg_data

    if chrtracer_dir is None or df_layout is None:
        st.error("Complete Step 1 first.")
        st.stop()

    if not select_spots:
        st.error("No validated spots found. Complete Steps 6–7 first.")
        st.stop()

    readout_folders = [chrtracer_dir / row["FolderName"]
                       for _, row in df_layout.iterrows()]
    fov_list = sorted(select_spots.keys())
    n_spots_total = sum(len(select_spots[f]) for f in fov_list)

    st.markdown(
        f"Ready to fit **{n_spots_total} spots** across **{len(fov_list)} FOV(s)** "
        f"× **{len(readout_folders)} hybs** = "
        f"**{n_spots_total * len(readout_folders)} fits**."
    )

    _ALL_DEFAULTS = {
        # Fitting parameters
        "fit_all_box":      30,
        "fit_all_nmxy":     108,
        "fit_all_nmz":      150,
        "fit_all_up":       4,
        "fit_all_ref":      1,
        "fit_all_workers":  min(4, os.cpu_count() or 1),
        # Quality filters
        "fit_all_minh":     200,
        "fit_all_maxwx":    3,    # Matlab maxSigma=2 × √2 ≈ 2.83 px (standard Python σ)
        "fit_all_maxwz":    9,    # Matlab maxSigmaZ × √2 (calibrated)
        "fit_all_hbratio":  1.2,
        "fit_all_ahratio":  0.25,
        "fit_all_xystep":   12,
        "fit_all_zstep":    8,
        "fit_all_fineshft": 5,
    }
    _QF_DEFAULTS = _ALL_DEFAULTS  # alias used by reset button
    for _k, _v in _ALL_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Skip if allFits already on disk ──────────────────────
    _do_reset = False
    if output_dir is not None:
        merged_path = output_dir / "allFits.csv"
        if merged_path.exists():
            st.success("Found existing allFits.csv.")
            _sk_col, _rst_col, _ = st.columns([1, 1, 5], vertical_alignment="bottom")
            with _sk_col:
                _do_load = _skip_button("⏭ Load existing results")
            with _rst_col:
                _do_reset = st.button("↺ Reset all to defaults", key="fit_all_qf_reset")
            if _do_load:
                merged = pd.read_csv(merged_path)
                loaded = {fov_n: grp.reset_index(drop=True)
                          for fov_n, grp in merged.groupby("fov")}
                loaded["_merged"] = merged
                st.session_state.all_fits = loaded
                st.rerun()
        else:
            _do_reset = st.button("↺ Reset all to defaults", key="fit_all_qf_reset")
        if _do_reset:
            for k, v in _QF_DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()
        st.divider()

    # ── Parameters ───────────────────────────────────────────
    st.subheader("Fitting Parameters")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    box_width = c1.number_input("Box width (px)", min_value=10, step=2,
                                 key="fit_all_box")
    nm_xy     = c2.number_input("nm / XY pixel",  min_value=1, step=1,
                                 key="fit_all_nmxy")
    nm_z      = c3.number_input("nm / Z slice",   min_value=1, step=1,
                                 key="fit_all_nmz")
    upsample  = c4.number_input("Upsample factor", min_value=1, step=1,
                                 key="fit_all_up")
    ref_hyb   = c5.number_input("Reference hyb", min_value=1,
                                 max_value=len(readout_folders), step=1,
                                 key="fit_all_ref")
    max_cores = os.cpu_count() or 1
    n_workers = c6.number_input(f"CPU cores (max {max_cores})",
                                 min_value=1, max_value=max_cores, step=1,
                                 key="fit_all_workers")

    st.subheader("Quality Filters")
    st.caption(
        "Readouts failing any filter are excluded (status ≠ ok). Defaults match ChrTracer3 Matlab reference. "
        "**Note on XY width:** Fits a full **3D Gaussian** on a 9×9×13 sub-crop (matching Matlab FitPsf3D). "
        "Python uses the standard convention exp(−½(r/σ)²), while Matlab uses exp(−(r/2σ)²), so "
        "σ_python = √2 × σ_matlab for the same PSF. Matlab's `maxSigma = 2` → **2.83 px** fitting bound; "
        "QC rejects fits with width > **2.0 px** (Matlab convention). "
        "**fitQuality** = sum(residuals²)/h_fit² over the 3D sub-crop — directly comparable to Matlab's resRatio = sum(3D residuals²)/a²."
    )
    q1, q2, q3, q4 = st.columns(4)
    min_h        = q1.number_input("Min amplitude (h)", min_value=0, step=10,
                                    key="fit_all_minh",
                                    help="datMinPeakHeight: minimum raw peak intensity")
    max_wx_px    = q2.number_input("Max XY width (px)", min_value=1, step=1,
                                    key="fit_all_maxwx",
                                    help="datMaxFitWidth / maxSigma: max Gaussian width in XY (Matlab convention). "
                                         "Matlab's maxSigma=2.0. Fitting bound = 2.0×√2 ≈ 2.83 px (Python σ convention). "
                                         "QC threshold = 2.0 px (Matlab convention, after dividing fitted σ by √2).")
    max_wz_sl    = q3.number_input("Max Z depth (slices)", min_value=1, step=1,
                                    key="fit_all_maxwz",
                                    help="datMaxFitZdepth: max Gaussian sigma in Z (slices)")
    min_hb_ratio = q4.number_input("Min h/bg ratio", min_value=0.0, step=0.1,
                                    format="%.1f", key="fit_all_hbratio",
                                    help="datMinHBratio: min raw peak / background ratio")
    q5, q6, q7, q8 = st.columns(4)
    min_ah_ratio = q5.number_input("Min amp/peak ratio", min_value=0.0, step=0.05,
                                    format="%.2f", key="fit_all_ahratio",
                                    help="datMinAHratio: min fitted amplitude / raw peak ratio")
    max_xy_step  = q6.number_input("Max XY step (px)", min_value=0, step=1,
                                    key="fit_all_xystep",
                                    help="maxXYstep: max Gaussian fit offset from crop centre (pixels)")
    max_z_step   = q7.number_input("Max Z step (slices)", min_value=0, step=1,
                                    key="fit_all_zstep",
                                    help="maxZstep: (reserved) max Z step filter")
    max_fine_shift = q8.number_input("Max fine shift (px)", min_value=0, step=1,
                                    key="fit_all_fineshft",
                                    help="Cap per-spot fine alignment shift; larger shifts are likely spurious phase-correlation noise")

    params = {
        "box_half": int(box_width) // 2,
        "nm_xy": int(nm_xy), "nm_z": int(nm_z),
        "upsample": int(upsample), "ref_hyb": int(ref_hyb), "n_ch": 2,
        # quality filters
        "min_h":           float(min_h),
        "max_wx_px":       float(max_wx_px),
        "max_wz_sl":       float(max_wz_sl),
        "min_hb_ratio":    float(min_hb_ratio),
        "min_ah_ratio":    float(min_ah_ratio),
        "max_xy_step":     float(max_xy_step),
        "max_z_step":      float(max_z_step),
        "max_fine_shift":  float(max_fine_shift),
    }

    # ── Confirmation warning ──────────────────────────────────
    if not st.session_state.fit_running:
        st.warning(
            "⚠ **Fitting can take a significant amount of time.** "
            "Please confirm all parameters above are correct before proceeding. "
            "Once started, the process runs until all FOVs are complete."
        )
        confirmed = st.checkbox("I have verified all parameters and want to run Fit All",
                                key="fitall_confirmed")
    else:
        confirmed = True  # already running, no gate needed

    # ── Run / Stop buttons ────────────────────────────────────
    col_run, col_stop, _ = st.columns([1, 1, 5], vertical_alignment="bottom")
    with col_run:
        run_clicked = _exec_button("Run Fit All",
                                   disabled=st.session_state.fit_running or not confirmed)
    with col_stop:
        stop_clicked = st.button("⏹ Stop",
                                 disabled=not st.session_state.fit_running,
                                 key="fitall_stop")

    # Handle Stop request
    if stop_clicked:
        st.session_state.fit_stop_requested = True

    # Handle Run: initialise per-FOV state and snapshot params
    if run_clicked and not st.session_state.fit_running:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.fit_running         = True
        st.session_state.fit_fov_idx         = 0
        st.session_state.fit_stop_requested  = False
        st.session_state.fit_all_rows        = []
        st.session_state.fit_by_fov          = {}
        st.session_state.fit_params_snapshot = {
            "params":          params,
            "n_workers":       int(n_workers),
            "fov_list":        fov_list,
            "readout_folders": readout_folders,
        }
        st.rerun()

    st.divider()

    # ── Per-FOV fitting (one FOV per Streamlit rerun) ─────────
    if st.session_state.fit_running:
        snap        = st.session_state.fit_params_snapshot
        _fov_list   = snap["fov_list"]
        _rf         = snap["readout_folders"]
        _params     = snap["params"]
        _n_workers  = snap["n_workers"]
        _fov_i      = st.session_state.fit_fov_idx
        _n_fovs     = len(_fov_list)
        _parallel   = _n_workers > 1

        # ── Progress display ──────────────────────────────────
        done_so_far = _fov_i  # FOVs completed
        frac_done   = done_so_far / _n_fovs if _n_fovs else 0
        parallel_note = f" ({_n_workers} cores)" if _parallel else ""

        if st.session_state.fit_stop_requested:
            st.warning("⏹ Stop requested — finalising after current FOV…")
        else:
            st.warning(
                f"⚠ Fitting in progress{parallel_note} — **do not refresh or navigate away** "
                "until complete."
            )

        _is_finalising = st.session_state.fit_stop_requested or _fov_i >= _n_fovs
        _next_fov_str  = f"{_fov_list[_fov_i]:02d}" if _fov_i < _n_fovs else "—"
        progress_bar = st.progress(
            frac_done,
            text=(f"Stopped after {done_so_far}/{_n_fovs} FOVs."
                  if _is_finalising
                  else f"FOV {done_so_far}/{_n_fovs} complete — fitting FOV {_next_fov_str}…")
        )
        status_text = st.empty()

        # Show trajectory of the last completed FOV (sequential mode only)
        live_plot = st.empty()
        if not _parallel and st.session_state.fit_by_fov:
            _last_fov_done = _fov_list[_fov_i - 1] if _fov_i > 0 else None
            if _last_fov_done is not None and _last_fov_done in st.session_state.fit_by_fov:
                df_prev = st.session_state.fit_by_fov[_last_fov_done]
                last_spot = df_prev["spot_id"].max()
                ok_prev = df_prev[(df_prev["spot_id"] == last_spot) & (df_prev["status"] == "ok")]
                if len(ok_prev) >= 2:
                    fig_t, axes = plt.subplots(3, 1, figsize=(10, 4), sharex=True)
                    for ax, col_name, label in zip(axes, ["x", "y", "z"],
                                                    ["X (nm)", "Y (nm)", "Z (nm)"]):
                        ax.plot(ok_prev["hybe"], ok_prev[col_name], "o-", markersize=3)
                        ax.set_ylabel(label, fontsize=8)
                        ax.axhline(ok_prev[col_name].mean(), color="gray",
                                   linewidth=0.7, linestyle="--")
                    axes[-1].set_xlabel("Hyb round")
                    axes[-1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    axes[0].set_title(
                        f"FOV {_last_fov_done:02d} — last spot trajectory "
                        f"({len(ok_prev)}/{len(_rf)} hybs fit)"
                    )
                    fig_t.tight_layout()
                    live_plot.pyplot(fig_t)
                    plt.close(fig_t)

        # ── Check if we're done or stopped ───────────────────
        if _is_finalising:
            # Finalise
            all_rows = st.session_state.fit_all_rows
            fits_by_fov = st.session_state.fit_by_fov
            st.session_state.all_fits = fits_by_fov

            if all_rows and output_dir is not None:
                merged = (pd.DataFrame(all_rows)
                            .sort_values(["fov", "spot_id", "hybe"])
                            .reset_index(drop=True))
                merged.to_csv(output_dir / "allFits.csv", index=False)
                st.session_state.all_fits["_merged"] = merged

            n_ok = sum(1 for r in all_rows if r["status"] == "ok")
            n_fovs_done = len(fits_by_fov)

            if st.session_state.fit_stop_requested:
                progress_bar.progress(
                    frac_done,
                    text=f"Stopped — {n_fovs_done}/{_n_fovs} FOVs completed."
                )
                status_text.warning(
                    f"Fitting stopped by user — {n_ok} / {len(all_rows)} fits successful "
                    f"across {n_fovs_done}/{_n_fovs} FOV(s)."
                )
            else:
                progress_bar.progress(1.0, text="Done!")
                status_text.success(
                    f"Fit complete — {n_ok} / {len(all_rows)} fits successful "
                    f"across {n_fovs_done} FOV(s)."
                )

            # Reset running state
            st.session_state.fit_running        = False
            st.session_state.fit_stop_requested = False

        else:
            # ── Process one FOV ───────────────────────────────
            fov = _fov_list[_fov_i]
            spots_df    = select_spots[fov]
            df_reg      = reg_data_all.get(fov, pd.DataFrame())
            n_spots_fov = len(spots_df)

            if _parallel:
                status_text.markdown(
                    f"**FOV {fov:02d}** ({_fov_i+1}/{_n_fovs}) — "
                    f"fitting {n_spots_fov} spots × {len(_rf)} hybs in parallel…"
                )

            gen = orca_fit.fit_all_spots_stream(
                fov=fov,
                spots_df=spots_df,
                readout_folders=_rf,
                reg_data=df_reg,
                params=_params,
                output_dir=output_dir,
                n_workers=_n_workers,
            )

            fov_rows = []
            try:
                for spot_idx, n_spots, hyb, n_hybs, result, rows_so_far in gen:
                    fov_rows = rows_so_far
                    if not _parallel:
                        frac = (_fov_i + (spot_idx * n_hybs + hyb) / (n_spots * n_hybs)) / _n_fovs
                        progress_bar.progress(
                            frac,
                            text=f"FOV {fov:02d} ({_fov_i+1}/{_n_fovs}) · "
                                 f"spot {spot_idx+1}/{n_spots} · hyb {hyb}/{n_hybs}"
                        )
                        status_text.markdown(
                            f"**FOV {fov:02d}** · spot {spot_idx+1}/{n_spots} · hyb {hyb}/{n_hybs}  —  "
                            f"x={result.get('x', float('nan')):.0f} nm  "
                            f"y={result.get('y', float('nan')):.0f} nm  "
                            f"z={result.get('z', float('nan')):.0f} nm  "
                            f"fitQ={result.get('fitQuality', float('nan')):.2f}"
                        )
            except Exception as exc:
                st.error(f"Error on FOV {fov}: {exc}")
                st.session_state.fit_running = False
                st.stop()

            # Accumulate results
            st.session_state.fit_all_rows.extend(fov_rows)
            if fov_rows:
                st.session_state.fit_by_fov[fov] = pd.DataFrame(fov_rows)

            n_ok_fov = sum(1 for r in fov_rows if r["status"] == "ok")
            status_text.markdown(
                f"**FOV {fov:02d}** complete — {n_ok_fov} / {len(fov_rows)} fits successful."
            )

            # Advance to next FOV and rerun
            st.session_state.fit_fov_idx = _fov_i + 1
            st.rerun()

    # ── Results summary ───────────────────────────────────────
    merged_df = st.session_state.all_fits.get("_merged")
    if merged_df is None and st.session_state.all_fits:
        non_meta = {k: v for k, v in st.session_state.all_fits.items()
                    if k != "_merged"}
        if non_meta:
            merged_df = pd.concat(non_meta.values(), ignore_index=True)

    if merged_df is not None and len(merged_df) > 0:
        ok_df = merged_df[merged_df["status"] == "ok"]
        st.subheader("Results Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total fits",    len(merged_df))
        m2.metric("Successful",    len(ok_df))
        m3.metric("Spots",         merged_df.groupby(["fov", "spot_id"]).ngroups)
        m4.metric("FOVs",          merged_df["fov"].nunique())

        status_counts = merged_df["status"].value_counts()
        if len(status_counts) > 1:
            with st.expander("Filter breakdown"):
                st.dataframe(status_counts.rename("count").reset_index().rename(
                    columns={"index": "status"}), hide_index=True)

        with st.expander("Browse fits table"):
            display_cols = ["fov", "spot_id", "locus_x", "locus_y",
                            "hybe", "x", "y", "z", "h", "wx", "wy", "wz",
                            "fitQuality", "status"]
            st.dataframe(merged_df[display_cols].round(1),
                         use_container_width=True)

        # Download button for merged CSV
        if output_dir is not None:
            merged_path = output_dir / "allFits.csv"
            if merged_path.exists():
                with open(merged_path, "rb") as f:
                    st.download_button(
                        "⬇ Download allFits.csv",
                        data=f,
                        file_name="allFits.csv",
                        mime="text/csv",
                    )

    # ── Navigation ────────────────────────────────────────────
    st.divider()
    col_back, col_fwd, _ = st.columns([1, 2, 5], vertical_alignment="bottom")
    if col_back.button("← Back", key="fitall_back2"):
        st.session_state.current_step = 7
        st.rerun()
    if st.session_state.all_fits:
        with col_fwd:
            if _nav_button("Proceed to Export to OLIVE →"):
                st.session_state.current_step = 9
                st.rerun()


# ============================================================
# STEP 9 — Export to OLIVE
# ============================================================

elif step == 9:
    st.header("Step 9 — Export to OLIVE")
    st.markdown(
        "- Convert fit results to the OLIVE format (`x, y, z, readout, s, fov`) "
        "for direct import into the [OLIVE chromatin trace visualizer](https://faryabilab.github.io/chromatin-traces-vis/).\n"
        "- Only successful fits (`status = ok`) are included."
    )

    output_dir = st.session_state.output_dir
    all_fits   = st.session_state.all_fits

    # Try loading from disk if session state is empty
    merged_df = all_fits.get("_merged")
    if merged_df is None and output_dir is not None:
        merged_path = output_dir / "allFits.csv"
        if merged_path.exists():
            merged_df = pd.read_csv(merged_path)
            st.session_state.all_fits["_merged"] = merged_df

    if merged_df is None or len(merged_df) == 0:
        st.warning("No fit results found. Complete Step 8 first.")
        if st.button("← Back"):
            st.session_state.current_step = 8
            st.rerun()
        st.stop()

    ok_df = merged_df[merged_df["status"] == "ok"].copy()
    all_hybs = sorted(ok_df["hybe"].unique())

    # ── Readout filter ────────────────────────────────────────
    st.subheader("Readout Selection")
    filter_mode = st.radio(
        "Which readout steps to include?",
        ["All", "Odd only", "Even only", "Custom"],
        horizontal=True,
    )

    if filter_mode == "Odd only":
        selected_hybs = [h for h in all_hybs if h % 2 == 1]
    elif filter_mode == "Even only":
        selected_hybs = [h for h in all_hybs if h % 2 == 0]
    elif filter_mode == "Custom":
        selected_hybs = st.multiselect(
            "Select readout steps to include",
            options=all_hybs,
            default=all_hybs,
            format_func=lambda h: f"Hyb {h}",
        )
    else:
        selected_hybs = all_hybs

    filtered_df = ok_df[ok_df["hybe"].isin(selected_hybs)].copy()

    # Renumber readouts 1..N in selection order so OLIVE gets a contiguous sequence
    hyb_to_readout = {h: i + 1 for i, h in enumerate(sorted(selected_hybs))}
    filtered_df["readout_olive"] = filtered_df["hybe"].map(hyb_to_readout)

    # ── Summary ──────────────────────────────────────────────
    n_fovs     = filtered_df["fov"].nunique()
    n_spots    = filtered_df.groupby(["fov", "spot_id"]).ngroups
    n_readouts = len(selected_hybs)
    total_ok   = len(filtered_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FOVs",           n_fovs)
    m2.metric("Spots (traces)", n_spots)
    m3.metric("Readouts",       n_readouts)
    m4.metric("Fit points",     total_ok)

    if not selected_hybs:
        st.warning("No readouts selected.")
        st.stop()

    # ── Build OLIVE CSV ───────────────────────────────────────
    olive_df = (
        filtered_df[["fov", "spot_id", "readout_olive", "x", "y", "z"]]
        .rename(columns={"spot_id": "s", "readout_olive": "readout"})
        [["x", "y", "z", "readout", "s", "fov"]]
        .reset_index(drop=True)
    )

    st.subheader("Preview")
    st.dataframe(olive_df.head(20), use_container_width=True)

    st.info(
        f"When importing into OLIVE, set **Total Readouts = {n_readouts}**."
    )

    # ── Save & download ───────────────────────────────────────
    st.subheader("Save & Download")
    # Save selection for Step 10
    st.session_state.olive_keep_hybs = selected_hybs

    suffix    = {"All": "all", "Odd only": "odd",
                 "Even only": "even", "Custom": "custom"}[filter_mode]
    filename  = f"allFits_OLIVE_{suffix}.csv"
    olive_csv = olive_df.to_csv(index=False)

    col_save, col_dl, _ = st.columns([1, 1, 5])

    if output_dir is not None:
        olive_path = output_dir / filename
        with col_save:
            if _nav_button("Save to output folder"):
                olive_path.write_text(olive_csv)
                st.success(f"Saved to `{olive_path}`")

    with col_dl:
        st.download_button(
            f"⬇ Download {filename}",
            data=olive_csv,
            file_name=filename,
            mime="text/csv",
        )

    st.divider()
    col_back, col_fwd, _ = st.columns([1, 2, 5], vertical_alignment="bottom")
    if col_back.button("← Back", key="olive_back"):
        st.session_state.current_step = 8
        st.rerun()
    with col_fwd:
        if _nav_button("Proceed to Combine, Impute & QC →"):
            st.session_state.current_step = 10
            st.rerun()


# ============================================================
# STEP 10 — Combine, Impute & QC
# ============================================================

elif step == 10:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    st.header("Step 10 — Combine, Impute & QC")
    st.markdown(
        "- Annotate each readout step as **keep** or **exclude** based on Step 9 selection, "
        "linearly impute missing keep-steps per trace, and save `_imputed.csv`.\n"
        "- Generate QC plots: per-step count, per-step intensity, and detected-steps-per-trace distribution."
    )

    output_dir = st.session_state.output_dir
    all_fits   = st.session_state.all_fits

    # ── Load combined allFits ─────────────────────────────────
    merged_df = all_fits.get("_merged")
    if merged_df is None and output_dir is not None:
        p = output_dir / "allFits.csv"
        if p.exists():
            merged_df = pd.read_csv(p)
            st.session_state.all_fits["_merged"] = merged_df

    if merged_df is None or len(merged_df) == 0:
        st.warning("No fit results found. Complete Step 8 first.")
        if st.button("← Back"):
            st.session_state.current_step = 9
            st.rerun()
        st.stop()

    all_hybs    = sorted(merged_df["hybe"].unique())
    n_all       = len(all_hybs)

    # ── Keep/exclude selection (inherit from Step 9, allow override) ──
    st.subheader("Keep / Exclude Annotation")

    prev = st.session_state.olive_keep_hybs or all_hybs
    mode_init = (
        "Odd only"  if prev == [h for h in all_hybs if h % 2 == 1] else
        "Even only" if prev == [h for h in all_hybs if h % 2 == 0] else
        "All"       if set(prev) == set(all_hybs) else "Custom"
    )
    filter_mode = st.radio(
        "Keep steps",
        ["All", "Odd only", "Even only", "Custom"],
        index=["All", "Odd only", "Even only", "Custom"].index(mode_init),
        horizontal=True,
        key="step10_filter",
    )
    if filter_mode == "Odd only":
        keep_hybs = [h for h in all_hybs if h % 2 == 1]
    elif filter_mode == "Even only":
        keep_hybs = [h for h in all_hybs if h % 2 == 0]
    elif filter_mode == "Custom":
        keep_hybs = st.multiselect(
            "Keep steps",
            options=all_hybs,
            default=prev,
            format_func=lambda h: f"Hyb {h}",
            key="step10_custom",
        )
    else:
        keep_hybs = all_hybs

    exclude_hybs = [h for h in all_hybs if h not in keep_hybs]
    st.caption(f"Keep: **{len(keep_hybs)}** steps · Exclude: **{len(exclude_hybs)}** steps")

    # ── Annotate ─────────────────────────────────────────────
    merged_df = merged_df.copy()
    merged_df["step_type"] = merged_df["hybe"].apply(
        lambda h: "keep" if h in keep_hybs else "exclude"
    )
    merged_df["trace_id"] = (
        "fov_" + merged_df["fov"].astype(str) + "_s_" + merged_df["spot_id"].astype(str)
    )

    # ── Impute ────────────────────────────────────────────────
    st.subheader("Linear Imputation")
    st.markdown(
        "For each trace, interpolate x/y/z linearly at keep-steps where the fit is missing or failed."
    )

    run_impute = _exec_button("Run Imputation")

    if run_impute:
        ok_keep = merged_df[
            (merged_df["step_type"] == "keep") & (merged_df["status"] == "ok")
        ].copy()

        imputed_rows = []
        traces = ok_keep["trace_id"].unique()
        prog   = st.progress(0, text="Imputing…")

        for i, tid in enumerate(traces):
            tr = ok_keep[ok_keep["trace_id"] == tid].set_index("hybe")
            fov_val   = tr["fov"].iloc[0]
            spot_val  = tr["spot_id"].iloc[0]

            # Reindex to full keep_hybs range; track which were originally present
            orig_hybs = set(tr.index)
            tr = tr.reindex(keep_hybs)
            for col in ["x", "y", "z"]:
                tr[col] = tr[col].interpolate(method="index", limit_direction="both")

            for h in keep_hybs:
                imputed_rows.append({
                    "fov": fov_val, "spot_id": spot_val, "hybe": h,
                    "x": tr.loc[h, "x"], "y": tr.loc[h, "y"], "z": tr.loc[h, "z"],
                    "status": "ok" if h in orig_hybs else "imputed",
                })
            prog.progress((i + 1) / len(traces))

        imputed_df = pd.DataFrame(imputed_rows)

        # OLIVE column order
        olive_imp = (
            imputed_df[["fov", "spot_id", "hybe", "x", "y", "z"]]
            .rename(columns={"spot_id": "s", "hybe": "readout"})
            [["x", "y", "z", "readout", "s", "fov"]]
        )
        # renumber readouts 1..N
        h2r = {h: i+1 for i, h in enumerate(keep_hybs)}
        olive_imp["readout"] = olive_imp["readout"].map(h2r)

        imputed_csv = olive_imp.to_csv(index=False)
        if output_dir is not None:
            imp_path = output_dir / "allFits_imputed.csv"
            imp_path.write_text(imputed_csv)
            st.success(f"Saved `{imp_path.name}` — {len(imputed_df)} rows, "
                       f"{imputed_df['spot_id'].nunique()} traces × {len(keep_hybs)} steps")

        st.download_button(
            "⬇ Download allFits_imputed.csv",
            data=imputed_csv,
            file_name="allFits_imputed.csv",
            mime="text/csv",
        )
        prog.progress(1.0, text="Done!")

    # ── QC Plots ─────────────────────────────────────────────
    st.subheader("QC Plots")

    ok_all = merged_df[merged_df["status"] == "ok"]

    # Build walk colours: keep steps get a spectral ramp, exclude steps are lightgrey
    import colorsys
    def _spectral(i, n):
        h = i / max(n - 1, 1)
        r, g, b = colorsys.hsv_to_rgb(h * 0.75, 0.85, 0.85)
        return (r, g, b)

    keep_list   = sorted(keep_hybs)
    bar_colors  = []
    ki = 0
    for h in all_hybs:
        if h in keep_hybs:
            bar_colors.append(_spectral(ki, len(keep_list)))
            ki += 1
        else:
            bar_colors.append((0.82, 0.82, 0.82))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Plot 1 — count per readout step
    ax = axes[0]
    counts = [len(ok_all[ok_all["hybe"] == h]) for h in all_hybs]
    ax.bar(all_hybs, counts, color=bar_colors, width=0.8)
    ax.set_xlabel("Readout #")
    ax.set_ylabel("Count")
    ax.set_title("Detected spots per readout")
    keep_patch   = mpatches.Patch(color=_spectral(0, 2), label="Keep")
    excl_patch   = mpatches.Patch(color=(0.82, 0.82, 0.82), label="Exclude")
    ax.legend(handles=[keep_patch, excl_patch], fontsize=8)

    # Plot 2 — intensity (h) per readout step
    ax = axes[1]
    for j, h in enumerate(all_hybs):
        vals = ok_all[ok_all["hybe"] == h]["h"].dropna().values
        if len(vals) == 0:
            continue
        bp = ax.boxplot(
            vals, positions=[h], widths=0.7,
            patch_artist=True,
            boxprops=dict(facecolor=bar_colors[j], linewidth=0.5),
            medianprops=dict(color="black", linewidth=1),
            whiskerprops=dict(linewidth=0.5),
            capprops=dict(linewidth=0.5),
            flierprops=dict(marker=".", markersize=1, alpha=0.3),
            showfliers=False,
        )
    ax.set_xlabel("Readout #")
    ax.set_ylabel("Intensity (h)")
    ax.set_title("Spot intensity per readout")
    ax.set_xlim(min(all_hybs) - 0.5, max(all_hybs) + 0.5)

    # Plot 3 — detected keep-steps per trace distribution
    ax = axes[2]
    keep_ok = merged_df[(merged_df["step_type"] == "keep") & (merged_df["status"] == "ok")]
    bridge_counts = keep_ok.groupby("trace_id").size()
    total_traces  = merged_df["trace_id"].nunique()
    n_bins = min(len(keep_hybs), 30)
    blues  = [colorsys.hsv_to_rgb(0.6, s, 0.85) for s in np.linspace(0.1, 0.9, n_bins)]
    ax.hist(
        bridge_counts.values,
        bins=np.arange(0.5, len(keep_hybs) + 1.5),
        weights=np.ones(len(bridge_counts)) / total_traces,
        color="steelblue", edgecolor="white", linewidth=0.3,
    )
    ax.set_xlabel(f"Detected keep-steps per trace (max {len(keep_hybs)})")
    ax.set_ylabel("Fraction of traces")
    ax.set_title("Step detection per trace")

    fig.tight_layout()
    if output_dir is not None:
        qc_path = output_dir / "QC_plots.png"
        fig.savefig(qc_path, dpi=150, bbox_inches="tight")
        st.caption(f"Saved `{qc_path.name}`")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    if st.button("← Back", key="step10_back"):
        st.session_state.current_step = 9
        st.rerun()
