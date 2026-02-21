# app.py
# ==========================================================
# ATLAST Rotation Dashboard (Master-powered + Raw fold caveat)
# - Population science view from master_results_clean.csv
# - Object view from master_results_clean.csv
# - Fold plots from raw bq-results.csv (NO geometry correction yet)
# - Periodogram / bootstrap / candidate table: Coming soon placeholders
# ==========================================================

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="ATLAST Rotation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Files (edit these if your repo paths differ)
# -------------------------
MASTER_PATH = Path("master_results_clean.csv")                 # required
RAW_PHOTO_PATH = Path("bq-results.csv")                        # optional (raw Rubin First Look photometry)

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure Designation exists
    if "Designation" not in df.columns:
        for c in ["provid", "PROVID", "designation", "name", "object_id"]:
            if c in df.columns:
                df = df.rename(columns={c: "Designation"})
                break
    return df

@st.cache_data(show_spinner=False)
def load_raw_photometry(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)

def format_float(x, nd=6) -> str:
    try:
        if np.isfinite(float(x)):
            return f"{float(x):.{nd}f}"
    except Exception:
        pass
    return "—"

def plot_fold(ax, t_hr: np.ndarray, mag: np.ndarray, bands: np.ndarray, P_hr: float, title: str):
    phase = (t_hr / float(P_hr)) % 1.0
    uniq = sorted(np.unique(bands).tolist())
    for b in uniq:
        m = (bands == b)
        ax.scatter(phase[m], mag[m], s=10, label=str(b))
        ax.scatter(phase[m] + 1.0, mag[m], s=10)
    ax.invert_yaxis()
    ax.set_xlabel("Phase")
    ax.set_ylabel("Raw mag")
    ax.set_title(title)

# -------------------------
# Load master
# -------------------------
if not MASTER_PATH.exists():
    st.error(f"Missing required file: {MASTER_PATH}")
    st.stop()

master = load_master(MASTER_PATH)

# Normalize numeric cols (don’t overwrite strings)
NUM_COLS = [
    "H Mag", "Mean Mag (r Band)", "Number of Observations", "Arc (days)",
    "LS peak period (hr)", "Adopted period (hr)", "Adopted K",
    "2P candidate (hr)", "ΔBIC(2P−P)",
    "Amplitude (Fourier)", "g - r", "g - i", "r - i", "Axial Elongation",
    "Bootstrap top_frac", "Bootstrap n_unique_winners", "Bootstrap family_size",
]
for c in NUM_COLS:
    if c in master.columns:
        master[c] = safe_num(master[c])

# -------------------------
# Header row: title + mode toggle
# -------------------------
h1, h2 = st.columns([0.75, 0.25])
with h1:
    st.title("ATLAST Asteroid Rotation Dashboard")
    st.caption("Population science from master results + object-level raw fold preview (geometry corrections coming soon).")
with h2:
    mode = st.radio(
        "Mode",
        ["Simple", "Research"],
        horizontal=True,
        label_visibility="collapsed",
        key="ui_mode",
    )
is_research = (mode == "Research")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")

# Reliability filter
rel_options = sorted([x for x in master.get("Reliability", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist()])
if not rel_options:
    rel_options = ["reliable", "ambiguous", "insufficient", "unknown"]

selected_rels = st.sidebar.multiselect(
    "Reliability",
    options=rel_options,
    default=rel_options,
)

# Period filter
p_col = "Adopted period (hr)"
pmin = float(np.nanmin(master[p_col])) if p_col in master.columns and master[p_col].notna().any() else 0.0
pmax = float(np.nanmax(master[p_col])) if p_col in master.columns and master[p_col].notna().any() else 100.0
p_lo, p_hi = st.sidebar.slider(
    "Adopted period range (hr)",
    min_value=float(max(0.0, pmin)),
    max_value=float(max(1.0, pmax)),
    value=(float(max(0.0, pmin)), float(max(1.0, pmax))),
)

# Observations filter
n_col = "Number of Observations"
nmin = int(np.nanmin(master[n_col])) if n_col in master.columns and master[n_col].notna().any() else 0
nmax = int(np.nanmax(master[n_col])) if n_col in master.columns and master[n_col].notna().any() else 1000
n_lo, n_hi = st.sidebar.slider(
    "Number of observations",
    min_value=int(max(0, nmin)),
    max_value=int(max(1, nmax)),
    value=(int(max(0, nmin)), int(max(1, nmax))),
)

# Text search
q = st.sidebar.text_input("Search designation contains", value="", placeholder="e.g., 2025 MA19")

# Apply filters
df_f = master.copy()
if "Reliability" in df_f.columns:
    df_f = df_f[df_f["Reliability"].astype(str).isin(selected_rels)]
if p_col in df_f.columns:
    df_f = df_f[df_f[p_col].between(p_lo, p_hi, inclusive="both")]
if n_col in df_f.columns:
    df_f = df_f[df_f[n_col].between(n_lo, n_hi, inclusive="both")]
if q.strip():
    df_f = df_f[df_f["Designation"].astype(str).str.contains(q.strip(), case=False, na=False)]

df_f = df_f.sort_values(["Reliability", "Adopted period (hr)"] if "Reliability" in df_f.columns and "Adopted period (hr)" in df_f.columns else ["Designation"])

st.sidebar.caption(f"{len(df_f):,} objects match filters")

# Pick object
designations = df_f["Designation"].astype(str).tolist()
if not designations:
    st.warning("No objects match your filters.")
    st.stop()

selected = st.sidebar.selectbox("Selected object", options=designations, index=0)

# Selected row
row = df_f[df_f["Designation"].astype(str) == str(selected)]
row = row.iloc[0].to_dict() if len(row) else {}

P_adopt = float(row.get("Adopted period (hr)", np.nan))
P_calc = P_adopt

# Period slider (always visible, but conservative in Simple mode)
if np.isfinite(P_adopt) and P_adopt > 0:
    pct_default = 2.0 if not is_research else 5.0
    pct = st.sidebar.slider("Fold period explore ±%", 0.0, 20.0, pct_default, 0.5)
    lo = max(1e-6, P_adopt * (1.0 - pct / 100.0))
    hi = P_adopt * (1.0 + pct / 100.0)
    P_calc = st.sidebar.slider("Fold period (hr)", float(lo), float(hi), float(P_adopt))
else:
    P_calc = st.sidebar.number_input("Fold period (hr)", min_value=1e-6, value=5.0)

# -------------------------
# Tabs
# -------------------------
tab_pop, tab_obj, tab_photo = st.tabs(["Population", "Object", "Photometry (Raw Fold Preview)"])

# ==========================================================
# TAB: Population
# ==========================================================
with tab_pop:
    st.subheader("Population overview (from master_results_clean.csv)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objects (filtered)", f"{len(df_f):,}")
    if "Reliability" in df_f.columns:
        c2.metric("Reliable", f"{int((df_f['Reliability'].astype(str) == 'reliable').sum()):,}")
        c3.metric("Ambiguous", f"{int((df_f['Reliability'].astype(str) == 'ambiguous').sum()):,}")
        c4.metric("Insufficient", f"{int((df_f['Reliability'].astype(str) == 'insufficient').sum()):,}")
    else:
        c2.metric("Reliable", "—")
        c3.metric("Ambiguous", "—")
        c4.metric("Insufficient", "—")

    # Key scatter: Period vs Amplitude
    if "Adopted period (hr)" in df_f.columns and "Amplitude (Fourier)" in df_f.columns:
        st.markdown("### Period vs amplitude")
        x = df_f["Adopted period (hr)"].to_numpy(float)
        y = df_f["Amplitude (Fourier)"].to_numpy(float)

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.scatter(x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)], s=10)
        ax.set_xlabel("Adopted period (hr)")
        ax.set_ylabel("Amplitude (Fourier, mag)")
        ax.set_title("Period vs amplitude (filtered)")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Period vs amplitude requires 'Adopted period (hr)' and 'Amplitude (Fourier)' in master.")

    # Histogram: adopted period
    if "Adopted period (hr)" in df_f.columns:
        st.markdown("### Adopted period distribution")
        periods = df_f["Adopted period (hr)"].to_numpy(float)
        periods = periods[np.isfinite(periods)]
        bins = 60 if is_research else 40

        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        ax.hist(periods, bins=bins)
        ax.set_xlabel("Adopted period (hr)")
        ax.set_ylabel("Count")
        ax.set_title("Adopted period histogram")
        st.pyplot(fig, clear_figure=True)

    # Table
    st.markdown("### Master table (filtered)")
    show_cols = [
        "Designation",
        "Adopted period (hr)",
        "LS peak period (hr)",
        "Amplitude (Fourier)",
        "Axial Elongation",
        "Reliability",
        "Bootstrap top_frac",
        "Number of Observations",
        "Arc (days)",
    ]
    show_cols = [c for c in show_cols if c in df_f.columns]
    st.dataframe(df_f[show_cols].reset_index(drop=True), use_container_width=True, height=420)

    # Download filtered
    st.download_button(
        "Download filtered master CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="master_results_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ==========================================================
# TAB: Object
# ==========================================================
with tab_obj:
    st.subheader(f"Object summary: {selected}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Adopted P (hr)", format_float(row.get("Adopted period (hr)", np.nan), 6))
    k2.metric("LS peak P (hr)", format_float(row.get("LS peak period (hr)", np.nan), 6))
    k3.metric("Amplitude (mag)", format_float(row.get("Amplitude (Fourier)", np.nan), 3))
    k4.metric("Axial elongation", format_float(row.get("Axial Elongation", np.nan), 3))
    k5.metric("Reliability", str(row.get("Reliability", "—")))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Bootstrap top_frac", format_float(row.get("Bootstrap top_frac", np.nan), 3))
    r2.metric("n_unique_winners", str(row.get("Bootstrap n_unique_winners", "—")))
    r3.metric("family_size", str(row.get("Bootstrap family_size", "—")))
    r4.metric("ΔBIC(2P−P)", format_float(row.get("ΔBIC(2P−P)", np.nan), 3))

    st.markdown("### Colors")
    c1, c2, c3 = st.columns(3)
    c1.metric("g - r", format_float(row.get("g - r", np.nan), 4))
    c2.metric("g - i", format_float(row.get("g - i", np.nan), 4))
    c3.metric("r - i", format_float(row.get("r - i", np.nan), 4))

    if is_research:
        st.markdown("### Ambiguity details")
        amb = row.get("Ambiguous candidates (P_hr:frac)", "")
        if isinstance(amb, str) and amb.strip():
            st.code(amb, language="text")
        else:
            st.caption("No ambiguous candidate string stored for this object.")

    st.markdown("### Coming soon (Research diagnostics)")
    st.info(
        "Periodogram, candidate BIC table, bootstrap histograms, and residual diagnostics will appear here once "
        "per-object Step 11/12 outputs and geometry-corrected photometry are integrated into the dashboard."
    )

# ==========================================================
# TAB: Photometry (Raw fold preview)
# ==========================================================
with tab_photo:
    st.subheader("Raw fold preview (Rubin First Look photometry)")

    st.warning(
        "Caveat: These fold plots use RAW Rubin First Look magnitudes (no geometric corrections / phase-distance correction / "
        "band-centering yet). Geometry-corrected folds are coming soon and will replace this as the default validation view."
    )

    if not RAW_PHOTO_PATH.exists():
        st.error(f"Raw photometry file not found: {RAW_PHOTO_PATH}")
        st.stop()

    df_raw = load_raw_photometry(RAW_PHOTO_PATH)

    # Identify likely columns in raw file
    # time: prefer t_hr else mjd/obstime_mjd (convert to hours since first obs in selection)
    # mag: prefer mag else any magnitude-like column
    # band: band if present
    if "band" not in df_raw.columns:
        df_raw["band"] = "x"
    df_raw["band"] = df_raw["band"].astype(str).str.strip().str.lower()

    # Select object rows (designation column may differ)
    # Try typical name columns
    obj_col = None
    for cand in ["Designation", "designation", "provid", "PROVID", "object", "object_id", "ssobjectid", "ssObjectId"]:
        if cand in df_raw.columns:
            obj_col = cand
            break
    if obj_col is None:
        st.error("Could not find an object identifier column in bq-results.csv (e.g., Designation/provid/ssObjectId).")
        st.stop()

    df_o = df_raw[df_raw[obj_col].astype(str) == str(selected)].copy()
    if len(df_o) == 0:
        st.info(f"No rows in raw photometry matched {obj_col} == '{selected}'.")
        st.stop()

    # Time column detection
    time_col = None
    for cand in ["t_hr", "t_hours", "time_hr", "time_hours"]:
        if cand in df_o.columns:
            time_col = cand
            break

    if time_col is None:
        if "mjd" in df_o.columns:
            t = safe_num(df_o["mjd"]).to_numpy(float)
            t0 = np.nanmin(t) if np.isfinite(t).any() else 0.0
            df_o["t_hr"] = (t - t0) * 24.0
            time_col = "t_hr"
        elif "obstime_mjd" in df_o.columns:
            t = safe_num(df_o["obstime_mjd"]).to_numpy(float)
            t0 = np.nanmin(t) if np.isfinite(t).any() else 0.0
            df_o["t_hr"] = (t - t0) * 24.0
            time_col = "t_hr"
        else:
            st.error("Could not find time column in bq-results.csv. Expected t_hr or mjd/obstime_mjd.")
            st.stop()

    # Mag column detection
    mag_col = None
    for cand in ["mag", "magnitude", "psfMag", "psfmag", "cModelMag", "mag_auto"]:
        if cand in df_o.columns:
            mag_col = cand
            break
    if mag_col is None:
        # last resort: any column containing "mag"
        mag_like = [c for c in df_o.columns if "mag" in c.lower()]
        if mag_like:
            mag_col = mag_like[0]
    if mag_col is None:
        st.error("Could not find a magnitude column in raw photometry (expected 'mag' or similar).")
        st.stop()

    df_o[time_col] = safe_num(df_o[time_col])
    df_o[mag_col] = safe_num(df_o[mag_col])
    df_o = df_o.dropna(subset=[time_col, mag_col])
    if len(df_o) < 5:
        st.warning("Very few usable raw points after cleaning — fold plots may not be informative.")

    t_hr = df_o[time_col].to_numpy(float)
    mag = df_o[mag_col].to_numpy(float)
    bands = df_o["band"].to_numpy(str)

    # 3-panel fold (P/2, P, 2P) — based on selected P_calc
    P_half = 0.5 * float(P_calc)
    P_two = 2.0 * float(P_calc)

    st.markdown("### 3-panel fold: P/2 vs P vs 2P (RAW)")
    cols = st.columns(3)
    periods = [P_half, float(P_calc), P_two]
    titles = [f"P/2 = {P_half:.6f} h", f"P = {float(P_calc):.6f} h", f"2P = {P_two:.6f} h"]

    for col, P_hr, title in zip(cols, periods, titles):
        with col:
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            plot_fold(ax, t_hr=t_hr, mag=mag, bands=bands, P_hr=P_hr, title=title)
            ax.legend(fontsize=7)
            st.pyplot(fig, clear_figure=True)

    st.markdown("### Raw time series (for context)")
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    for b in sorted(np.unique(bands).tolist()):
        m = (bands == b)
        ax.scatter(t_hr[m], mag[m], s=10, label=b)
    ax.invert_yaxis()
    ax.set_xlabel(f"{time_col} (hr since first obs in selection)")
    ax.set_ylabel(mag_col)
    ax.set_title("Raw magnitude vs time (no geometric corrections)")
    ax.legend(fontsize=8, ncol=6)
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Coming soon")
    st.info(
        "Periodogram, bootstrap histogram, candidate tables, and geometry-corrected folds will be enabled once "
        "per-object pipeline outputs (Step 5/11/12) are included in the repo and linked by Designation."
    )
