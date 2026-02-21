# app.py
# ==========================================================
# ATLAST Rotation Dashboard (Master-powered + Raw fold caveat)
# FIXES / UPGRADES:
# - Sidebar "Object" -> "Asteroid"
# - Photometry tab header shows: "Raw fold preview: <asteroid> (reliable/ambiguous/insufficient)"
# - Fold controls order: Fold period slider ABOVE ±% slider
# - "Reset to adopted" button restores default fold period + default ±% range
# - Population tab no longer “blank” when Reliability multiselect is empty:
#     * Empty selection is treated as “all reliabilities”
#     * If filters still yield 0 objects, shows a clear warning + skips plots cleanly
# - Raw photometry matching improved via normalization (spaces/underscores/case)
# ==========================================================

from __future__ import annotations

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
# Files (edit if paths differ)
# -------------------------
MASTER_PATH = Path("master_results_clean.csv")   # required
RAW_PHOTO_PATH = Path("bq-results.csv")          # optional (raw Rubin First Look photometry)

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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

def format_float(x, nd=6) -> str:
    try:
        v = float(x)
        if np.isfinite(v):
            return f"{v:.{nd}f}"
    except Exception:
        pass
    return "—"

def reliability_badge(rel: str) -> str:
    r = (rel or "").strip().lower()
    if r == "reliable":
        return "✅ reliable"
    if r == "ambiguous":
        return "⚠️ ambiguous"
    if r == "insufficient":
        return "❌ insufficient"
    return "—"

def reliability_short(rel: str) -> str:
    r = (rel or "").strip().lower()
    return r if r in {"reliable", "ambiguous", "insufficient"} else "unknown"

def norm_id(x) -> str:
    """Normalize IDs for matching across files (spaces/underscores/case differences)."""
    if x is None:
        return ""
    s = str(x).strip().lower()
    # collapse common separators
    for ch in [" ", "_", "-", "\t", "\n", "\r"]:
        s = s.replace(ch, "")
    return s

def resolve_time_hours(df: pd.DataFrame) -> tuple[pd.Series, str]:
    # 1) Already-hours candidates
    hour_cands = ["t_hr", "t_hours", "time_hr", "time_hours", "tHours", "timeHours"]
    for c in hour_cands:
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce")
            if t.notna().sum() >= 3:
                return t, c

    # 2) MJD-like numeric candidates
    mjd_cands = [
        "mjd", "MJD", "obstime_mjd", "obs_mjd", "mjd_mid", "mjd_obs",
        "mjd_tai", "mjd_utc", "midpointMjdTai", "midpointMjdUtc"
    ]
    for c in mjd_cands:
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce")
            if t.notna().sum() >= 3:
                t0 = t.min()
                return (t - t0) * 24.0, f"{c} (→ hr since min)"

    # 3) JD-like numeric candidates
    jd_cands = ["jd", "JD", "obstime_jd", "obs_jd", "jd_mid", "jd_obs"]
    for c in jd_cands:
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce")
            if t.notna().sum() >= 3:
                t0 = t.min()
                return (t - t0) * 24.0, f"{c} (→ hr since min)"

    # 4) Datetime/ISO string candidates
    dt_cands = ["obstime", "obsTime", "obs_time", "datetime", "date", "time"]
    for c in dt_cands:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dt.notna().sum() >= 3:
                dt0 = dt.min()
                t_hr = (dt - dt0).dt.total_seconds() / 3600.0
                return t_hr, f"{c} (datetime → hr since min)"

    raise ValueError("No recognizable time column found (hours, MJD, JD, or datetime).")

def plot_fold(ax, t_hr: np.ndarray, mag: np.ndarray, bands: np.ndarray, P_hr: float, title: str, mag_label: str):
    phase = (t_hr / float(P_hr)) % 1.0
    uniq = sorted(np.unique(bands).tolist())
    for b in uniq:
        m = (bands == b)
        ax.scatter(phase[m], mag[m], s=10, label=str(b))
        ax.scatter(phase[m] + 1.0, mag[m], s=10)
    ax.invert_yaxis()
    ax.set_xlabel("Phase")
    ax.set_ylabel(mag_label)
    ax.set_title(title)

# -------------------------
# Load master
# -------------------------
if not MASTER_PATH.exists():
    st.error(f"Missing required file: {MASTER_PATH}")
    st.stop()

master = load_master(MASTER_PATH)

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

# Default slider settings
DEFAULT_PCT = 5.0

# Persist UI state
if "fold_pct" not in st.session_state:
    st.session_state.fold_pct = DEFAULT_PCT
if "fold_period" not in st.session_state:
    st.session_state.fold_period = None  # set after we know adopted P

# -------------------------
# Header
# -------------------------
st.title("ATLAST Asteroid Rotation Dashboard")
st.caption(
    "Photometry shows a raw fold preview from Rubin First Look photometry (no geometric corrections yet). "
    "Characterisation + Population are master-file driven."
)

# -------------------------
# Sidebar: ASTEROID selection + fold controls
# -------------------------
st.sidebar.header("Asteroid")

q = st.sidebar.text_input("Search designation contains", value="", placeholder="e.g., 2025 MB36")

df_pick = master.copy()
if q.strip():
    df_pick = df_pick[df_pick["Designation"].astype(str).str.contains(q.strip(), case=False, na=False)]
df_pick = df_pick.sort_values("Designation")

designations = df_pick["Designation"].astype(str).tolist()
if not designations:
    st.warning("No asteroids match your search.")
    st.stop()

selected = st.sidebar.selectbox("Selected asteroid", options=designations, index=0)

row = master[master["Designation"].astype(str) == str(selected)]
row = row.iloc[0].to_dict() if len(row) else {}

P_adopt = float(row.get("Adopted period (hr)", np.nan))
if not (np.isfinite(P_adopt) and P_adopt > 0):
    P_adopt = 5.0  # fallback for UI

# Initialize fold period state once per new selection
if st.session_state.fold_period is None or st.session_state.get("fold_period_for") != selected:
    st.session_state.fold_period = float(P_adopt)
    st.session_state.fold_period_for = selected
    st.session_state.fold_pct = DEFAULT_PCT

st.sidebar.subheader("Photometry fold controls")

# Reset button under period slider (requested)
# We'll place it after the period slider, but the button affects both sliders.
# (Streamlit buttons trigger rerun immediately.)
# Control order requested: Fold period slider ABOVE ±% slider.
# To do that: choose pct first (defines bounds), but DISPLAY period slider first by using current pct value.
pct = float(st.session_state.fold_pct)
lo = max(1e-6, float(P_adopt) * (1.0 - pct / 100.0))
hi = float(P_adopt) * (1.0 + pct / 100.0)
# Clamp current period into bounds
curP = float(st.session_state.fold_period)
curP = min(max(curP, lo), hi)

st.session_state.fold_period = st.sidebar.slider(
    "Fold period (hr)",
    min_value=float(lo),
    max_value=float(hi),
    value=float(curP),
    step=float((hi - lo) / 400.0) if hi > lo else 1e-6,
    key="fold_period_slider",
)

if st.sidebar.button("Reset to adopted", use_container_width=True):
    st.session_state.fold_pct = DEFAULT_PCT
    st.session_state.fold_period = float(P_adopt)
    # Also update the widget keys on rerun
    st.session_state["fold_pct_slider"] = DEFAULT_PCT
    st.session_state["fold_period_slider"] = float(P_adopt)
    st.rerun()

st.session_state.fold_pct = st.sidebar.slider(
    "Explore ±% around adopted P",
    0.0, 20.0, float(st.session_state.fold_pct), 0.5,
    key="fold_pct_slider",
)

# Recompute bounds after pct change and clamp fold_period if needed
pct2 = float(st.session_state.fold_pct)
lo2 = max(1e-6, float(P_adopt) * (1.0 - pct2 / 100.0))
hi2 = float(P_adopt) * (1.0 + pct2 / 100.0)
if not (lo2 <= float(st.session_state.fold_period) <= hi2):
    st.session_state.fold_period = float(min(max(float(st.session_state.fold_period), lo2), hi2))

P_calc = float(st.session_state.fold_period)

# -------------------------
# Sidebar: POPULATION filters (below object sliders)
# -------------------------
st.sidebar.markdown("---")
st.sidebar.header("Population filters")

rel_series = master.get("Reliability", pd.Series([], dtype=str)).dropna().astype(str)
rel_options = sorted(rel_series.unique().tolist()) if len(rel_series) else ["reliable", "ambiguous", "insufficient", "unknown"]

# default to all
selected_rels = st.sidebar.multiselect("Reliability", options=rel_options, default=rel_options)

# IMPORTANT: if user clears it, treat as "all" (prevents blank population view confusion)
if not selected_rels:
    selected_rels = rel_options

p_col = "Adopted period (hr)"
if p_col in master.columns and master[p_col].notna().any():
    pmin = float(np.nanmin(master[p_col]))
    pmax = float(np.nanmax(master[p_col]))
else:
    pmin, pmax = 0.0, 100.0

p_lo, p_hi = st.sidebar.slider(
    "Adopted period range (hr)",
    min_value=float(max(0.0, pmin)),
    max_value=float(max(1.0, pmax)),
    value=(float(max(0.0, pmin)), float(max(1.0, pmax))),
)

n_col = "Number of Observations"
if n_col in master.columns and master[n_col].notna().any():
    nmin = int(np.nanmin(master[n_col]))
    nmax = int(np.nanmax(master[n_col]))
else:
    nmin, nmax = 0, 1000

n_lo, n_hi = st.sidebar.slider(
    "Number of observations",
    min_value=int(max(0, nmin)),
    max_value=int(max(1, nmax)),
    value=(int(max(0, nmin)), int(max(1, nmax))),
)

df_f = master.copy()
if "Reliability" in df_f.columns:
    df_f = df_f[df_f["Reliability"].astype(str).isin(selected_rels)]
if p_col in df_f.columns:
    df_f = df_f[df_f[p_col].between(p_lo, p_hi, inclusive="both")]
if n_col in df_f.columns:
    df_f = df_f[df_f[n_col].between(n_lo, n_hi, inclusive="both")]

st.sidebar.caption(f"{len(df_f):,} objects match population filters")

# ==========================================================
# Tabs: Photometry -> Characterisation -> Population
# ==========================================================
tab_photo, tab_char, tab_pop = st.tabs(["Photometry", "Characterisation", "Population"])

# ==========================================================
# TAB: Photometry (raw fold preview)
# ==========================================================
with tab_photo:
    rel_here = reliability_short(str(row.get("Reliability", "")))
    st.subheader(f"Raw fold preview: {selected} ({rel_here})")

    st.warning(
        "Caveat: These fold plots use RAW Rubin First Look magnitudes (no geometric corrections / phase-distance correction / "
        "band-centering yet). Geometry-corrected folds are coming soon."
    )

    if not RAW_PHOTO_PATH.exists():
        st.error(f"Raw photometry file not found: {RAW_PHOTO_PATH}")
        st.stop()

    df_raw = load_raw_photometry(RAW_PHOTO_PATH)

    # Band
    if "band" not in df_raw.columns:
        df_raw["band"] = "x"
    df_raw["band"] = df_raw["band"].astype(str).str.strip().str.lower()

    # Object id column discovery (we'll try several; then match by normalized value)
    id_candidates = ["Designation", "designation", "provid", "PROVID", "object", "object_id", "ssobjectid", "ssObjectId"]
    obj_col = next((c for c in id_candidates if c in df_raw.columns), None)
    if obj_col is None:
        st.error("Could not find an object identifier column in bq-results.csv (e.g., Designation/provid/ssObjectId).")
        with st.expander("Debug: show columns"):
            st.write(list(df_raw.columns))
        st.stop()

    # Normalized match
    target = norm_id(selected)
    s_norm = df_raw[obj_col].map(norm_id)
    df_o = df_raw[s_norm == target].copy()

    if len(df_o) == 0:
        st.info(f"No rows in raw photometry matched {obj_col} == '{selected}' (after normalization).")
        with st.expander("Debug: show unique IDs (first 200)"):
            st.write(df_raw[obj_col].astype(str).unique().tolist()[:200])
        st.stop()

    # Time → hours
    try:
        t_hr_series, time_label = resolve_time_hours(df_o)
        df_o["t_hr"] = pd.to_numeric(t_hr_series, errors="coerce")
    except Exception as e:
        st.error(str(e))
        with st.expander("Debug: show columns"):
            st.write(list(df_o.columns))
            st.dataframe(df_o.head(30), use_container_width=True)
        st.stop()

    # Magnitude column detection
    mag_col = None
    for cand in ["mag", "magnitude", "psfMag", "psfmag", "cModelMag", "mag_auto"]:
        if cand in df_o.columns:
            mag_col = cand
            break
    if mag_col is None:
        mag_like = [c for c in df_o.columns if "mag" in c.lower()]
        if mag_like:
            mag_col = mag_like[0]
    if mag_col is None:
        st.error("Could not find a magnitude column in raw photometry (expected 'mag' or similar).")
        with st.expander("Debug: show columns"):
            st.write(list(df_o.columns))
        st.stop()

    df_o[mag_col] = pd.to_numeric(df_o[mag_col], errors="coerce")
    df_o = df_o.dropna(subset=["t_hr", mag_col])
    if len(df_o) < 5:
        st.warning("Very few usable raw points after cleaning — fold plots may not be informative.")

    t_hr = df_o["t_hr"].to_numpy(float)
    mag = df_o[mag_col].to_numpy(float)
    bands = df_o["band"].to_numpy(str)

    # 3-panel fold using current P_calc
    P_half = 0.5 * float(P_calc)
    P_two = 2.0 * float(P_calc)

    st.markdown("### 3-panel fold: P/2 vs P vs 2P (RAW)")
    cols = st.columns(3)
    periods = [P_half, float(P_calc), P_two]
    titles = [f"P/2 = {P_half:.6f} h", f"P = {float(P_calc):.6f} h", f"2P = {P_two:.6f} h"]

    for col, P_hr, title in zip(cols, periods, titles):
        with col:
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            plot_fold(ax, t_hr=t_hr, mag=mag, bands=bands, P_hr=P_hr, title=title, mag_label=mag_col)
            ax.legend(fontsize=7)
            st.pyplot(fig, clear_figure=True)

    st.markdown("### Raw time series (context)")
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    for b in sorted(np.unique(bands).tolist()):
        m = (bands == b)
        ax.scatter(t_hr[m], mag[m], s=10, label=b)
    ax.invert_yaxis()
    ax.set_xlabel(time_label)
    ax.set_ylabel(mag_col)
    ax.set_title("Raw magnitude vs time (no geometric corrections)")
    ax.legend(fontsize=8, ncol=6)
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Coming soon")
    st.info(
        "Periodogram, bootstrap histogram, candidate tables, and geometry-corrected folds will be enabled once "
        "per-object pipeline outputs (Step 5/11/12) are included in the repo and linked by Designation."
    )

# ==========================================================
# TAB: Characterisation (master KPIs)
# ==========================================================
with tab_char:
    st.subheader(f"Characterisation: {selected}")
    st.caption("Values in this tab come from master_results_clean.csv (Step 13 exports).")

    st.markdown(f"**Status:** {reliability_badge(str(row.get('Reliability', '')))}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Adopted P (hr)", format_float(row.get("Adopted period (hr)", np.nan), 6))
    k2.metric("LS peak P (hr)", format_float(row.get("LS peak period (hr)", np.nan), 6))
    k3.metric("Adopted K", "—" if pd.isna(row.get("Adopted K", np.nan)) else str(int(row.get("Adopted K"))))
    k4.metric("Amplitude (mag)", format_float(row.get("Amplitude (Fourier)", np.nan), 3))
    k5.metric("Axial elongation", format_float(row.get("Axial Elongation", np.nan), 3))

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("N obs", "—" if pd.isna(row.get("Number of Observations", np.nan)) else str(int(row.get("Number of Observations"))))
    b2.metric("Arc (days)", format_float(row.get("Arc (days)", np.nan), 3))
    b3.metric("2P candidate (hr)", format_float(row.get("2P candidate (hr)", np.nan), 6))
    b4.metric("ΔBIC(2P−P)", format_float(row.get("ΔBIC(2P−P)", np.nan), 3))
    b5.metric("Bootstrap top_frac", format_float(row.get("Bootstrap top_frac", np.nan), 3))

    st.markdown("### Colors")
    c1, c2, c3 = st.columns(3)
    c1.metric("g - r", format_float(row.get("g - r", np.nan), 4))
    c2.metric("g - i", format_float(row.get("g - i", np.nan), 4))
    c3.metric("r - i", format_float(row.get("r - i", np.nan), 4))

    st.markdown("### Coming soon (deep diagnostics)")
    st.info(
        "Candidate BIC table, bootstrap histograms, residual diagnostics, and geometry-corrected fold validation "
        "will appear once per-object Step 11/12 outputs are integrated."
    )

# ==========================================================
# TAB: Population (filtered)
# ==========================================================
with tab_pop:
    st.subheader("Population overview (filtered by sidebar)")

    if len(df_f) == 0:
        st.warning(
            "No objects match your current population filters. "
            "Tip: re-add Reliability options (or click the 'x' chips) and/or widen the period / N_obs ranges."
        )
        st.stop()

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

    # Period vs amplitude
    if "Adopted period (hr)" in df_f.columns and "Amplitude (Fourier)" in df_f.columns:
        st.markdown("### Period vs amplitude")
        x = df_f["Adopted period (hr)"].to_numpy(float)
        y = df_f["Amplitude (Fourier)"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)

        if m.sum() == 0:
            st.info("No finite Period/Amplitude points under current filters.")
        else:
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.scatter(x[m], y[m], s=10)
            ax.set_xlabel("Adopted period (hr)")
            ax.set_ylabel("Amplitude (Fourier, mag)")
            ax.set_title("Period vs amplitude (filtered)")
            st.pyplot(fig, clear_figure=True)

    # Histogram
    if "Adopted period (hr)" in df_f.columns:
        st.markdown("### Adopted period distribution")
        periods = df_f["Adopted period (hr)"].to_numpy(float)
        periods = periods[np.isfinite(periods)]
        if len(periods) == 0:
            st.info("No finite adopted periods under current filters.")
        else:
            fig, ax = plt.subplots(figsize=(8.5, 4.0))
            ax.hist(periods, bins=50)
            ax.set_xlabel("Adopted period (hr)")
            ax.set_ylabel("Count")
            ax.set_title("Adopted period histogram")
            st.pyplot(fig, clear_figure=True)

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
    st.dataframe(df_f[show_cols].reset_index(drop=True), use_container_width=True, height=460)

    st.download_button(
        "Download filtered master CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="master_results_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
