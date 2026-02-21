# app.py
# ==========================================================
# ATLAST Rotation Dashboard (Master-powered + Raw fold preview)
# CHANGES REQUESTED:
# - Reliability label colored appropriately (Reliable/Ambiguous/Insufficient)
# - Remove the big Caveat warning text; replace with supporting stats on Photometry tab:
#     * Number of observations (from master)
#     * Nights of observations (computed from bq-results.csv, if date/time available)
# - Remove N obs + Arc days from Characterisation (they live on Photometry now)
# - Remove ±% slider entirely (simpler)
# - Replace with a small "Fold Period (hr)" slider + a "Reset to Adopted" button (no session_state crash)
# - Improve titles/labels (Title Case) + cleaner sidebar grouping
# - Fix Reset-to-Adopted Streamlit session_state error by NOT writing to widget keys after creation
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
    page_title="ATLAST Asteroid Rotation Dashboard",
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

def reliability_short(rel: str) -> str:
    r = (rel or "").strip().lower()
    return r if r in {"reliable", "ambiguous", "insufficient"} else "unknown"

def reliability_html(rel: str) -> str:
    r = reliability_short(rel)
    if r == "reliable":
        return '<span style="color:#22c55e;font-weight:700;">Reliable</span>'      # green
    if r == "ambiguous":
        return '<span style="color:#f59e0b;font-weight:700;">Ambiguous</span>'    # amber
    if r == "insufficient":
        return '<span style="color:#ef4444;font-weight:700;">Insufficient</span>' # red
    return '<span style="color:#64748b;font-weight:700;">Unknown</span>'          # slate

def norm_id(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    for ch in [" ", "_", "-", "\t", "\n", "\r"]:
        s = s.replace(ch, "")
    return s

def resolve_time_hours(df: pd.DataFrame) -> tuple[pd.Series, str]:
    hour_cands = ["t_hr", "t_hours", "time_hr", "time_hours", "tHours", "timeHours"]
    for c in hour_cands:
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce")
            if t.notna().sum() >= 3:
                return t, c

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

    jd_cands = ["jd", "JD", "obstime_jd", "obs_jd", "jd_mid", "jd_obs"]
    for c in jd_cands:
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce")
            if t.notna().sum() >= 3:
                t0 = t.min()
                return (t - t0) * 24.0, f"{c} (→ hr since min)"

    dt_cands = ["obstime", "obsTime", "obs_time", "datetime", "date", "time"]
    for c in dt_cands:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dt.notna().sum() >= 3:
                dt0 = dt.min()
                t_hr = (dt - dt0).dt.total_seconds() / 3600.0
                return t_hr, f"{c} (datetime → hr since min)"

    raise ValueError("No recognizable time column found (hours, MJD, JD, or datetime).")

def resolve_nights(df: pd.DataFrame) -> tuple[int | None, str | None]:
    """
    Try to compute number of distinct nights for this object's raw photometry.
    Returns (n_nights, used_col_label).
    """
    # Prefer explicit night column if present
    for c in ["night", "night_id", "night_col", "nightNum", "nightnum"]:
        if c in df.columns:
            s = df[c].astype(str)
            if s.notna().sum() >= 3:
                return int(s.nunique()), c

    # Datetime-ish
    for c in ["obstime", "obsTime", "obs_time", "datetime", "date", "time"]:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dt.notna().sum() >= 3:
                return int(dt.dt.date.nunique()), f"DATE({c})"

    # MJD/JD
    for c in ["mjd", "MJD", "obstime_mjd", "mjd_obs", "midpointMjdTai", "midpointMjdUtc"]:
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce")
            if t.notna().sum() >= 3:
                # Each integer MJD corresponds to a UTC day boundary-ish; good enough for "nights so far" stat.
                return int(np.floor(t).nunique()), f"floor({c})"

    for c in ["jd", "JD", "obstime_jd", "jd_obs"]:
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce")
            if t.notna().sum() >= 3:
                return int(np.floor(t - 0.5).nunique()), f"floor({c}-0.5)"

    return None, None

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

# -------------------------
# Title
# -------------------------
st.markdown("## ATLAST Asteroid Rotation Dashboard")
st.caption("Photometry uses raw Rubin First Look magnitudes for fold previews. Geometry-corrected validation is coming soon.")

# -------------------------
# Sidebar: Asteroid selection
# -------------------------
st.sidebar.markdown("## Asteroid")

q = st.sidebar.text_input("Search Designation", value="", placeholder="e.g., 2025 ME69")

df_pick = master.copy()
if q.strip():
    df_pick = df_pick[df_pick["Designation"].astype(str).str.contains(q.strip(), case=False, na=False)]
df_pick = df_pick.sort_values("Designation")

designations = df_pick["Designation"].astype(str).tolist()
if not designations:
    st.warning("No asteroids match your search.")
    st.stop()

selected = st.sidebar.selectbox("Selected Asteroid", options=designations, index=0)

row = master[master["Designation"].astype(str) == str(selected)]
row = row.iloc[0].to_dict() if len(row) else {}

rel = reliability_short(str(row.get("Reliability", "")))
P_adopt = float(row.get("Adopted period (hr)", np.nan))
if not (np.isfinite(P_adopt) and P_adopt > 0):
    P_adopt = 5.0

# -------------------------
# Sidebar: Fold period controls (no % slider)
# -------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("## Photometry Controls")

# Maintain state without writing into widget keys after creation (prevents StreamlitAPIException)
if "fold_period" not in st.session_state or st.session_state.get("fold_period_for") != selected:
    st.session_state.fold_period = float(P_adopt)
    st.session_state.fold_period_for = selected

# Give a reasonable range so the slider is useful but stable
lo = max(1e-6, float(P_adopt) / 2.0)
hi = float(P_adopt) * 2.0

# Slider drives session_state directly (value= uses current state)
P_calc = st.sidebar.slider(
    "Fold Period (hr)",
    min_value=float(lo),
    max_value=float(hi),
    value=float(st.session_state.fold_period),
    step=float((hi - lo) / 400.0) if hi > lo else 1e-6,
)

st.session_state.fold_period = float(P_calc)

# Reset button uses session_state then rerun (no touching widget keys)
if st.sidebar.button("Reset To Adopted Period", use_container_width=True):
    st.session_state.fold_period = float(P_adopt)
    st.rerun()

# -------------------------
# Sidebar: Population filters (below)
# -------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("## Population Filters")

rel_series = master.get("Reliability", pd.Series([], dtype=str)).dropna().astype(str)
rel_options = sorted(rel_series.unique().tolist()) if len(rel_series) else ["reliable", "ambiguous", "insufficient", "unknown"]

selected_rels = st.sidebar.multiselect("Reliability", options=rel_options, default=rel_options)
if not selected_rels:
    selected_rels = rel_options

p_col = "Adopted period (hr)"
pmin = float(np.nanmin(master[p_col])) if (p_col in master.columns and master[p_col].notna().any()) else 0.0
pmax = float(np.nanmax(master[p_col])) if (p_col in master.columns and master[p_col].notna().any()) else 100.0
p_lo, p_hi = st.sidebar.slider(
    "Adopted Period Range (hr)",
    min_value=float(max(0.0, pmin)),
    max_value=float(max(1.0, pmax)),
    value=(float(max(0.0, pmin)), float(max(1.0, pmax))),
)

n_col = "Number of Observations"
nmin = int(np.nanmin(master[n_col])) if (n_col in master.columns and master[n_col].notna().any()) else 0
nmax = int(np.nanmax(master[n_col])) if (n_col in master.columns and master[n_col].notna().any()) else 1000
n_lo, n_hi = st.sidebar.slider(
    "Number Of Observations",
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

st.sidebar.caption(f"{len(df_f):,} Asteroids Match Filters")

# -------------------------
# Tabs
# -------------------------
tab_photo, tab_char, tab_pop = st.tabs(["Photometry", "Characterisation", "Population"])

# ==========================================================
# Photometry Tab
# ==========================================================
with tab_photo:
    st.markdown(
        f"### Raw Fold Preview: **{selected}** &nbsp;&nbsp;•&nbsp;&nbsp; {reliability_html(rel)}",
        unsafe_allow_html=True,
    )

    # Supporting stats (requested)
    n_obs_master = row.get("Number of Observations", np.nan)
    arc_days = row.get("Arc (days)", np.nan)

    # Load raw photometry for this asteroid (if available) to compute nights
    n_nights = None
    n_nights_note = None

    raw_ok = RAW_PHOTO_PATH.exists()
    df_o = None

    if raw_ok:
        df_raw = load_raw_photometry(RAW_PHOTO_PATH)

        if "band" not in df_raw.columns:
            df_raw["band"] = "x"
        df_raw["band"] = df_raw["band"].astype(str).str.strip().str.lower()

        id_candidates = ["Designation", "designation", "provid", "PROVID", "object", "object_id", "ssobjectid", "ssObjectId"]
        obj_col = next((c for c in id_candidates if c in df_raw.columns), None)

        if obj_col is not None:
            target = norm_id(selected)
            s_norm = df_raw[obj_col].map(norm_id)
            df_o = df_raw[s_norm == target].copy()
            if len(df_o):
                n_nights, n_nights_note = resolve_nights(df_o)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Adopted Period (hr)", format_float(row.get("Adopted period (hr)", np.nan), 6))
    s2.metric("Fold Period (hr)", format_float(P_calc, 6))
    s3.metric("Observations (N)", "—" if pd.isna(n_obs_master) else str(int(n_obs_master)))
    if n_nights is None:
        s4.metric("Nights (Raw)", "—")
    else:
        s4.metric("Nights (Raw)", str(int(n_nights)))

    # Remove N obs + arc from Characterisation as requested, but keep arc as a caption here if available
    if np.isfinite(float(arc_days)):
        st.caption(f"Arc length (days): {format_float(arc_days, 3)}" + (f" • Nights computed from {n_nights_note}" if n_nights_note else ""))

    if not raw_ok:
        st.error(f"Raw photometry file not found: {RAW_PHOTO_PATH}")
        st.stop()

    if df_o is None or len(df_o) == 0:
        st.info("No raw photometry rows found for this asteroid in bq-results.csv.")
        st.stop()

    # Time → hours
    try:
        t_hr_series, time_label = resolve_time_hours(df_o)
        df_o["t_hr"] = pd.to_numeric(t_hr_series, errors="coerce")
    except Exception as e:
        st.error(str(e))
        with st.expander("Debug: Show Columns"):
            st.write(list(df_o.columns))
            st.dataframe(df_o.head(30), use_container_width=True)
        st.stop()

    # Mag column detection
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
        with st.expander("Debug: Show Columns"):
            st.write(list(df_o.columns))
        st.stop()

    df_o[mag_col] = pd.to_numeric(df_o[mag_col], errors="coerce")
    df_o = df_o.dropna(subset=["t_hr", mag_col])
    if len(df_o) < 5:
        st.warning("Very few usable raw points after cleaning — fold plots may not be informative.")

    t_hr = df_o["t_hr"].to_numpy(float)
    mag = df_o[mag_col].to_numpy(float)
    bands = df_o["band"].to_numpy(str)

    # 3-panel fold
    P_half = 0.5 * float(P_calc)
    P_two = 2.0 * float(P_calc)

    st.markdown("#### Three-Panel Fold (P/2 • P • 2P)")
    cols = st.columns(3)
    periods = [P_half, float(P_calc), P_two]
    titles = [f"P/2 = {P_half:.6f} hr", f"P = {float(P_calc):.6f} hr", f"2P = {P_two:.6f} hr"]

    for col, P_hr, title in zip(cols, periods, titles):
        with col:
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            plot_fold(ax, t_hr=t_hr, mag=mag, bands=bands, P_hr=P_hr, title=title, mag_label=mag_col)
            ax.legend(fontsize=7)
            st.pyplot(fig, clear_figure=True)

    st.markdown("#### Raw Magnitude vs Time")
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    for b in sorted(np.unique(bands).tolist()):
        m = (bands == b)
        ax.scatter(t_hr[m], mag[m], s=10, label=b)
    ax.invert_yaxis()
    ax.set_xlabel(time_label)
    ax.set_ylabel(mag_col)
    ax.set_title("Raw magnitude vs time")
    ax.legend(fontsize=8, ncol=6)
    st.pyplot(fig, clear_figure=True)

# ==========================================================
# Characterisation Tab
# ==========================================================
with tab_char:
    st.markdown(
        f"### Characterisation: **{selected}** &nbsp;&nbsp;•&nbsp;&nbsp; {reliability_html(rel)}",
        unsafe_allow_html=True,
    )
    st.caption("All values on this tab come from master_results_clean.csv (Step 13 summary exports).")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Adopted Period (hr)", format_float(row.get("Adopted period (hr)", np.nan), 6))
    k2.metric("LS Peak Period (hr)", format_float(row.get("LS peak period (hr)", np.nan), 6))
    k3.metric("Adopted K", "—" if pd.isna(row.get("Adopted K", np.nan)) else str(int(row.get("Adopted K"))))
    k4.metric("Amplitude (mag)", format_float(row.get("Amplitude (Fourier)", np.nan), 3))
    k5.metric("Axial Elongation", format_float(row.get("Axial Elongation", np.nan), 3))

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("2P Candidate (hr)", format_float(row.get("2P candidate (hr)", np.nan), 6))
    b2.metric("ΔBIC(2P−P)", format_float(row.get("ΔBIC(2P−P)", np.nan), 3))
    b3.metric("Bootstrap Top_Frac", format_float(row.get("Bootstrap top_frac", np.nan), 3))
    b4.metric("Unique Winners", "—" if pd.isna(row.get("Bootstrap n_unique_winners", np.nan)) else str(int(row.get("Bootstrap n_unique_winners"))))
    b5.metric("Family Size", "—" if pd.isna(row.get("Bootstrap family_size", np.nan)) else str(int(row.get("Bootstrap family_size"))))

    st.markdown("#### Colors")
    c1, c2, c3 = st.columns(3)
    c1.metric("g − r", format_float(row.get("g - r", np.nan), 4))
    c2.metric("g − i", format_float(row.get("g - i", np.nan), 4))
    c3.metric("r − i", format_float(row.get("r - i", np.nan), 4))

    st.markdown("#### Coming Soon")
    st.info("Candidate BIC tables, bootstrap histograms, residual diagnostics, and geometry-corrected validation plots will appear here soon.")

# ==========================================================
# Population Tab
# ==========================================================
with tab_pop:
    st.markdown("### Population Overview (Filtered)")
    if len(df_f) == 0:
        st.warning("No asteroids match your current population filters. Widen the ranges or reselect reliability options.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Asteroids (Filtered)", f"{len(df_f):,}")
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
        st.markdown("#### Period vs Amplitude")
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
        st.markdown("#### Adopted Period Distribution")
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

    st.markdown("#### Master Table (Filtered)")
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
        "Download Filtered Master CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="master_results_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
