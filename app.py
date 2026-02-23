# app.py
# ==========================================================
# ATLAST Rotation Dashboard (Master-powered + GEOMETRY-CORRECTED fold preview)
# UPDATE:
# - Photometry now comes from BigQuery on demand (no RFL.csv)
# - Runs Step 5 geometry correction on the fly for selected asteroid
# - Folds and raw-vs-time plots use geometry-corrected mags (mag_geo_bandcenter by default)
# - Queries ONLY the columns needed to minimize BigQuery cost
# ==========================================================

from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# BigQuery client (requires google-cloud-bigquery in requirements)
from google.cloud import bigquery

# Step 5 function must be importable here.
# If your Step 5 is in a separate file, do:
# from lsst_functions import step5_geometry_horizons_range
#
# Otherwise, paste the function definition above this app.py.
from step5_module import step5_geometry_horizons_range  # <-- CHANGE THIS IMPORT


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="ATLAST Asteroid Rotation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Local file(s)
# -------------------------
MASTER_PATH = Path("master_results_clean.csv")   # required

# -------------------------
# BigQuery config (EDIT THESE)
# -------------------------
BQ_PROJECT = "lsst-484623"
BQ_DATASET = "asteroid_institute_mpc_replica"
BQ_TABLE   = "public_obs_sbn"

# Station filter (keeps query small)
BQ_STN = "X05"

# Optional safety limit (should be plenty for one object)
BQ_ROW_LIMIT = 20000

# Horizons location should match station for correction
HORIZONS_LOCATION = "X05"


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
        return '<span style="color:#22c55e;font-weight:800;">Reliable</span>'
    if r == "ambiguous":
        return '<span style="color:#f59e0b;font-weight:800;">Ambiguous</span>'
    if r == "insufficient":
        return '<span style="color:#ef4444;font-weight:800;">Insufficient</span>'
    return '<span style="color:#64748b;font-weight:800;">Unknown</span>'

def norm_id(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    for ch in [" ", "_", "-", "\t", "\n", "\r"]:
        s = s.replace(ch, "")
    return s

def resolve_nights(df: pd.DataFrame) -> int | None:
    for c in ["night", "night_id", "night_col", "nightNum", "nightnum"]:
        if c in df.columns:
            s = df[c].astype(str)
            if s.notna().sum() >= 3:
                return int(s.nunique())

    for c in ["obstime_dt", "obstime", "obsTime", "obs_time", "datetime", "date", "time"]:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dt.notna().sum() >= 3:
                return int(dt.dt.date.nunique())
    return None

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
# BigQuery access (cached)
# -------------------------
@st.cache_resource
def get_bq_client() -> bigquery.Client:
    # On Streamlit Cloud, configure credentials via st.secrets and GOOGLE_APPLICATION_CREDENTIALS
    return bigquery.Client(project=BQ_PROJECT)

@st.cache_data(show_spinner=False, ttl=3600)
def bq_load_photometry_for_provid(provid: str) -> pd.DataFrame:
    """
    Minimal-column query for ONE asteroid.
    Columns needed for Step 5 + plotting:
      - obstime (timestamp)
      - mag (float)
      - rmsmag (float, optional weights)
      - band (str)
      - provid (id)
    """
    client = get_bq_client()

    q = f"""
    SELECT
      provid,
      obstime,
      band,
      SAFE_CAST(mag AS FLOAT64) AS mag,
      SAFE_CAST(rmsmag AS FLOAT64) AS rmsmag
    FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
    WHERE stn = @stn
      AND provid = @prov
      AND SAFE_CAST(mag AS FLOAT64) IS NOT NULL
    ORDER BY obstime
    LIMIT {int(BQ_ROW_LIMIT)}
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("stn", "STRING", BQ_STN),
            bigquery.ScalarQueryParameter("prov", "STRING", provid),
        ]
    )
    df = client.query(q, job_config=job_config).to_dataframe()
    return df

def make_df1_from_bq(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert BigQuery photometry slice into df1 expected by Step 5:
      obstime_dt (UTC datetime), mag, band, optional rmsmag, plus t_hr.
    """
    df = df_raw.copy()

    # Ensure expected columns
    if "obstime" not in df.columns:
        raise ValueError("BigQuery result missing 'obstime' column.")
    if "mag" not in df.columns:
        raise ValueError("BigQuery result missing 'mag' column.")
    if "band" not in df.columns:
        df["band"] = "x"

    df["obstime_dt"] = pd.to_datetime(df["obstime"], errors="coerce", utc=True)
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    if "rmsmag" in df.columns:
        df["rmsmag"] = pd.to_numeric(df["rmsmag"], errors="coerce")
    df["band"] = df["band"].astype(str).str.strip().str.lower()

    df = df.dropna(subset=["obstime_dt", "mag", "band"]).sort_values("obstime_dt").reset_index(drop=True)
    if len(df) == 0:
        return df

    t0 = df["obstime_dt"].min()
    df["t_hr"] = (df["obstime_dt"] - t0).dt.total_seconds() / 3600.0
    return df

@st.cache_data(show_spinner=False, ttl=24*3600)
def geo_correct_cached(df1: pd.DataFrame, provid: str) -> tuple[pd.DataFrame, dict]:
    """
    Run Step 5 on-the-fly but do NOT write files.
    Cache results so repeated asteroid selections are instant.
    """
    df_geo, meta = step5_geometry_horizons_range(
        df1,
        PROVID=provid,
        OUTDIR=".",  # unused when save_tables/save_plots False
        HORIZONS_LOCATION=HORIZONS_LOCATION,
        save_tables=False,
        save_plots=False,
        show_plots=False,
        FAIL_ON_UNMATCHED=False,
        verbose=False,
    )
    return df_geo, meta


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
st.caption("Photometry is loaded from BigQuery per asteroid and folded using on-the-fly geometry correction (Horizons).")


# -------------------------
# Sidebar: Asteroid selection
# -------------------------
st.sidebar.markdown("## Asteroid")

q = st.sidebar.text_input("Search Designation", value="", placeholder="E.g., 2025 ME69")

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
# Sidebar: Fold controls
# -------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("## Fold Controls")

if "fold_period" not in st.session_state or st.session_state.get("fold_period_for") != selected:
    st.session_state.fold_period = float(P_adopt)
    st.session_state.fold_period_for = selected

lo = max(1e-6, float(P_adopt) / 2.0)
hi = float(P_adopt) * 2.0

P_calc = st.sidebar.slider(
    "Fold Period (Hr)",
    min_value=float(lo),
    max_value=float(hi),
    value=float(st.session_state.fold_period),
    step=float((hi - lo) / 400.0) if hi > lo else 1e-6,
)
st.session_state.fold_period = float(P_calc)

if st.sidebar.button("Reset To Adopted Period", use_container_width=True):
    st.session_state.fold_period = float(P_adopt)
    st.rerun()

# ---- Bands filter (keep UI / options) ----
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]

if "raw_band_filter" not in st.session_state:
    st.session_state.raw_band_filter = LSST_BANDS[:]  # default all
if "raw_band_for" not in st.session_state:
    st.session_state.raw_band_for = None

sel_bands_sidebar = st.sidebar.multiselect(
    "Bands",
    options=LSST_BANDS,
    default=st.session_state.raw_band_filter if isinstance(st.session_state.raw_band_filter, list) else LSST_BANDS,
    key="raw_band_widget",
)
if not sel_bands_sidebar:
    sel_bands_sidebar = LSST_BANDS
st.session_state.raw_band_filter = sel_bands_sidebar


# -------------------------
# Sidebar: Population filters (unchanged)
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
    "Adopted Period Range (Hr)",
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
# Photometry Tab (BigQuery + Step 5 on the fly)
# ==========================================================
with tab_photo:
    st.markdown(
        f"### Geometry-Corrected Fold Preview: **{selected}** &nbsp;&nbsp;•&nbsp;&nbsp; {reliability_html(rel)}",
        unsafe_allow_html=True,
    )

    n_obs_master = row.get("Number of Observations", np.nan)
    arc_days = row.get("Arc (days)", np.nan)

    # ---- Load from BigQuery ----
    with st.spinner("Querying BigQuery photometry for this asteroid..."):
        # If your master Designation is NOT exactly the BigQuery provid, change mapping here.
        # For numbered objects, you can query permid instead (requires SQL changes).
        df_raw = bq_load_photometry_for_provid(str(selected))

    if df_raw is None or len(df_raw) == 0:
        st.info("No photometry rows found in BigQuery for this asteroid (stn=X05).")
        st.stop()

    df1 = make_df1_from_bq(df_raw)
    if len(df1) < 5:
        st.warning("Very few usable points after cleaning. Cannot fold reliably.")
        st.dataframe(df1.head(50), use_container_width=True)
        st.stop()

    # ---- Geometry correction on the fly ----
    with st.spinner("Running on-the-fly geometry correction (Horizons)..."):
        try:
            df_geo, meta5 = geo_correct_cached(df1, str(selected))
        except Exception as e:
            st.error("Geometry correction failed (Horizons). Falling back to raw mags.")
            st.exception(e)
            df_geo = df1.copy()
            df_geo["mag_geo_bandcenter"] = np.nan
            df_geo["mag_geo"] = np.nan
            meta5 = {}

    # Choose corrected series (prefer band-centered)
    mag_col = "mag_geo_bandcenter" if ("mag_geo_bandcenter" in df_geo.columns and df_geo["mag_geo_bandcenter"].notna().sum() >= 5) else "mag"

    # Apply band filter from sidebar (keep only selected LSST bands that exist)
    df_geo["band"] = df_geo["band"].astype(str).str.strip().str.lower()
    avail_bands = set(df_geo["band"].unique().tolist())
    sel_bands = [b for b in st.session_state.raw_band_filter if b in avail_bands]
    if not sel_bands:
        sel_bands = sorted(list(avail_bands))

    dfp = df_geo[df_geo["band"].astype(str).isin(sel_bands)].copy()
    dfp = dfp.dropna(subset=["t_hr", mag_col, "band"])

    n_nights = resolve_nights(dfp)

    # Stats row
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Adopted Period (Hr)", format_float(row.get("Adopted period (hr)", np.nan), 6))
    s2.metric("Fold Period (Hr)", format_float(P_calc, 6))
    s3.metric("Observations (Master)", "—" if pd.isna(n_obs_master) else str(int(n_obs_master)))
    s4.metric("Nights (Photometry)", "—" if n_nights is None else str(int(n_nights)))

    if np.isfinite(float(arc_days)):
        st.caption(f"Arc Length (Days): {format_float(arc_days, 3)}")

    # Download corrected photometry for this object
    st.download_button(
        "Download Geometry-Corrected Photometry (CSV)",
        data=df_geo.to_csv(index=False).encode("utf-8"),
        file_name=f"{norm_id(selected)}_geo_corrected.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if mag_col == "mag":
        st.warning("Using raw 'mag' for folding (mag_geo_bandcenter not available).")
    else:
        st.caption("Folding using geometry-corrected, band-centered magnitudes: mag_geo_bandcenter")

    if len(dfp) < 5:
        st.warning("Very few points remain after band filtering. Try selecting more bands.")
        st.stop()

    t_hr = dfp["t_hr"].to_numpy(float)
    mag = dfp[mag_col].to_numpy(float)
    bands = dfp["band"].to_numpy(str)

    # 3-panel fold
    P_half = 0.5 * float(P_calc)
    P_two = 2.0 * float(P_calc)

    st.markdown("#### Three-Panel Fold (P/2 • P • 2P)")
    cols = st.columns(3)
    periods = [P_half, float(P_calc), P_two]
    titles = [f"P/2 = {P_half:.6f} Hr", f"P = {float(P_calc):.6f} Hr", f"2P = {P_two:.6f} Hr"]

    for col, P_hr, title in zip(cols, periods, titles):
        with col:
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            plot_fold(ax, t_hr=t_hr, mag=mag, bands=bands, P_hr=P_hr, title=title, mag_label=mag_col)
            ax.legend(fontsize=7)
            st.pyplot(fig, clear_figure=True)

    # Magnitude vs Time (corrected)
    st.markdown("#### Magnitude vs Time (Geometry-Corrected)")
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    for b in sorted(np.unique(bands).tolist()):
        m = (bands == b)
        ax.scatter(t_hr[m], mag[m], s=10, label=b)
    ax.invert_yaxis()
    ax.set_xlabel("Hours Since First Observation")
    ax.set_ylabel(mag_col)
    ax.set_title("Magnitude vs Time")
    ax.legend(fontsize=8, ncol=6)
    st.pyplot(fig, clear_figure=True)

    # Optional: show Step 5 QA summary
    with st.expander("Geometry Correction QA (Step 5 meta)", expanded=False):
        st.json(meta5)


# ==========================================================
# Characterisation Tab (unchanged)
# ==========================================================
with tab_char:
    st.markdown(
        f"### Characterisation: **{selected}** &nbsp;&nbsp;•&nbsp;&nbsp; {reliability_html(rel)}",
        unsafe_allow_html=True,
    )
    st.caption("All values on this tab come from master_results_clean.csv (Step 13 Summary Exports).")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Adopted Period (Hr)", format_float(row.get("Adopted period (hr)", np.nan), 6))
    k2.metric("LS Peak Period (Hr)", format_float(row.get("LS peak period (hr)", np.nan), 6))
    k3.metric("Adopted K", "—" if pd.isna(row.get("Adopted K", np.nan)) else str(int(row.get("Adopted K"))))
    k4.metric("Amplitude (Mag)", format_float(row.get("Amplitude (Fourier)", np.nan), 3))
    k5.metric("Axial Elongation", format_float(row.get("Axial Elongation", np.nan), 3))

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("2P Candidate (Hr)", format_float(row.get("2P candidate (hr)", np.nan), 6))
    b2.metric("ΔBIC(2P−P)", format_float(row.get("ΔBIC(2P−P)", np.nan), 3))
    b3.metric("Bootstrap Top_Frac", format_float(row.get("Bootstrap top_frac", np.nan), 3))
    b4.metric("Unique Winners", "—" if pd.isna(row.get("Bootstrap n_unique_winners", np.nan)) else str(int(row.get("Bootstrap n_unique_winners"))))
    b5.metric("Family Size", "—" if pd.isna(row.get("Bootstrap family_size", np.nan)) else str(int(row.get("Bootstrap family_size"))))

    st.markdown("#### Colors")
    c1, c2, c3 = st.columns(3)
    c1.metric("g − r", format_float(row.get("g - r", np.nan), 4))
    c2.metric("g − i", format_float(row.get("g - i", np.nan), 4))
    c3.metric("r − i", format_float(row.get("r - i", np.nan), 4))


# ==========================================================
# Population Tab (unchanged)
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
            ax.set_xlabel("Adopted Period (Hr)")
            ax.set_ylabel("Amplitude (Fourier, Mag)")
            ax.set_title("Period vs Amplitude (Filtered)")
            st.pyplot(fig, clear_figure=True)

    if "Adopted period (hr)" in df_f.columns:
        st.markdown("#### Adopted Period Distribution")
        periods = df_f["Adopted period (hr)"].to_numpy(float)
        periods = periods[np.isfinite(periods)]
        if len(periods) == 0:
            st.info("No finite adopted periods under current filters.")
        else:
            fig, ax = plt.subplots(figsize=(8.5, 4.0))
            ax.hist(periods, bins=50)
            ax.set_xlabel("Adopted Period (Hr)")
            ax.set_ylabel("Count")
            ax.set_title("Adopted Period Histogram")
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
