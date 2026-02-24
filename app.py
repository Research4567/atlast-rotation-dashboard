from __future__ import annotations

from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from google.cloud import bigquery
from google.oauth2 import service_account

from astropy.time import Time
from astroquery.jplhorizons import Horizons


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
MASTER_PATH = Path("master_results_clean.csv")  # required


# -------------------------
# BigQuery config
# -------------------------
BQ_PROJECT = "lsst-484623"
BQ_LOCATION = "US"  # change to "EU" if the dataset Details says EU
BQ_DATASET = "asteroid_institute__mpc_replica_views"  # <-- DOUBLE underscore
BQ_TABLE   = "public_obs_sbn_clustered"
BQ_STN     = "X05"
BQ_ROW_LIMIT = 20000

# BigQuery on-demand analysis pricing ballpark:
BQ_USD_PER_TB = 5.0  # USD / TB (10^12 bytes)


# -------------------------
# Horizons config
# -------------------------
HORIZONS_LOCATION = "X05"
HG_G_DEFAULT = 0.15


# -------------------------
# Helpers: cost formatting
# -------------------------
def bytes_to_human(n):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(n)
    for u in units:
        if x < 1000.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1000.0
    return f"{x:.2f} B"


def est_usd_cost(bytes_processed):
    return (float(bytes_processed) / 1e12) * float(BQ_USD_PER_TB)


# -------------------------
# BigQuery (Streamlit Cloud safe) — no decorator caching
# -------------------------
def get_bq_client():
    # Cache the client in session_state (works in all Streamlit versions)
    if "_bq_client" in st.session_state:
        return st.session_state["_bq_client"]

    if "gcp_service_account" not in st.secrets:
        st.error("Missing Streamlit secret: [gcp_service_account].")
        st.stop()

    sa = dict(st.secrets["gcp_service_account"])
    if ("client_email" not in sa) or ("private_key" not in sa):
        st.error("Your [gcp_service_account] secret is incomplete.")
        st.stop()

    creds = service_account.Credentials.from_service_account_info(sa)
    client = bigquery.Client(project=BQ_PROJECT, credentials=creds)

    st.session_state["_bq_client"] = client
    return client


def bq_load_photometry_for_provid(provid):
    client = get_bq_client()
    source_table = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    query = f"""
    SELECT
      provid,
      obstime,
      band,
      SAFE_CAST(mag AS FLOAT64)    AS mag,
      SAFE_CAST(rmsmag AS FLOAT64) AS rmsmag
    FROM `{source_table}`
    WHERE stn = @stn
      AND provid = @prov
      AND mag IS NOT NULL
    ORDER BY obstime
    LIMIT {int(BQ_ROW_LIMIT)}
    """

    params = [
        bigquery.ScalarQueryParameter("stn", "STRING", BQ_STN),
        bigquery.ScalarQueryParameter("prov", "STRING", provid),
    ]

    # ---- Dry run (estimate bytes processed) ----
    dry_config = bigquery.QueryJobConfig(
        query_parameters=params,
        dry_run=True,
        use_query_cache=False,
    )
    dry_job = client.query(query, job_config=dry_config)
    dry_bytes = int(getattr(dry_job, "total_bytes_processed", 0) or 0)

    bq_meta = {
        "provid": provid,
        "source_table": source_table,
        "dry_run_bytes_processed": dry_bytes,
        "dry_run_bytes_human": bytes_to_human(dry_bytes) if dry_bytes else "—",
        "dry_run_est_cost_usd": est_usd_cost(dry_bytes) if dry_bytes else None,
        "note": "BigQuery charges by bytes processed. USD estimate uses ~$5/TB (10^12 bytes).",
    }

    # ---- Actual run ----
    run_config = bigquery.QueryJobConfig(
        query_parameters=params,
        use_query_cache=True,
    )
    job = client.query(query, job_config=run_config)
    df = job.to_dataframe()

    actual_bytes = int(getattr(job, "total_bytes_processed", 0) or 0)
    bq_meta.update({
        "actual_bytes_processed": actual_bytes if actual_bytes else None,
        "actual_bytes_human": bytes_to_human(actual_bytes) if actual_bytes else "—",
        "actual_est_cost_usd": est_usd_cost(actual_bytes) if actual_bytes else None,
        "cache_hit": bool(getattr(job, "cache_hit", False)),
    })

    return df, bq_meta


# -------------------------
# Band normalization
# -------------------------
LSST_CANON = {"u", "g", "r", "i", "z", "y"}

def normalize_lsst_band(x):
    if x is None:
        return ""
    s = str(x).strip().lower()

    # common in some exports: 'lg','lr','li','lu','lz','ly' -> 'g','r','i','u','z','y'
    if len(s) == 2 and s[0] == "l" and s[1] in LSST_CANON:
        return s[1]

    m = re.match(r"^(?:lsst)?([ugrizy])$", s)
    if m:
        return m.group(1)

    return s


# ======================================================================
# STEP 5 — Geometry correction using JPL Horizons (range query)
# ======================================================================
def step5_geometry_horizons_range(
    df1,
    *,
    PROVID,
    OUTDIR,
    HORIZONS_LOCATION="X05",
    G_DEFAULT=0.15,
    STEP_MINUTES=1,
    PAD_MINUTES=5,
    FAIL_ON_UNMATCHED=True,
    TOL_DAYS=None,
    show_plots=False,
    save_plots=False,
    save_tables=False,
    verbose=False,
):
    if df1 is None or len(df1) == 0:
        raise ValueError("STEP 5: df1 is empty.")

    need = ["obstime_dt", "mag", "band"]
    missing = [c for c in need if c not in df1.columns]
    if missing:
        raise ValueError(f"STEP 5: df1 missing columns required: {missing}")

    if TOL_DAYS is None:
        TOL_DAYS = max(2e-3, (STEP_MINUTES / 1440.0) * 1.2)

    if save_tables or save_plots:
        os.makedirs(OUTDIR, exist_ok=True)

    dfG = df1.copy()
    dfG["band"] = dfG["band"].map(normalize_lsst_band)

    dfG["obstime_dt"] = pd.to_datetime(dfG["obstime_dt"], errors="coerce", utc=True)
    dfG = dfG.dropna(subset=["obstime_dt"]).sort_values("obstime_dt").reset_index(drop=True)
    if len(dfG) == 0:
        raise ValueError("STEP 5: all obstime_dt are NaT after coercion.")

    if "t_hr" not in dfG.columns:
        t0 = dfG["obstime_dt"].min()
        dfG["t_hr"] = (dfG["obstime_dt"] - t0).dt.total_seconds() / 3600.0

    if ("night_id" not in dfG.columns) and ("night_utc" not in dfG.columns):
        dfG["night_utc"] = dfG["obstime_dt"].dt.strftime("%Y-%m-%d")
    night_key = "night_id" if "night_id" in dfG.columns else "night_utc"

    t_utc = Time(dfG["obstime_dt"].dt.to_pydatetime(), scale="utc")
    dfG["jd_utc_obs"] = t_utc.jd.astype(float)

    def query_horizons_range_smallbody(desig, start_utc, stop_utc, step_minutes=1, location="X05"):
        obj = Horizons(
            id=desig,
            id_type="smallbody",
            location=location,
            epochs={"start": start_utc, "stop": stop_utc, "step": f"{int(step_minutes)}m"},
        )
        eph = obj.ephemerides()
        df = eph.to_pandas()

        for k in ["datetime_jd", "r", "delta", "alpha", "lighttime"]:
            if k not in df.columns:
                raise KeyError(f"Horizons response missing '{k}'. Columns={list(df.columns)}")

        return pd.DataFrame({
            "jd_utc_eph": df["datetime_jd"].astype(float).to_numpy(),
            "r_au": df["r"].astype(float).to_numpy(),
            "delta_au": df["delta"].astype(float).to_numpy(),
            "alpha_deg": df["alpha"].astype(float).to_numpy(),
            "lighttime_min": df["lighttime"].astype(float).to_numpy(),
        })

    eph_parts = []
    blocks = sorted(dfG[night_key].dropna().unique())

    for block in blocks:
        sub = dfG[dfG[night_key] == block]
        tmin = pd.to_datetime(sub["obstime_dt"].min(), utc=True) - pd.Timedelta(minutes=PAD_MINUTES)
        tmax = pd.to_datetime(sub["obstime_dt"].max(), utc=True) + pd.Timedelta(minutes=PAD_MINUTES)

        start_utc = tmin.strftime("%Y-%m-%d %H:%M")
        stop_utc  = tmax.strftime("%Y-%m-%d %H:%M")

        eph_b = query_horizons_range_smallbody(
            PROVID, start_utc, stop_utc,
            step_minutes=STEP_MINUTES,
            location=HORIZONS_LOCATION,
        )
        eph_parts.append(eph_b)

    eph_df = (
        pd.concat(eph_parts, ignore_index=True)
          .drop_duplicates(subset=["jd_utc_eph"])
          .sort_values("jd_utc_eph")
          .reset_index(drop=True)
    )

    obs = dfG.sort_values("jd_utc_obs").reset_index(drop=True)
    eph = eph_df.sort_values("jd_utc_eph").reset_index(drop=True)

    dfM = pd.merge_asof(
        obs, eph,
        left_on="jd_utc_obs",
        right_on="jd_utc_eph",
        direction="nearest",
        tolerance=TOL_DAYS,
    )

    dfM["dt_match_sec"] = (dfM["jd_utc_obs"] - dfM["jd_utc_eph"]) * 86400.0

    matched = int(dfM["r_au"].notna().sum())
    n_total = int(len(dfM))
    n_unmatched = n_total - matched

    if n_unmatched > 0 and FAIL_ON_UNMATCHED:
        raise RuntimeError("Unmatched ephemeris rows remain after merge_asof.")

    dfM["lighttime_days"] = dfM["lighttime_min"] / 1440.0
    dfM["jd_utc_emit"] = dfM["jd_utc_obs"] - dfM["lighttime_days"]
    t0_emit = float(np.nanmin(dfM["jd_utc_emit"].to_numpy(float)))
    dfM["t_emit_hr"] = (dfM["jd_utc_emit"] - t0_emit) * 24.0

    def phi1(alpha_rad):
        return np.exp(-3.33 * np.power(np.tan(alpha_rad / 2.0), 0.63))

    def phi2(alpha_rad):
        return np.exp(-1.87 * np.power(np.tan(alpha_rad / 2.0), 1.22))

    def phase_HG(alpha_deg, G=0.15):
        a = np.deg2rad(alpha_deg)
        p = (1.0 - G) * phi1(a) + G * phi2(a)
        p = np.clip(p, 1e-12, None)
        return -2.5 * np.log10(p)

    r = pd.to_numeric(dfM["r_au"], errors="coerce").to_numpy(float)
    d = pd.to_numeric(dfM["delta_au"], errors="coerce").to_numpy(float)
    alpha = pd.to_numeric(dfM["alpha_deg"], errors="coerce").to_numpy(float)

    dfM["dist_term"] = 5.0 * np.log10(r * d)
    dfM["phase_term"] = phase_HG(alpha, G=G_DEFAULT)
    dfM["mag_geo"] = pd.to_numeric(dfM["mag"], errors="coerce") - dfM["dist_term"] - dfM["phase_term"]

    ok = np.isfinite(dfM["mag_geo"].to_numpy(float))
    dfM["mag_geo_bandcenter"] = np.nan
    if ok.any():
        dfM.loc[ok, "mag_geo_bandcenter"] = (
            dfM.loc[ok, "mag_geo"] - dfM.loc[ok].groupby("band")["mag_geo"].transform("median")
        )

    step5_meta = {
        "HORIZONS_LOCATION": str(HORIZONS_LOCATION),
        "G_DEFAULT": float(G_DEFAULT),
        "STEP_MINUTES": int(STEP_MINUTES),
        "PAD_MINUTES": int(PAD_MINUTES),
        "TOL_DAYS": float(TOL_DAYS),
        "night_key": str(night_key),
        "n_blocks": int(len(blocks)),
        "n_obs": int(n_total),
        "n_matched": int(matched),
        "n_unmatched": int(n_unmatched),
    }

    return dfM, step5_meta


# -------------------------
# More helpers
# -------------------------
def load_master(path):
    return pd.read_csv(path)

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def format_float(x, nd=6):
    try:
        v = float(x)
        if np.isfinite(v):
            return f"{v:.{nd}f}"
    except Exception:
        pass
    return "—"

def reliability_short(rel):
    r = (rel or "").strip().lower()
    return r if r in {"reliable", "ambiguous", "insufficient"} else "unknown"

def reliability_html(rel):
    r = reliability_short(rel)
    if r == "reliable":
        return '<span style="color:#22c55e;font-weight:800;">Reliable</span>'
    if r == "ambiguous":
        return '<span style="color:#f59e0b;font-weight:800;">Ambiguous</span>'
    if r == "insufficient":
        return '<span style="color:#ef4444;font-weight:800;">Insufficient</span>'
    return '<span style="color:#64748b;font-weight:800;">Unknown</span>'

def norm_id(x):
    if x is None:
        return ""
    s = str(x).strip().lower()
    for ch in [" ", "_", "-", "\t", "\n", "\r"]:
        s = s.replace(ch, "")
    return s

def resolve_nights(df):
    for c in ["night", "night_id", "night_utc"]:
        if c in df.columns:
            s = df[c].astype(str)
            if s.notna().sum() >= 3:
                return int(s.nunique())
    if "obstime_dt" in df.columns:
        dt = pd.to_datetime(df["obstime_dt"], errors="coerce", utc=True)
        if dt.notna().sum() >= 3:
            return int(dt.dt.date.nunique())
    return None

def plot_fold(ax, t_hr, mag, bands, P_hr, title, mag_label, two_cycles=False):
    phase = (t_hr / float(P_hr)) % 1.0
    uniq = sorted(np.unique(bands).tolist())
    for b in uniq:
        m = (bands == b)
        ax.scatter(phase[m], mag[m], s=10, label=str(b))
        if two_cycles:
            ax.scatter(phase[m] + 1.0, mag[m], s=10)
    ax.invert_yaxis()
    ax.set_xlabel("Phase (0–1)" if not two_cycles else "Phase (0–2)")
    ax.set_ylabel(mag_label)
    ax.set_title(title)
    ax.set_xlim(0.0, 2.0 if two_cycles else 1.0)


def make_df1_from_bq(df_raw):
    df = df_raw.copy()
    df["obstime_dt"] = pd.to_datetime(df["obstime"], errors="coerce", utc=True)
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["rmsmag"] = pd.to_numeric(df.get("rmsmag", np.nan), errors="coerce")
    df["band"] = df.get("band", "x").map(normalize_lsst_band)

    df = df.dropna(subset=["obstime_dt", "mag", "band"]).sort_values("obstime_dt").reset_index(drop=True)
    if len(df) == 0:
        return df

    t0 = df["obstime_dt"].min()
    df["t_hr"] = (df["obstime_dt"] - t0).dt.total_seconds() / 3600.0
    df["night_utc"] = df["obstime_dt"].dt.strftime("%Y-%m-%d")
    return df


def geo_correct_cached(df1, provid):
    df_geo, meta = step5_geometry_horizons_range(
        df1,
        PROVID=provid,
        OUTDIR=".",
        HORIZONS_LOCATION=HORIZONS_LOCATION,
        G_DEFAULT=HG_G_DEFAULT,
        STEP_MINUTES=10,   # DEMO
        PAD_MINUTES=10,    # DEMO
        FAIL_ON_UNMATCHED=False,
        save_tables=False,
        save_plots=False,
        show_plots=False,
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

# If your master has a different column name, rename to "Designation"
if "Designation" not in master.columns:
    for c in ["provid", "PROVID", "designation", "name", "object_id"]:
        if c in master.columns:
            master = master.rename(columns={c: "Designation"})
            break

# Convert numeric columns if present (optional)
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

# Reliable count label
if "Reliability" in master.columns:
    _rel_short_all = master["Reliability"].astype(str).map(reliability_short)
    RELIABLE_COUNT = int((_rel_short_all == "reliable").sum())
else:
    RELIABLE_COUNT = 0


# -------------------------
# Sidebar + layout
# -------------------------
st.sidebar.markdown("## Mode")
mode = st.sidebar.radio("View", ["Asteroid Viewer", "Population Explorer"], index=0)

st.markdown("## ATLAST Asteroid Rotation Dashboard")


# ==========================================================
# MODE 1: ASTEROID VIEWER
# ==========================================================
if mode == "Asteroid Viewer":
    st.caption("Photometry is queried from BigQuery per asteroid and folded using on-the-fly geometry correction (Horizons).")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Asteroid")

    if "reliable_only" not in st.session_state:
        st.session_state["reliable_only"] = True

    q = st.sidebar.text_input("Search Designation", value="", placeholder="E.g., 2025 ME69")

    reliable_only_state = bool(st.session_state.get("reliable_only", False))
    df_pick = master.copy()

    if q.strip():
        df_pick = df_pick[df_pick["Designation"].astype(str).str.contains(q.strip(), case=False, na=False)]

    if reliable_only_state and ("Reliability" in df_pick.columns):
        rel_s = df_pick["Reliability"].astype(str).map(reliability_short)
        df_pick = df_pick[rel_s == "reliable"]

    df_pick = df_pick.sort_values("Designation")
    designations = df_pick["Designation"].astype(str).tolist()

    if not designations:
        st.sidebar.warning("No asteroids match your current search." if not reliable_only_state
                           else "No reliable-period asteroids match your current search.")
        st.stop()

    selected = st.sidebar.selectbox(
        "Selected Asteroid",
        options=designations,
        index=0,
        key="selected_asteroid",
    )

    st.sidebar.checkbox(
        f"Reliable Periods only ({RELIABLE_COUNT:,})",
        value=reliable_only_state,
        key="reliable_only",
    )

    if bool(st.session_state.get("reliable_only", False)) and (selected not in designations):
        st.session_state.selected_asteroid = designations[0]
        st.rerun()

    row = master[master["Designation"].astype(str) == str(selected)]
    row = row.iloc[0].to_dict() if len(row) else {}
    rel = reliability_short(str(row.get("Reliability", "")))

    P_adopt = float(row.get("Adopted period (hr)", np.nan))
    if not (np.isfinite(P_adopt) and P_adopt > 0):
        P_adopt = 5.0

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

    LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
    sel_bands_sidebar = st.sidebar.multiselect(
        "Bands",
        options=LSST_BANDS,
        default=["g", "r", "i"],
    )

    two_cycles = st.sidebar.checkbox("Show two cycles (0–2)", value=False)

    tab_photo, tab_char = st.tabs(["Photometry", "Characterisation"])

    with tab_photo:
        st.markdown(
            f"### Geometry-Corrected Fold Preview: **{selected}** &nbsp;&nbsp;•&nbsp;&nbsp; {reliability_html(rel)}",
            unsafe_allow_html=True,
        )

        n_obs_master = row.get("Number of Observations", np.nan)
        arc_days = row.get("Arc (days)", np.nan)

        with st.spinner("Querying BigQuery photometry (minimal columns) ..."):
            df_raw, bq_meta = bq_load_photometry_for_provid(str(selected))

        with st.expander("BigQuery Cost & Query Diagnostics", expanded=False):
            st.write("If bytes are huge (GB–TB), the query is scanning too much.")
            st.json(bq_meta)

        if df_raw is None or len(df_raw) == 0:
            st.info("No photometry rows found in BigQuery for this asteroid (stn filter + provid match).")
            st.stop()

        df1 = make_df1_from_bq(df_raw)
        if len(df1) < 5:
            st.warning("Very few usable points after cleaning.")
            st.dataframe(df1.head(50), use_container_width=True)
            st.stop()

        with st.spinner("Running geometry correction (Horizons) ..."):
            try:
                df_geo, meta5 = geo_correct_cached(df1, str(selected))
            except Exception as e:
                st.error("Geometry correction failed (Horizons). Showing raw mags instead.")
                st.exception(e)
                df_geo = df1.copy()
                df_geo["mag_geo"] = np.nan
                df_geo["mag_geo_bandcenter"] = np.nan
                meta5 = {}

        if "mag_geo_bandcenter" in df_geo.columns and df_geo["mag_geo_bandcenter"].notna().sum() >= 5:
            mag_col = "mag_geo_bandcenter"
            mag_label = "mag_geo_bandcenter (corrected, band-centered)"
        elif "mag_geo" in df_geo.columns and df_geo["mag_geo"].notna().sum() >= 5:
            mag_col = "mag_geo"
            mag_label = "mag_geo (corrected)"
        else:
            mag_col = "mag"
            mag_label = "mag (raw)"

        df_geo["band"] = df_geo["band"].map(normalize_lsst_band)

        avail = set(df_geo["band"].dropna().astype(str).unique().tolist())
        sel_bands = [b for b in sel_bands_sidebar if b in avail]
        if not sel_bands:
            sel_bands = sorted(list(avail))

        dfp = df_geo[df_geo["band"].isin(sel_bands)].copy()
        dfp = dfp.dropna(subset=["t_hr", mag_col, "band"])
        n_nights = resolve_nights(dfp)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Adopted Rotation Period (hours)", format_float(row.get("Adopted period (hr)", np.nan), 6))
        s2.metric("Fold Period (hours)", format_float(P_calc, 6))
        s3.metric("Observations (number)", "—" if pd.isna(n_obs_master) else str(int(n_obs_master)))
        s4.metric("Nights (photometry)", "—" if n_nights is None else str(int(n_nights)))

        if np.isfinite(float(arc_days)):
            st.caption(f"Arc Length (days): {format_float(arc_days, 3)}")

        st.caption(f"Folding uses: **{mag_label}**")

        st.download_button(
            "Download Photometry CSV",
            data=df_geo.to_csv(index=False).encode("utf-8"),
            file_name=f"{norm_id(selected)}_photometry_geo.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Geometry Correction QA (Step 5 meta)", expanded=False):
            st.json(meta5)

        if len(dfp) < 5:
            st.warning("Very few points remain after band filtering. Try selecting more bands.")
            st.stop()

        t_hr = dfp["t_hr"].to_numpy(float)
        mag = pd.to_numeric(dfp[mag_col], errors="coerce").to_numpy(float)
        bands = dfp["band"].to_numpy(str)

        P_half = 0.5 * float(P_calc)
        P_two = 2.0 * float(P_calc)

        st.markdown("#### Three-Panel Fold (P/2 • P • 2P)")
        cols = st.columns(3)
        periods = [P_half, float(P_calc), P_two]
        titles = [f"P/2 = {P_half:.6f} Hr", f"P = {float(P_calc):.6f} Hr", f"2P = {P_two:.6f} Hr"]

        for col, P_hr, title in zip(cols, periods, titles):
            with col:
                fig, ax = plt.subplots(figsize=(5.2, 3.6))
                plot_fold(
                    ax,
                    t_hr=t_hr,
                    mag=mag,
                    bands=bands,
                    P_hr=P_hr,
                    title=title,
                    mag_label=mag_label,
                    two_cycles=two_cycles,
                )
                ax.legend(fontsize=7)
                st.pyplot(fig, clear_figure=True)

        st.markdown("#### Magnitude vs Time")
        fig, ax = plt.subplots(figsize=(10.5, 3.6))
        for b in sorted(np.unique(bands).tolist()):
            m = (bands == b)
            ax.scatter(t_hr[m], mag[m], s=10, label=b)
        ax.invert_yaxis()
        ax.set_xlabel("Hours Since First Observation")
        ax.set_ylabel(mag_label)
        ax.set_title("Magnitude vs Time")
        ax.legend(fontsize=8, ncol=6)
        st.pyplot(fig, clear_figure=True)

    with tab_char:
        st.markdown(
            f"### Characterisation: **{selected}** &nbsp;&nbsp;•&nbsp;&nbsp; {reliability_html(rel)}",
            unsafe_allow_html=True,
        )
        st.caption("All values on this tab come from master_results_clean.csv (Step 13 Summary Exports).")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Adopted Rotation Period (hours)", format_float(row.get("Adopted period (hr)", np.nan), 6))
        k2.metric("Adopted K", "—" if pd.isna(row.get("Adopted K", np.nan)) else str(int(row.get("Adopted K"))))
        k3.metric("Amplitude (Mag)", format_float(row.get("Amplitude (Fourier)", np.nan), 3))
        k4.metric("Axial Elongation", format_float(row.get("Axial Elongation", np.nan), 3))

        b1, b2, b3 = st.columns(3)
        b1.metric("2P Candidate (Hr)", format_float(row.get("2P candidate (hr)", np.nan), 6))
        b2.metric("ΔBIC(2P−P)", format_float(row.get("ΔBIC(2P−P)", np.nan), 3))
        b3.metric("Bootstrap Top_Frac", format_float(row.get("Bootstrap top_frac", np.nan), 3))

        st.markdown("#### Color Indices")
        c1, c2, c3 = st.columns(3)
        c1.metric("g − r", format_float(row.get("g - r", np.nan), 4))
        c2.metric("g − i", format_float(row.get("g - i", np.nan), 4))
        c3.metric("r − i", format_float(row.get("r - i", np.nan), 4))


# ==========================================================
# MODE 2: POPULATION EXPLORER
# ==========================================================
else:
    st.caption("Explore the population distribution using filters in the sidebar.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Population Filters")

    rel_series = master.get("Reliability", pd.Series([], dtype=str)).dropna().astype(str)
    rel_options = sorted(rel_series.unique().tolist()) if len(rel_series) else ["reliable", "ambiguous", "insufficient", "unknown"]

    default_rels = ["reliable"] if "reliable" in rel_options else rel_options
    selected_rels = st.sidebar.multiselect("Reliability", options=rel_options, default=default_rels)
    if not selected_rels:
        selected_rels = default_rels

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





