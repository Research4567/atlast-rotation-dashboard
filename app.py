# app.py — Iteration 3
# ==========================================================
# ATLAST Asteroid Rotation Dashboard
# MASTER_rotation_summary_v2026-03-10.csv
# (76 asteroids, pipeline v54, 2025 cohort)
#
# Iteration 3:
#   Tab 1 "Characterisation": key values, fold plots, period
#         candidates, 2P decision, colour indices
#   Tab 2 "Evidence": raw lightcurve, Lomb-Scargle periodogram,
#         WRMS scan, bootstrap, residuals, BIC table, pipeline
#         diagnostic images (loaded from GCS when available)
#   Light theme, slider only, no +/- box
# ==========================================================

from __future__ import annotations

from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import Forbidden, BadRequest, NotFound

from astropy.time import Time
from astropy.timeseries import LombScargle
from astroquery.jplhorizons import Horizons


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="ATLAST Asteroid Rotation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Light-theme CSS — clean, scientific, readable
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', 'Source Sans Pro', sans-serif;
}
code, pre, .stCode, [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
}

/* Metric cards — light */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 14px 8px 14px;
}
[data-testid="stMetricLabel"] {
    font-size: 0.76rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
}

/* Badges */
.badge-row {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 4px;
}
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 5px;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.02em;
}
.badge-reliable     { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.badge-ambiguous    { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
.badge-insufficient { background: #fecaca; color: #991b1b; border: 1px solid #fca5a5; }
.badge-unknown      { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
.badge-2p           { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
.badge-review       { background: #fecaca; color: #991b1b; border: 1px solid #fca5a5; }

.section-rule { border: none; border-top: 1px solid #e2e8f0; margin: 1rem 0; }

/* Sidebar */
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 0.88rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: #64748b; margin-top: 0.5rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px; font-weight: 600;
}

/* Subheader style for sections */
.section-head {
    font-size: 1.05rem; font-weight: 700; color: #334155;
    margin: 0.8rem 0 0.3rem 0; letter-spacing: -0.01em;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# Data file
# -------------------------
MASTER_PATH = Path("MASTER_rotation_summary_v2026-03-10.csv")


# -------------------------
# BigQuery config
# -------------------------
BQ_PROJECT  = "lsst-484623"
BQ_LOCATION = "US"
BQ_DATASET  = "atlast_photometry"
BQ_TABLE    = "public_obs_x05"

BQ_DEFAULT_ROW_LIMIT = 20000
BQ_MAX_ROW_LIMIT     = 200000
BQ_USD_PER_TB        = 5.0

HORIZONS_LOCATION = "X05"
HG_G_DEFAULT      = 0.15


# ============================================================
# Band palette — consistent across all plots
# ============================================================
BAND_COLORS = {
    "u": "#7c3aed", "g": "#059669", "r": "#ea580c",
    "i": "#dc2626", "z": "#7c3aed", "y": "#d97706",
}
BAND_ORDER = ["u", "g", "r", "i", "z", "y"]

def band_color(b): return BAND_COLORS.get(b, "#64748b")


# ============================================================
# Matplotlib light style
# ============================================================
def setup_mpl_style():
    mpl.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#fafbfc",
        "axes.edgecolor":    "#cbd5e1",
        "axes.labelcolor":   "#334155",
        "axes.titlesize":    11,
        "axes.titleweight":  600,
        "axes.titlecolor":   "#1e293b",
        "axes.grid":         True,
        "grid.color":        "#e2e8f0",
        "grid.linewidth":    0.6,
        "grid.alpha":        0.8,
        "xtick.color":       "#64748b",
        "ytick.color":       "#64748b",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "text.color":        "#1e293b",
        "legend.facecolor":  "white",
        "legend.edgecolor":  "#e2e8f0",
        "legend.fontsize":   8.5,
        "legend.labelcolor": "#334155",
        "savefig.facecolor": "white",
        "figure.dpi":        110,
    })

setup_mpl_style()


# ============================================================
# Column name constants
# ============================================================
C_DESIG       = "Designation"
C_PERIOD      = "Period"
C_ARC         = "Arc"
C_NOBS        = "Number of Observations"
C_HMAG        = "H Mag"
C_AMPLITUDE   = "Amplitude"
C_RELIABILITY = "Reliability"
C_ADDITIONAL  = "Additional periods (hr)"
C_ADD_DBIC    = "Additional periods dBIC"
C_ADD_N       = "Additional periods n"
C_PREFER_2P   = "adopt_prefer_2P"
C_PREFER_2P_R = "adopt_prefer_2P_reason"
C_DBIC_2P     = "adopt_delta_BIC_2P_vs_P"
C_OE_RATIO    = "adopt_oe_ratio"
C_AMP_RATIO   = "adopt_amp_ratio_2p"
C_BOOT_BASE   = "adopt_boot_frac_base"
C_BOOT_2P     = "adopt_boot_frac_2P"
C_MORPH_FORCE = "adopt_morph_force_2p"
C_BOOT_VETO   = "adopt_boot_veto_2p"
C_REVIEW      = "needs_human_review_2p"
C_BOOT_HW_FLAG= "Boot harmonic winner flag"
C_BOOT_HW_P   = "Boot harmonic winner P hr"
C_GR          = "g - r"
C_GI          = "g - i"
C_RI          = "r - i"
C_AXIAL       = "Axial Elongation"
C_MEAN_MAG    = "Mean Mag"
C_AMBIG       = "Step11 ambiguous_flag"
C_FAM_ID      = "Step11 family id"
C_FAM_SRC     = "Step11 family source"
C_FAM_BOOT    = "Step11 family bootstrap frac"
C_STRONG_BOOT = "Step11 strong bootstrap override"


# ============================================================
# GCS config for pipeline diagnostic images
# ============================================================
GCS_BUCKET     = "atlast-pipeline-outputs"
GCS_PREFIX     = "ls_outputs"
PIPELINE_TAG   = "v2026-03-10"


# -------------------------
# Helpers
# -------------------------
def bytes_to_human(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1000.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1000.0

def est_usd_cost(b): return (float(b) / 1e12) * BQ_USD_PER_TB
def safe_num(s):      return pd.to_numeric(s, errors="coerce")

def format_float(x, nd=6):
    try:
        v = float(x)
        if np.isfinite(v):
            return f"{v:.{nd}f}"
    except Exception: pass
    return "—"

def _safe_period(val):
    try:
        v = float(val)
        if np.isfinite(v) and v > 0: return round(v, 6)
    except Exception: pass
    return None

def reliability_short(rel):
    r = (rel or "").strip().lower()
    return r if r in {"reliable", "ambiguous", "insufficient"} else "unknown"

def reliability_badge(rel):
    r = reliability_short(rel)
    return f'<span class="badge badge-{r}">{r.capitalize()}</span>'


# ============================================================
# Parse pipe-separated additional periods
# ============================================================
def parse_pipe_list(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    result = []
    for p in str(val).split("|"):
        try:
            v = float(p.strip())
            if np.isfinite(v) and v > 0:
                result.append(round(v, 6))
        except Exception: pass
    return result


def build_period_candidates(row):
    candidates, seen = [], []
    P_adopt = _safe_period(row.get(C_PERIOD)) or 0.0

    def _is_dup(p):
        return any(abs(p - s) / max(s, 1e-9) < 0.005 for s in seen)

    def _is_harmonic(p):
        if P_adopt <= 0: return False
        ratio = p / P_adopt
        return any(abs(ratio - h) / h < 0.02 for h in [0.5, 1.0, 2.0])

    def _add(label, period, note=None, is_adopted=False):
        if period is None or _is_dup(period): return
        seen.append(period)
        candidates.append({"label": label, "period": period, "note": note, "is_adopted": is_adopted})

    _add("Adopted", _safe_period(row.get(C_PERIOD)), is_adopted=True)

    add_periods = parse_pipe_list(row.get(C_ADDITIONAL))
    add_dbics   = parse_pipe_list(row.get(C_ADD_DBIC))
    alt_idx = 1
    for i, p in enumerate(add_periods):
        if _is_harmonic(p): continue
        note = f"ΔBIC = {add_dbics[i]:.2f}" if i < len(add_dbics) else None
        _add(f"Alt {alt_idx}", p, note=note)
        alt_idx += 1

    if row.get(C_BOOT_HW_FLAG):
        p_hw = _safe_period(row.get(C_BOOT_HW_P))
        if p_hw and not _is_harmonic(p_hw):
            _add("Boot harmonic", p_hw, note="bootstrap winner")

    return candidates


# ============================================================
# Load master CSV
# ============================================================
def load_master(path):
    df = pd.read_csv(path)
    num_cols = [C_PERIOD, C_ARC, C_NOBS, C_HMAG, C_AMPLITUDE,
                C_DBIC_2P, C_OE_RATIO, C_AMP_RATIO,
                C_BOOT_BASE, C_BOOT_2P, C_GR, C_GI, C_RI,
                C_AXIAL, C_MEAN_MAG, C_FAM_BOOT]
    for c in num_cols:
        if c in df.columns:
            df[c] = safe_num(df[c])
    return df


def resolve_nights(df):
    for c in ["night", "night_id", "night_utc"]:
        if c in df.columns:
            s = df[c].astype(str)
            if s.notna().sum() >= 3: return int(s.nunique())
    if "obstime_dt" in df.columns:
        dt = pd.to_datetime(df["obstime_dt"], errors="coerce", utc=True)
        if dt.notna().sum() >= 3: return int(dt.dt.date.nunique())
    return None


# ============================================================
# BigQuery helpers
# ============================================================
LSST_CANON = {"u", "g", "r", "i", "z", "y"}

def normalize_lsst_band(x):
    if x is None: return ""
    s = str(x).strip().lower()
    if len(s) == 2 and s[0] == "l" and s[1] in LSST_CANON: return s[1]
    m = re.match(r"^(?:lsst)?([ugrizy])$", s)
    return m.group(1) if m else s


def get_bq_client():
    if "_bq_client" in st.session_state:
        return st.session_state["_bq_client"]
    if "gcp_service_account" not in st.secrets:
        st.error("Missing Streamlit secret: [gcp_service_account].")
        st.stop()
    sa = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(sa)
    client = bigquery.Client(project=BQ_PROJECT, credentials=creds)
    try:
        client.query("SELECT 1", location=BQ_LOCATION).result()
    except Exception as e:
        st.error("BigQuery smoke-test failed.")
        st.exception(e)
        st.stop()
    st.session_state["_bq_client"] = client
    return client


def bq_load_photometry(provid, *, row_limit):
    client = get_bq_client()
    source = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    row_limit = int(max(1, min(row_limit, BQ_MAX_ROW_LIMIT)))
    query = f"""
    SELECT provid, obstime, band,
           SAFE_CAST(mag AS FLOAT64) AS mag,
           SAFE_CAST(rmsmag AS FLOAT64) AS rmsmag
    FROM `{source}`
    WHERE provid = @prov AND mag IS NOT NULL
    ORDER BY obstime
    LIMIT {row_limit}
    """
    params = [bigquery.ScalarQueryParameter("prov", "STRING", provid)]
    bq_meta = {"provid": provid, "source_table": source, "row_limit": row_limit}

    try:
        dry = client.query(query, location=BQ_LOCATION,
                           job_config=bigquery.QueryJobConfig(
                               query_parameters=params, dry_run=True, use_query_cache=False))
        est = int(getattr(dry, "total_bytes_processed", 0) or 0)
        bq_meta.update({"dry_run_ok": True,
                         "estimated_bytes_human": bytes_to_human(est),
                         "estimated_cost_usd": round(est_usd_cost(est), 6)})
    except Exception as e:
        st.error("BigQuery dry-run failed."); st.exception(e); raise

    job = client.query(query, location=BQ_LOCATION,
                       job_config=bigquery.QueryJobConfig(
                           query_parameters=params, use_query_cache=True))
    try:
        df = job.to_dataframe(create_bqstorage_client=False)
    except (Forbidden, BadRequest, NotFound) as e:
        st.error("BigQuery query failed."); st.exception(e); raise

    actual = int(getattr(job, "total_bytes_processed", 0) or 0)
    bq_meta.update({"actual_bytes_human": bytes_to_human(actual),
                     "actual_cost_usd": round(est_usd_cost(actual), 6),
                     "cache_hit": bool(getattr(job, "cache_hit", False)),
                     "job_id": getattr(job, "job_id", None),
                     "returned_rows": len(df),
                     "may_be_truncated": len(df) >= row_limit})
    return df, bq_meta


@st.cache_data(ttl=3600, show_spinner=False)
def bq_fetch_cached(provid: str, row_limit: int):
    return bq_load_photometry(provid, row_limit=row_limit)


def make_df1(df_raw):
    df = df_raw.copy()
    df["obstime_dt"] = pd.to_datetime(df["obstime"], errors="coerce", utc=True)
    df["mag"]  = pd.to_numeric(df["mag"],  errors="coerce")
    df["band"] = df.get("band", "x").map(normalize_lsst_band)
    df = df.dropna(subset=["obstime_dt", "mag", "band"]).reset_index(drop=True)
    if len(df) == 0: return df
    df = df.sort_values("obstime_dt")
    t0 = df["obstime_dt"].min()
    df["t_hr"]     = (df["obstime_dt"] - t0).dt.total_seconds() / 3600.0
    df["t_day"]    = (df["obstime_dt"] - t0).dt.total_seconds() / 86400.0
    df["night_utc"]= df["obstime_dt"].dt.strftime("%Y-%m-%d")
    return df


# ============================================================
# Geometry correction (Step 5 / Horizons)
# ============================================================
def geo_correct_full(df1, provid):
    if df1 is None or len(df1) == 0:
        raise ValueError("df1 is empty.")

    dfG = df1.copy()
    dfG["band"] = dfG["band"].map(normalize_lsst_band)
    dfG["obstime_dt"] = pd.to_datetime(dfG["obstime_dt"], errors="coerce", utc=True)
    dfG = dfG.dropna(subset=["obstime_dt"]).sort_values("obstime_dt").reset_index(drop=True)
    if len(dfG) == 0: raise ValueError("All obstime_dt NaT.")

    if "t_hr" not in dfG.columns:
        t0 = dfG["obstime_dt"].min()
        dfG["t_hr"] = (dfG["obstime_dt"] - t0).dt.total_seconds() / 3600.0
    if "night_utc" not in dfG.columns:
        dfG["night_utc"] = dfG["obstime_dt"].dt.strftime("%Y-%m-%d")
    night_key = "night_id" if "night_id" in dfG.columns else "night_utc"

    t_utc = Time(dfG["obstime_dt"].dt.to_pydatetime(), scale="utc")
    dfG["jd_utc_obs"] = t_utc.jd.astype(float)

    TOL_DAYS = max(2e-3, (10 / 1440.0) * 1.2)

    def _query_horizons(desig, start_utc, stop_utc, step_min, loc):
        obj = Horizons(id=desig, id_type="smallbody", location=loc,
                       epochs={"start": start_utc, "stop": stop_utc, "step": f"{int(step_min)}m"})
        eph = obj.ephemerides().to_pandas()
        return pd.DataFrame({
            "jd_utc_eph":    eph["datetime_jd"].astype(float).to_numpy(),
            "r_au":          eph["r"].astype(float).to_numpy(),
            "delta_au":      eph["delta"].astype(float).to_numpy(),
            "alpha_deg":     eph["alpha"].astype(float).to_numpy(),
            "lighttime_min": eph["lighttime"].astype(float).to_numpy(),
        })

    eph_parts = []
    for block in sorted(dfG[night_key].dropna().unique()):
        sub  = dfG[dfG[night_key] == block]
        tmin = pd.to_datetime(sub["obstime_dt"].min(), utc=True) - pd.Timedelta(minutes=10)
        tmax = pd.to_datetime(sub["obstime_dt"].max(), utc=True) + pd.Timedelta(minutes=10)
        eph_parts.append(_query_horizons(provid,
                                         tmin.strftime("%Y-%m-%d %H:%M"),
                                         tmax.strftime("%Y-%m-%d %H:%M"),
                                         10, HORIZONS_LOCATION))

    eph_df = (pd.concat(eph_parts, ignore_index=True)
                .drop_duplicates("jd_utc_eph")
                .sort_values("jd_utc_eph").reset_index(drop=True))

    dfM = pd.merge_asof(dfG.sort_values("jd_utc_obs"),
                        eph_df.sort_values("jd_utc_eph"),
                        left_on="jd_utc_obs", right_on="jd_utc_eph",
                        direction="nearest", tolerance=TOL_DAYS)

    dfM["lighttime_days"] = dfM["lighttime_min"] / 1440.0
    dfM["jd_utc_emit"]    = dfM["jd_utc_obs"] - dfM["lighttime_days"]
    t0e = float(np.nanmin(dfM["jd_utc_emit"].to_numpy(float)))
    dfM["t_emit_hr"] = (dfM["jd_utc_emit"] - t0e) * 24.0

    def _phi1(a): return np.exp(-3.33 * np.power(np.tan(a/2), 0.63))
    def _phi2(a): return np.exp(-1.87 * np.power(np.tan(a/2), 1.22))
    G = HG_G_DEFAULT
    r     = pd.to_numeric(dfM["r_au"],    errors="coerce").to_numpy(float)
    d     = pd.to_numeric(dfM["delta_au"],errors="coerce").to_numpy(float)
    alpha = pd.to_numeric(dfM["alpha_deg"],errors="coerce").to_numpy(float)
    a_rad = np.deg2rad(alpha)
    phase_corr = -2.5 * np.log10(np.clip((1-G)*_phi1(a_rad) + G*_phi2(a_rad), 1e-12, None))

    dfM["dist_term"]  = 5.0 * np.log10(r * d)
    dfM["phase_term"] = phase_corr
    dfM["mag_geo"]    = pd.to_numeric(dfM["mag"], errors="coerce") - dfM["dist_term"] - dfM["phase_term"]

    ok = np.isfinite(dfM["mag_geo"].to_numpy(float))
    dfM["mag_geo_bandcenter"] = np.nan
    if ok.any():
        dfM.loc[ok, "mag_geo_bandcenter"] = (
            dfM.loc[ok, "mag_geo"] - dfM.loc[ok].groupby("band")["mag_geo"].transform("median")
        )

    matched = int(dfM["r_au"].notna().sum())
    meta = {"n_obs": len(dfM), "n_matched": matched,
            "n_unmatched": len(dfM) - matched, "G": G}
    return dfM, meta


# ============================================================
# Plotting helpers
# ============================================================
def plot_fold(ax, t_hr, mag, bands, P_hr, title, mag_label, two_cycles=False):
    phase = (t_hr / float(P_hr)) % 1.0
    for b in [b for b in BAND_ORDER if b in np.unique(bands).tolist()]:
        m = bands == b
        ax.scatter(phase[m], mag[m], s=24, label=b, color=band_color(b),
                   alpha=0.8, edgecolors="none", zorder=3)
        if two_cycles:
            ax.scatter(phase[m] + 1.0, mag[m], s=24, color=band_color(b),
                       alpha=0.35, edgecolors="none", zorder=2)
    ax.invert_yaxis()
    ax.set_xlabel("Phase (0–1)" if not two_cycles else "Phase (0–2)", fontsize=9)
    ax.set_ylabel(mag_label, fontsize=9)
    ax.set_title(title, pad=8)
    ax.set_xlim(0.0, 2.0 if two_cycles else 1.0)


def plot_raw_mag_vs_time(ax, t_day, mag, bands, mag_label):
    for b in [b for b in BAND_ORDER if b in np.unique(bands).tolist()]:
        m = bands == b
        ax.scatter(t_day[m], mag[m], s=18, label=b, color=band_color(b),
                   alpha=0.8, edgecolors="none")
    ax.invert_yaxis()
    ax.set_xlabel("Days since first observation", fontsize=9)
    ax.set_ylabel(mag_label, fontsize=9)
    ax.set_title("Lightcurve: Magnitude vs Time", pad=8)
    ax.legend(fontsize=8, ncol=6, loc="upper right", framealpha=0.85)


def plot_lomb_scargle(ax, t_hr, mag, dy, P_adopted):
    """Compute and plot LS periodogram (0.5–50 h range)."""
    freq = np.linspace(1.0/50.0, 1.0/0.5, 60000)
    periods = 1.0 / freq
    ls = LombScargle(t_hr, mag, dy if dy is not None else None)
    power = ls.power(freq)

    ax.plot(periods, power, color="#334155", lw=0.6, alpha=0.9)
    ax.axvline(P_adopted, color="#dc2626", lw=1.2, ls="--", label=f"Adopted P = {P_adopted:.4f} h")
    ax.axvline(P_adopted * 2, color="#2563eb", lw=0.9, ls=":", alpha=0.7, label=f"2P = {P_adopted*2:.4f} h")
    ax.axvline(P_adopted / 2, color="#059669", lw=0.9, ls=":", alpha=0.7, label=f"P/2 = {P_adopted/2:.4f} h")

    peak_idx = int(np.argmax(power))
    P_peak = float(periods[peak_idx])
    ax.axvline(P_peak, color="#f59e0b", lw=1, ls="-", alpha=0.6, label=f"LS peak = {P_peak:.4f} h")

    ax.set_xlabel("Period (hours)", fontsize=9)
    ax.set_ylabel("LS Power", fontsize=9)
    ax.set_title("Lomb-Scargle Periodogram", pad=8)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
    ax.set_xlim(0.5, 50.0)
    return P_peak


def plot_wrms_scan(ax, t_hr, mag, dy, P_adopted, K=2):
    """WRMS of K=2 Fourier fit across period grid."""
    freq_grid = np.linspace(1.0/50.0, 1.0/0.5, 2000)

    def _wrms_at_P(P_hr):
        phase = (t_hr / P_hr) * 2 * np.pi
        cols = [np.ones(len(t_hr))]
        for k in range(1, K+1):
            cols.append(np.cos(k * phase))
            cols.append(np.sin(k * phase))
        X = np.column_stack(cols)
        try:
            beta, _, _, _ = np.linalg.lstsq(X, mag, rcond=None)
            resid = mag - X @ beta
            return float(np.sqrt(np.mean(resid**2)))
        except Exception:
            return np.nan

    wrms = np.array([_wrms_at_P(1.0/f) for f in freq_grid])
    periods = 1.0 / freq_grid

    ax.plot(periods, wrms, color="#334155", lw=0.6, alpha=0.9)
    ax.axvline(P_adopted, color="#dc2626", lw=1.2, ls="--", label=f"Adopted P")
    ax.axvline(P_adopted*2, color="#2563eb", lw=0.9, ls=":", alpha=0.7, label="2P")
    ax.axvline(P_adopted/2, color="#059669", lw=0.9, ls=":", alpha=0.7, label="P/2")

    valid = wrms[np.isfinite(wrms)]
    if len(valid):
        thresh = float(np.percentile(valid, 15))
        ax.axhline(thresh, color="#94a3b8", lw=0.7, ls=":", alpha=0.6)

    ax.set_xlabel("Period (hours)", fontsize=9)
    ax.set_ylabel("WRMS (K=2 Fourier)", fontsize=9)
    ax.set_title("WRMS Sigma Scan", pad=8)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
    ax.set_xlim(0.5, 50.0)


def plot_residuals(ax_time, ax_hist, t_hr, mag, P_hr, K=2):
    phase = (t_hr / P_hr) * 2 * np.pi
    cols = [np.ones(len(t_hr))]
    for k in range(1, K+1):
        cols.append(np.cos(k * phase))
        cols.append(np.sin(k * phase))
    X = np.column_stack(cols)
    try:
        beta, _, _, _ = np.linalg.lstsq(X, mag, rcond=None)
        resid = mag - X @ beta
    except Exception:
        resid = np.full_like(mag, np.nan)

    ok = np.isfinite(resid)
    ax_time.scatter(t_hr[ok] / 24.0, resid[ok], s=10, color="#64748b", alpha=0.6, edgecolors="none")
    ax_time.axhline(0, color="#dc2626", lw=0.8, ls="--")
    ax_time.set_xlabel("Days", fontsize=9)
    ax_time.set_ylabel("Residual (mag)", fontsize=9)
    ax_time.set_title(f"Residuals vs Time (K={K} @ P={P_hr:.4f} h)", pad=8)

    ax_hist.hist(resid[ok], bins=40, color="#94a3b8", edgecolor="white", alpha=0.85)
    rms = float(np.sqrt(np.mean(resid[ok]**2)))
    ax_hist.axvline(0, color="#dc2626", lw=0.8, ls="--")
    ax_hist.set_xlabel("Residual (mag)", fontsize=9)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.set_title(f"Residual Distribution (RMS = {rms:.4f})", pad=8)


# ============================================================
# GCS image loader (for pre-generated pipeline plots)
# ============================================================
def load_gcs_image(provid, filename):
    try:
        from google.cloud import storage
        sa = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(sa)
        client = storage.Client(project=BQ_PROJECT, credentials=creds)
        bucket = client.bucket(GCS_BUCKET)
        tag_provid = provid.replace(" ", "_")
        blob_path = f"{GCS_PREFIX}/{tag_provid}/{filename}"
        blob = bucket.blob(blob_path)
        if blob.exists():
            return blob.download_as_bytes()
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_gcs_image_cached(provid: str, filename: str):
    return load_gcs_image(provid, filename)


# ============================================================
# App start
# ============================================================
st.markdown("## ATLAST Asteroid Rotation Dashboard")
st.caption("76 asteroids · pipeline v54 · 2025 cohort · 2026-03-10")

if not MASTER_PATH.exists():
    st.error(f"Missing required file: {MASTER_PATH}")
    st.stop()

master = load_master(MASTER_PATH)
RELIABLE_COUNT = int((master[C_RELIABILITY].astype(str).str.lower() == "reliable").sum()) \
    if C_RELIABILITY in master.columns else 0

st.sidebar.markdown("## Mode")
mode = st.sidebar.radio("View", ["Asteroid Viewer", "Population Explorer"],
                        index=0, label_visibility="collapsed")


# ============================================================
# MODE 1: ASTEROID VIEWER
# ============================================================
if mode == "Asteroid Viewer":

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Asteroid")

    if "reliable_only" not in st.session_state:
        st.session_state["reliable_only"] = False

    reliable_only = st.sidebar.checkbox(f"Reliable only ({RELIABLE_COUNT})", key="reliable_only")
    q = st.sidebar.text_input("Search", value="", placeholder="e.g. 2025 ME15")

    df_pick = master.copy()
    if reliable_only and C_RELIABILITY in df_pick.columns:
        df_pick = df_pick[df_pick[C_RELIABILITY].astype(str).map(reliability_short) == "reliable"]
    if q.strip():
        df_pick = df_pick[df_pick[C_DESIG].astype(str).str.contains(q.strip(), case=False, na=False)]

    df_pick = df_pick.sort_values(C_DESIG)
    designations = df_pick[C_DESIG].astype(str).tolist()

    if not designations:
        st.sidebar.warning("No match.")
        st.stop()

    selected = st.sidebar.selectbox("Selected", designations, index=0, key="sel_ast")

    row = master[master[C_DESIG].astype(str) == str(selected)]
    row = row.iloc[0].to_dict() if len(row) else {}
    rel = reliability_short(str(row.get(C_RELIABILITY, "")))

    P_adopt = float(row.get(C_PERIOD, np.nan))
    if not (np.isfinite(P_adopt) and P_adopt > 0):
        P_adopt = 5.0

    prefer_2p = bool(row.get(C_PREFER_2P, False))
    review    = bool(row.get(C_REVIEW, False))

    # ---- Fold controls ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Fold Controls")

    if st.session_state.get("fold_for") != selected:
        old = st.session_state.get("fold_for")
        if old:
            for k in list(st.session_state.keys()):
                if k.startswith(f"_ph_{old}"): del st.session_state[k]
        st.session_state["fold_for"]   = selected
        st.session_state["_reset"]     = True

    candidates = build_period_candidates(row)
    all_bounds = [P_adopt]
    for c in candidates:
        all_bounds.extend([c["period"]/2, c["period"], c["period"]*2])
    lo   = max(1e-6, min(all_bounds) * 0.9)
    hi   = max(all_bounds) * 1.1
    step = float((hi - lo) / 2000.0) if hi > lo else 1e-6

    if st.session_state.pop("_reset", False):
        target = float(np.clip(P_adopt, lo, hi))
        target = round(round((target - lo) / step) * step + lo, 8)
        st.session_state["fold_sl"] = float(np.clip(target, lo, hi))

    if "_set_p" in st.session_state:
        p_set = st.session_state.pop("_set_p")
        target = float(np.clip(p_set, lo, hi))
        target = round(round((target - lo) / step) * step + lo, 8)
        st.session_state["fold_sl"] = float(np.clip(target, lo, hi))

    P_calc = st.sidebar.slider("Fold Period (hours)",
                               min_value=float(lo), max_value=float(hi),
                               step=step, key="fold_sl")
    st.session_state["fold_period"] = float(P_calc)

    if st.sidebar.button("↩ Reset to Adopted", use_container_width=True):
        st.session_state["_reset"] = True
        st.rerun()

    LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
    sel_bands_sidebar = st.sidebar.multiselect("Bands", LSST_BANDS, default=["g", "r", "i"])
    two_cycles = st.sidebar.checkbox("Show two cycles (0–2)", value=False)

    row_limit = BQ_DEFAULT_ROW_LIMIT

    # ---- TABS ----
    tab_char, tab_ev = st.tabs(["Characterisation", "Evidence"])

    # ---- Cache photometry ----
    ck = f"_ph_{selected}"
    if ck not in st.session_state:
        with st.spinner(f"Loading photometry for {selected} ..."):
            try:
                df_raw, bq_meta = bq_fetch_cached(str(selected), row_limit)
            except Exception:
                df_raw, bq_meta = None, {}
        if df_raw is not None and len(df_raw) >= 5:
            df1 = make_df1(df_raw)
            with st.spinner("Geometry correction (Horizons) ..."):
                try:
                    df_geo, meta5 = geo_correct_full(df1, str(selected))
                except Exception as e:
                    df_geo = df1.copy()
                    df_geo["mag_geo"] = df_geo["mag_geo_bandcenter"] = np.nan
                    meta5 = {"error": str(e)}
        else:
            df_geo, meta5 = None, {}
        st.session_state[ck] = {"bq_meta": bq_meta, "df_geo": df_geo, "meta5": meta5}

    cached  = st.session_state[ck]
    bq_meta = cached["bq_meta"]
    df_geo  = cached["df_geo"]
    meta5   = cached["meta5"]

    # Resolve mag column and filter
    if df_geo is not None and len(df_geo) > 0:
        df_geo["band"] = df_geo["band"].map(normalize_lsst_band)
        if df_geo["mag_geo_bandcenter"].notna().sum() >= 5:
            mag_col, mag_label = "mag_geo_bandcenter", "Corrected (band-centred)"
        elif df_geo["mag_geo"].notna().sum() >= 5:
            mag_col, mag_label = "mag_geo", "Corrected"
        else:
            mag_col, mag_label = "mag", "Raw"

        avail = sorted(set(df_geo["band"].dropna().unique()) & set(LSST_BANDS))
        sel_bands = [b for b in sel_bands_sidebar if b in avail] or avail
        dfp = df_geo[df_geo["band"].isin(sel_bands)].dropna(subset=["t_hr", mag_col, "band"])
    else:
        dfp = None

    # Prepare numpy arrays if we have data
    t_hr = t_day = mag_arr = bands_arr = dy_arr = None
    if dfp is not None and len(dfp) > 0:
        t_hr      = dfp["t_hr"].to_numpy(float)
        t_day     = dfp["t_day"].to_numpy(float)
        mag_arr   = pd.to_numeric(dfp[mag_col], errors="coerce").to_numpy(float)
        bands_arr = dfp["band"].to_numpy(str)
        if "rmsmag" in dfp.columns:
            dy_arr = pd.to_numeric(dfp["rmsmag"], errors="coerce").to_numpy(float)
            if not np.any(np.isfinite(dy_arr)): dy_arr = None

    # ==================================================================
    # TAB 1 — CHARACTERISATION
    # ==================================================================
    with tab_char:
        # Header badges
        hdr = f'<div class="badge-row">'
        hdr += f'<span style="font-size:1.3rem;font-weight:700;color:#1e293b;">{selected}</span>'
        hdr += reliability_badge(rel)
        if prefer_2p: hdr += '<span class="badge badge-2p">2P preferred</span>'
        if review:    hdr += '<span class="badge badge-review">⚠ Needs review</span>'
        hdr += '</div>'
        st.markdown(hdr, unsafe_allow_html=True)

        if prefer_2p and row.get(C_PREFER_2P_R):
            reason = str(row[C_PREFER_2P_R])
            if reason and reason != "nan":
                st.info(f"Pipeline 2P preference: *{reason}*")

        # ---- 1. Physical Properties ----
        st.markdown('<p class="section-head">Physical Properties</p>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Adopted Period (h)", format_float(row.get(C_PERIOD), 6))
        k2.metric("Amplitude (mag)",    format_float(row.get(C_AMPLITUDE), 3))
        k3.metric("H Mag",              format_float(row.get(C_HMAG), 2))
        k4.metric("Axial Elongation",   format_float(row.get(C_AXIAL), 3))

        # ---- 2. Colour Indices ----
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<p class="section-head">Colour Indices</p>', unsafe_allow_html=True)
        ci1, ci2, ci3, ci4 = st.columns(4)
        ci1.metric("g − r", format_float(row.get(C_GR), 4))
        ci2.metric("g − i", format_float(row.get(C_GI), 4))
        ci3.metric("r − i", format_float(row.get(C_RI), 4))
        ci4.metric("Mean Mag", format_float(row.get(C_MEAN_MAG), 3))

        # ---- 3. Rotation Lightcurve (three-panel fold) ----
        if t_hr is not None:
            n_nights = resolve_nights(dfp)
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.markdown('<p class="section-head">Rotation Lightcurve</p>', unsafe_allow_html=True)

            st.caption(
                f"Fold period: **{format_float(P_calc, 6)} h** · "
                f"{len(dfp):,} obs · "
                f"{n_nights if n_nights else '—'} nights · "
                f"**{mag_label}** · bands: {', '.join(sel_bands)}"
            )

            P_half, P_two = 0.5 * P_calc, 2.0 * P_calc
            for col, P_hr, title in zip(
                st.columns(3),
                [P_half, P_calc, P_two],
                [f"P/2 = {P_half:.4f} h", f"P = {P_calc:.4f} h", f"2P = {P_two:.4f} h"],
            ):
                with col:
                    fig, ax = plt.subplots(figsize=(5.2, 3.8))
                    plot_fold(ax, t_hr, mag_arr, bands_arr, P_hr, title, mag_label, two_cycles)
                    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
                    fig.tight_layout(pad=1.0)
                    st.pyplot(fig, clear_figure=True)
        else:
            st.info("No photometry available for this asteroid.")

        # ---- 4. Observations ----
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<p class="section-head">Observations</p>', unsafe_allow_html=True)
        o1, o2, o3 = st.columns(3)
        o1.metric("Observations", f"{int(row.get(C_NOBS, 0)):,}" if pd.notna(row.get(C_NOBS)) else "—")
        o2.metric("Arc (days)",   format_float(row.get(C_ARC), 2))
        o3.metric("Nights",      "—" if not (t_hr is not None and resolve_nights(dfp)) else str(resolve_nights(dfp)))

        # ---- 5. Period Candidates ----
        alt_cands = [c for c in candidates if not c.get("is_adopted")]
        if alt_cands:
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.markdown('<p class="section-head">Period Candidates</p>', unsafe_allow_html=True)

            rows_tbl = []
            for cand in candidates:
                p = cand["period"]
                rows_tbl.append({
                    "":           "★" if cand.get("is_adopted") else "",
                    "Source":     cand["label"],
                    "Period (h)": f"{p:.6f}",
                    "P/2 (h)":   f"{p/2:.6f}",
                    "2P (h)":    f"{p*2:.6f}",
                    "Note":      cand.get("note") or "—",
                })
            st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True, hide_index=True)

            st.caption("Click to fold at a candidate period:")
            cols = st.columns(min(len(alt_cands), 4))
            for col, cand in zip(cols, alt_cands):
                p = cand["period"]
                if col.button(f"{cand['label']}: {p:.4f} h", key=f"cb_{p}",
                              use_container_width=True):
                    st.session_state["_set_p"] = p
                    st.rerun()

        # ---- 6. 2P Decision (compact) ----
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<p class="section-head">Half-Period / 2P Analysis</p>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.metric("Pipeline prefers 2P", "Yes ✓" if prefer_2p else "No")
        b2.metric("ΔBIC (2P vs P)",      format_float(row.get(C_DBIC_2P), 2))
        b3.metric("OE Ratio",            format_float(row.get(C_OE_RATIO), 3))

    # ==================================================================
    # TAB 2 — EVIDENCE
    # ==================================================================
    with tab_ev:
        hdr2 = f'<div class="badge-row">'
        hdr2 += f'<span style="font-size:1.3rem;font-weight:700;color:#1e293b;">Evidence: {selected}</span>'
        hdr2 += reliability_badge(rel)
        hdr2 += '</div>'
        st.markdown(hdr2, unsafe_allow_html=True)
        st.caption("Diagnostic plots and data supporting the adopted solution")

        if t_hr is None:
            st.info("No photometry data available.")
            st.stop()

        # ---- 1. Raw lightcurve ----
        st.markdown('<p class="section-head">1. Raw Lightcurve</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10.5, 3.8))
        plot_raw_mag_vs_time(ax, t_day, mag_arr, bands_arr, mag_label)
        fig.tight_layout(pad=1.0)
        st.pyplot(fig, clear_figure=True)

        # ---- 2. Lomb-Scargle Periodogram ----
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<p class="section-head">2. Lomb-Scargle Periodogram</p>', unsafe_allow_html=True)
        st.caption("Computed from geometry-corrected, band-centred magnitudes. "
                   "Red dashed = adopted period; harmonics marked for reference.")

        fig, ax = plt.subplots(figsize=(10.5, 4.0))
        P_ls_peak = plot_lomb_scargle(ax, t_hr, mag_arr, dy_arr, P_adopt)
        fig.tight_layout(pad=1.0)
        st.pyplot(fig, clear_figure=True)

        ls1, ls2 = st.columns(2)
        ls1.metric("LS Peak Period (h)", format_float(P_ls_peak, 4))
        ls2.metric("Adopted Period (h)", format_float(P_adopt, 6))
        if abs(P_ls_peak - P_adopt) / max(P_adopt, 1e-9) > 0.05:
            st.warning(f"LS peak ({P_ls_peak:.4f} h) differs from adopted ({P_adopt:.6f} h) by "
                       f"{abs(P_ls_peak - P_adopt)/P_adopt*100:.1f}%. "
                       "The Fourier/BIC analysis may have preferred a different solution.")

        # ---- 3. WRMS Sigma Scan ----
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<p class="section-head">3. WRMS Sigma Scan</p>', unsafe_allow_html=True)
        st.caption("K=2 Fourier fit RMS across period grid. Lower = better fit. "
                   "Adopted period should sit near a minimum.")
        fig, ax = plt.subplots(figsize=(10.5, 4.0))
        plot_wrms_scan(ax, t_hr, mag_arr, dy_arr, P_adopt)
        fig.tight_layout(pad=1.0)
        st.pyplot(fig, clear_figure=True)

        # ---- 4. Residuals ----
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<p class="section-head">4. Fourier Fit Residuals</p>', unsafe_allow_html=True)
        st.caption(f"K=2 Fourier fit at adopted period ({P_adopt:.6f} h).")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
        plot_residuals(ax1, ax2, t_hr, mag_arr, P_adopt)
        fig.tight_layout(pad=1.2)
        st.pyplot(fig, clear_figure=True)

        # ---- 5. Pipeline diagnostic images (from GCS) ----
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<p class="section-head">5. Pipeline Diagnostic Plots</p>', unsafe_allow_html=True)

        DIAG_IMAGES = [
            (f"step14_sarah_style_{PIPELINE_TAG}.png",
             "4-Panel Diagnostic (Adopted LC, Alt LC, LS periodogram, WRMS scan)"),
            (f"step14_boot_winner_hist_{PIPELINE_TAG}.png",
             "Bootstrap Winner Histogram"),
            (f"step3_raw_mag_vs_time_all_{PIPELINE_TAG}.png",
             "Step 3: Raw Magnitude vs Time (pre-correction)"),
            (f"step5_drift_raw_vs_geo_{PIPELINE_TAG}.png",
             "Step 5: Raw vs Geometry-Corrected Drift"),
        ]

        any_found = False
        for fname, caption in DIAG_IMAGES:
            img_bytes = load_gcs_image_cached(selected, fname)
            if img_bytes:
                st.image(img_bytes, caption=caption, use_container_width=True)
                any_found = True

        if not any_found:
            st.info(
                "No pre-generated pipeline images found in GCS. "
                "Re-run the pipeline with Cell 16 (GCS export) to populate these. "
                "The on-the-fly plots above are computed directly from BigQuery data."
            )

        # ---- 6. BigQuery diagnostics ----
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        with st.expander("BigQuery diagnostics & cost", expanded=False):
            if bq_meta.get("cache_hit") or bq_meta.get("actual_cost_usd", 1) == 0:
                st.success("Served from cache — $0.00 billed")
            else:
                cost = bq_meta.get("actual_cost_usd", 0)
                st.caption(f"Scanned: {bq_meta.get('actual_bytes_human', '—')} · Cost: ${cost:.6f}")
            st.json(bq_meta)

        with st.expander("Full pipeline data row", expanded=False):
            st.json({k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                     for k, v in row.items()})


# ============================================================
# MODE 2: POPULATION EXPLORER
# ============================================================
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Filters")

    rel_options = sorted(master[C_RELIABILITY].dropna().astype(str).unique().tolist()) \
        if C_RELIABILITY in master.columns else []
    selected_rels = st.sidebar.multiselect("Reliability", rel_options, default=rel_options)
    if not selected_rels: selected_rels = rel_options

    pmin = float(master[C_PERIOD].min()) if master[C_PERIOD].notna().any() else 0.0
    pmax = float(master[C_PERIOD].max()) if master[C_PERIOD].notna().any() else 100.0
    p_lo, p_hi = st.sidebar.slider("Period range (hours)",
                                    float(max(0.0, pmin)), float(max(1.0, pmax)),
                                    (float(max(0.0, pmin)), float(max(1.0, pmax))))

    prefer2p_filter = st.sidebar.checkbox("Pipeline-preferred 2P only", value=False)

    df_f = master.copy()
    if C_RELIABILITY in df_f.columns:
        df_f = df_f[df_f[C_RELIABILITY].astype(str).isin(selected_rels)]
    df_f = df_f[df_f[C_PERIOD].between(p_lo, p_hi, inclusive="both")]
    if prefer2p_filter and C_PREFER_2P in df_f.columns:
        df_f = df_f[df_f[C_PREFER_2P].astype(str) == "True"]

    st.sidebar.caption(f"{len(df_f):,} match")

    st.markdown("### Population Overview")
    if len(df_f) == 0:
        st.warning("No asteroids match filters."); st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total",    f"{len(df_f):,}")
    m2.metric("Reliable", f"{int((df_f[C_RELIABILITY].astype(str)=='reliable').sum()):,}")
    m3.metric("Ambiguous", f"{int((df_f[C_RELIABILITY].astype(str)=='ambiguous').sum()):,}")
    m4.metric("2P preferred", f"{int((df_f[C_PREFER_2P].astype(str)=='True').sum()):,}")

    REL_COLORS = {"reliable": "#16a34a", "ambiguous": "#ca8a04", "insufficient": "#dc2626"}

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Period Distribution")
        periods = df_f[C_PERIOD].dropna().to_numpy(float)
        periods = periods[np.isfinite(periods)]
        if len(periods):
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.hist(periods, bins=30, color="#3b82f6", edgecolor="white", alpha=0.85)
            ax.set_xlabel("Period (hours)"); ax.set_ylabel("Count")
            ax.set_title("Rotation Period Histogram")
            fig.tight_layout(); st.pyplot(fig, clear_figure=True)

    with col_r:
        st.markdown("#### Period vs Amplitude")
        x = df_f[C_PERIOD].to_numpy(float)
        y = df_f[C_AMPLITUDE].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum():
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            for rv, clr in REL_COLORS.items():
                m = ok & (df_f[C_RELIABILITY].astype(str) == rv).to_numpy()
                if m.sum():
                    ax.scatter(x[m], y[m], s=28, color=clr, label=rv, alpha=0.85, edgecolors="none")
            ax.set_xlabel("Period (hours)"); ax.set_ylabel("Amplitude (mag)")
            ax.set_title("Period vs Amplitude"); ax.legend(fontsize=8)
            fig.tight_layout(); st.pyplot(fig, clear_figure=True)

    if C_BOOT_BASE in df_f.columns and C_BOOT_2P in df_f.columns:
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown("#### Bootstrap: Base vs 2P")
        st.caption("Above diagonal = stronger 2P support.")
        xb = df_f[C_BOOT_BASE].to_numpy(float)
        yb = df_f[C_BOOT_2P].to_numpy(float)
        ok2 = np.isfinite(xb) & np.isfinite(yb)
        if ok2.sum():
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            for rv, clr in REL_COLORS.items():
                m = ok2 & (df_f[C_RELIABILITY].astype(str) == rv).to_numpy()
                if m.sum():
                    ax.scatter(xb[m], yb[m], s=28, color=clr, label=rv, alpha=0.8, edgecolors="none")
            ax.axline((0,0), slope=1, color="#94a3b8", lw=0.8, ls="--")
            ax.set_xlabel("Bootstrap frac (base)"); ax.set_ylabel("Bootstrap frac (2P)")
            ax.set_title("Base vs 2P Bootstrap"); ax.legend(fontsize=8)
            fig.tight_layout(); st.pyplot(fig, clear_figure=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("#### Master Table")
    show_cols = [c for c in [
        C_DESIG, C_PERIOD, C_AMPLITUDE, C_RELIABILITY,
        C_PREFER_2P, C_DBIC_2P, C_OE_RATIO,
        C_BOOT_BASE, C_BOOT_2P, C_NOBS, C_ARC,
    ] if c in df_f.columns]
    st.dataframe(df_f[show_cols].reset_index(drop=True), use_container_width=True, height=460)

    st.download_button("Download Filtered CSV",
                       data=df_f.to_csv(index=False).encode("utf-8"),
                       file_name="master_filtered.csv", mime="text/csv",
                       use_container_width=True)
