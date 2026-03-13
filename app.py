# app.py — Iteration 2
# ==========================================================
# ATLAST Asteroid Rotation Dashboard
# Updated to use MASTER_rotation_summary_v2026-03-10.csv
# (76 asteroids, pipeline v54, 2025 cohort)
#
# Iteration 2 improvements over 1b:
#   - Custom CSS for visual hierarchy and professional appearance
#   - Numeric period input alongside slider for precise control
#   - Clickable candidate period buttons (fold at any candidate)
#   - Styled matplotlib plots: consistent band palette, gridlines,
#     larger points, proper figure theming
#   - Smart band defaults based on available data per asteroid
#   - Improved information architecture: context vs. evaluation zones
#   - Population Explorer: click-to-navigate from scatter dots
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
# Custom CSS — visual hierarchy, trust signals, polish
# ============================================================
st.markdown("""
<style>
/* ---- Global type & spacing ---- */
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', 'Source Sans Pro', sans-serif;
}
code, pre, .stCode, [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30,41,59,0.55) 0%, rgba(15,23,42,0.65) 100%);
    border: 1px solid rgba(100,116,139,0.25);
    border-radius: 10px;
    padding: 14px 16px 10px 16px;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(56,189,248,0.45);
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #94a3b8 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

/* ---- Section dividers ---- */
.section-rule {
    border: none;
    border-top: 1px solid rgba(100,116,139,0.2);
    margin: 1.2rem 0;
}

/* ---- Badge row ---- */
.badge-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 6px;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.badge-reliable    { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-ambiguous   { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-insufficient{ background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-unknown     { background: rgba(100,116,139,0.15);color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }
.badge-2p          { background: rgba(59,130,246,0.15);  color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.badge-review      { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

/* ---- Candidate buttons ---- */
.cand-btn-row {
    display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0;
}

/* ---- Sidebar polish ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 0.92rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-top: 0.6rem;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 24px;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* ---- App header ---- */
.app-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 2px;
}
.app-header h2 {
    margin: 0;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.app-header .app-subtitle {
    font-size: 0.85rem;
    color: #64748b;
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

# -------------------------
# Horizons config
# -------------------------
HORIZONS_LOCATION = "X05"
HG_G_DEFAULT      = 0.15


# ============================================================
# Consistent band colour palette
# ============================================================
BAND_COLORS = {
    "u": "#7c3aed",   # violet
    "g": "#22d3ee",   # cyan
    "r": "#f97316",   # orange
    "i": "#ef4444",   # red
    "z": "#a78bfa",   # lavender
    "y": "#fbbf24",   # amber
}
BAND_ORDER = ["u", "g", "r", "i", "z", "y"]

def band_color(b: str) -> str:
    return BAND_COLORS.get(b, "#94a3b8")


# ============================================================
# Matplotlib dark theme setup
# ============================================================
def setup_mpl_style():
    """Set a clean dark style for all plots."""
    mpl.rcParams.update({
        "figure.facecolor":  "#0f172a",
        "axes.facecolor":    "#1e293b",
        "axes.edgecolor":    "#334155",
        "axes.labelcolor":   "#cbd5e1",
        "axes.titlesize":    11,
        "axes.titleweight":  600,
        "axes.titlecolor":   "#e2e8f0",
        "axes.grid":         True,
        "grid.color":        "#334155",
        "grid.linewidth":    0.5,
        "grid.alpha":        0.6,
        "xtick.color":       "#94a3b8",
        "ytick.color":       "#94a3b8",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "text.color":        "#e2e8f0",
        "legend.facecolor":  "#1e293b",
        "legend.edgecolor":  "#334155",
        "legend.fontsize":   8.5,
        "legend.labelcolor": "#cbd5e1",
        "savefig.facecolor": "#0f172a",
        "figure.dpi":        110,
    })

setup_mpl_style()


# ============================================================
# Column name constants (single source of truth for new CSV)
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


# -------------------------
# Helpers
# -------------------------
def bytes_to_human(n):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(n)
    for u in units:
        if x < 1000.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1000.0
    return f"{x:.2f} B"


def est_usd_cost(b):
    return (float(b) / 1e12) * float(BQ_USD_PER_TB)


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


def _safe_period(val) -> float | None:
    try:
        v = float(val)
        if np.isfinite(v) and v > 0:
            return round(v, 6)
    except Exception:
        pass
    return None


def reliability_short(rel):
    r = (rel or "").strip().lower()
    return r if r in {"reliable", "ambiguous", "insufficient"} else "unknown"


def reliability_badge_html(rel):
    r = reliability_short(rel)
    return f'<span class="badge badge-{r}">{r.capitalize()}</span>'


# ============================================================
# New CSV: parse pipe-separated additional periods
# ============================================================
def parse_pipe_list(val) -> list[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    parts = str(val).split("|")
    result = []
    for p in parts:
        try:
            v = float(p.strip())
            if np.isfinite(v) and v > 0:
                result.append(round(v, 6))
        except Exception:
            pass
    return result


def build_period_candidates(row: dict) -> list[dict]:
    """
    Build ordered period candidate list from a new-format master row.
    Returns list of dicts:
      {"label": str, "period": float, "note": str | None, "is_adopted": bool}
    """
    candidates: list[dict] = []
    seen: list[float] = []

    P_adopt = _safe_period(row.get(C_PERIOD)) or 0.0

    PANEL_HARMONICS = [0.5, 1.0, 2.0]
    HARMONIC_TOL    = 0.02

    def _is_panel_harmonic(p: float) -> bool:
        if P_adopt <= 0:
            return False
        ratio = p / P_adopt
        return any(abs(ratio - h) / h < HARMONIC_TOL for h in PANEL_HARMONICS)

    def _is_dup(p: float) -> bool:
        return any(abs(p - s) / max(s, 1e-9) < 0.005 for s in seen)

    def _add(label, period, note=None, is_adopted=False):
        if period is None:
            return
        if _is_dup(period):
            return
        seen.append(period)
        candidates.append({"label": label, "period": period, "note": note, "is_adopted": is_adopted})

    _add("Adopted", _safe_period(row.get(C_PERIOD)), is_adopted=True)

    add_periods = parse_pipe_list(row.get(C_ADDITIONAL))
    add_dbics   = parse_pipe_list(row.get(C_ADD_DBIC))

    alt_idx = 1
    for i, p in enumerate(add_periods):
        if _is_panel_harmonic(p):
            continue
        dbic_note = None
        if i < len(add_dbics):
            dbic_note = f"dBIC = {add_dbics[i]:.2f}"
        _add(f"Alt {alt_idx}", p, note=dbic_note)
        alt_idx += 1

    if row.get(C_BOOT_HW_FLAG):
        p_hw = _safe_period(row.get(C_BOOT_HW_P))
        if p_hw and not _is_panel_harmonic(p_hw):
            _add("Boot harmonic", p_hw, note="bootstrap winner")

    return candidates


# ============================================================
# Load master
# ============================================================
def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = [
        C_PERIOD, C_ARC, C_NOBS, C_HMAG, C_AMPLITUDE,
        C_DBIC_2P, C_OE_RATIO, C_AMP_RATIO,
        C_BOOT_BASE, C_BOOT_2P,
        C_GR, C_GI, C_RI, C_AXIAL,
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = safe_num(df[c])
    return df


def resolve_nights(df: pd.DataFrame):
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


# ============================================================
# BigQuery
# ============================================================
LSST_CANON = {"u", "g", "r", "i", "z", "y"}

def normalize_lsst_band(x):
    if x is None:
        return ""
    s = str(x).strip().lower()
    if len(s) == 2 and s[0] == "l" and s[1] in LSST_CANON:
        return s[1]
    m = re.match(r"^(?:lsst)?([ugrizy])$", s)
    if m:
        return m.group(1)
    return s


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


def bq_load_photometry_for_provid(provid, *, row_limit):
    client = get_bq_client()
    source = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    row_limit = int(max(1, min(row_limit, BQ_MAX_ROW_LIMIT)))

    query = f"""
    SELECT provid, obstime, band,
           SAFE_CAST(mag AS FLOAT64) AS mag,
           SAFE_CAST(rmsmag AS FLOAT64) AS rmsmag
    FROM `{source}`
    WHERE provid = @prov
      AND mag IS NOT NULL
    ORDER BY obstime
    LIMIT {row_limit}
    """
    params = [bigquery.ScalarQueryParameter("prov", "STRING", provid)]
    bq_meta = {"provid": provid, "source_table": source, "row_limit": row_limit}

    try:
        dry = client.query(query, location=BQ_LOCATION,
                           job_config=bigquery.QueryJobConfig(
                               query_parameters=params, dry_run=True,
                               use_query_cache=False))
        est = int(getattr(dry, "total_bytes_processed", 0) or 0)
        bq_meta.update({"dry_run_ok": True,
                         "estimated_bytes_human": bytes_to_human(est),
                         "estimated_cost_usd": round(est_usd_cost(est), 6)})
    except Exception as e:
        st.error("BigQuery dry-run failed.")
        st.exception(e)
        raise

    job = client.query(query, location=BQ_LOCATION,
                       job_config=bigquery.QueryJobConfig(
                           query_parameters=params, use_query_cache=True))
    try:
        df = job.to_dataframe(create_bqstorage_client=False)
    except (Forbidden, BadRequest, NotFound) as e:
        st.error("BigQuery query failed.")
        st.exception(e)
        raise

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
    return bq_load_photometry_for_provid(provid, row_limit=row_limit)


def make_df1_from_bq(df_raw):
    df = df_raw.copy()
    df["obstime_dt"] = pd.to_datetime(df["obstime"], errors="coerce", utc=True)
    df["mag"]  = pd.to_numeric(df["mag"],  errors="coerce")
    df["band"] = df.get("band", "x").map(normalize_lsst_band)
    df = df.dropna(subset=["obstime_dt", "mag", "band"]).reset_index(drop=True)
    if len(df) == 0:
        return df
    df = df.sort_values("obstime_dt")
    t0 = df["obstime_dt"].min()
    df["t_hr"]     = (df["obstime_dt"] - t0).dt.total_seconds() / 3600.0
    df["t_day"]    = (df["obstime_dt"] - t0).dt.total_seconds() / 86400.0
    df["night_utc"]= df["obstime_dt"].dt.strftime("%Y-%m-%d")
    return df


# ============================================================
# Geometry correction (Step 5 / Horizons) — unchanged
# ============================================================
def step5_geometry_horizons_range(df1, *, PROVID, OUTDIR,
                                   HORIZONS_LOCATION="X05", G_DEFAULT=0.15,
                                   STEP_MINUTES=10, PAD_MINUTES=10,
                                   FAIL_ON_UNMATCHED=False, TOL_DAYS=None,
                                   save_tables=False, **_kw):
    if df1 is None or len(df1) == 0:
        raise ValueError("STEP 5: df1 is empty.")
    need = ["obstime_dt", "mag", "band"]
    missing = [c for c in need if c not in df1.columns]
    if missing:
        raise ValueError(f"STEP 5: df1 missing: {missing}")
    if TOL_DAYS is None:
        TOL_DAYS = max(2e-3, (STEP_MINUTES / 1440.0) * 1.2)
    if save_tables:
        os.makedirs(OUTDIR, exist_ok=True)

    dfG = df1.copy()
    dfG["band"] = dfG["band"].map(normalize_lsst_band)
    dfG["obstime_dt"] = pd.to_datetime(dfG["obstime_dt"], errors="coerce", utc=True)
    dfG = dfG.dropna(subset=["obstime_dt"]).sort_values("obstime_dt").reset_index(drop=True)
    if len(dfG) == 0:
        raise ValueError("STEP 5: all obstime_dt NaT.")
    if "t_hr" not in dfG.columns:
        t0 = dfG["obstime_dt"].min()
        dfG["t_hr"] = (dfG["obstime_dt"] - t0).dt.total_seconds() / 3600.0
    if "night_id" not in dfG.columns and "night_utc" not in dfG.columns:
        dfG["night_utc"] = dfG["obstime_dt"].dt.strftime("%Y-%m-%d")
    night_key = "night_id" if "night_id" in dfG.columns else "night_utc"

    t_utc = Time(dfG["obstime_dt"].dt.to_pydatetime(), scale="utc")
    dfG["jd_utc_obs"] = t_utc.jd.astype(float)

    def _query_horizons(desig, start_utc, stop_utc, step_min, loc):
        obj = Horizons(id=desig, id_type="smallbody", location=loc,
                       epochs={"start": start_utc, "stop": stop_utc, "step": f"{int(step_min)}m"})
        df = obj.ephemerides().to_pandas()
        for k in ["datetime_jd", "r", "delta", "alpha", "lighttime"]:
            if k not in df.columns:
                raise KeyError(f"Horizons missing '{k}'")
        return pd.DataFrame({
            "jd_utc_eph":    df["datetime_jd"].astype(float).to_numpy(),
            "r_au":          df["r"].astype(float).to_numpy(),
            "delta_au":      df["delta"].astype(float).to_numpy(),
            "alpha_deg":     df["alpha"].astype(float).to_numpy(),
            "lighttime_min": df["lighttime"].astype(float).to_numpy(),
        })

    eph_parts = []
    for block in sorted(dfG[night_key].dropna().unique()):
        sub  = dfG[dfG[night_key] == block]
        tmin = pd.to_datetime(sub["obstime_dt"].min(), utc=True) - pd.Timedelta(minutes=PAD_MINUTES)
        tmax = pd.to_datetime(sub["obstime_dt"].max(), utc=True) + pd.Timedelta(minutes=PAD_MINUTES)
        eph_parts.append(_query_horizons(PROVID,
                                         tmin.strftime("%Y-%m-%d %H:%M"),
                                         tmax.strftime("%Y-%m-%d %H:%M"),
                                         STEP_MINUTES, HORIZONS_LOCATION))

    eph_df = (pd.concat(eph_parts, ignore_index=True)
                .drop_duplicates("jd_utc_eph")
                .sort_values("jd_utc_eph")
                .reset_index(drop=True))

    dfM = pd.merge_asof(dfG.sort_values("jd_utc_obs"),
                        eph_df.sort_values("jd_utc_eph"),
                        left_on="jd_utc_obs", right_on="jd_utc_eph",
                        direction="nearest", tolerance=TOL_DAYS)

    matched   = int(dfM["r_au"].notna().sum())
    n_total   = int(len(dfM))
    n_unmatched = n_total - matched
    if n_unmatched > 0 and FAIL_ON_UNMATCHED:
        raise RuntimeError("Unmatched ephemeris rows after merge_asof.")

    dfM["lighttime_days"] = dfM["lighttime_min"] / 1440.0
    dfM["jd_utc_emit"]    = dfM["jd_utc_obs"] - dfM["lighttime_days"]
    t0e = float(np.nanmin(dfM["jd_utc_emit"].to_numpy(float)))
    dfM["t_emit_hr"] = (dfM["jd_utc_emit"] - t0e) * 24.0

    def _phi1(a): return np.exp(-3.33 * np.power(np.tan(a/2), 0.63))
    def _phi2(a): return np.exp(-1.87 * np.power(np.tan(a/2), 1.22))
    def _phase_HG(alpha_deg, G=0.15):
        a = np.deg2rad(alpha_deg)
        p = np.clip((1-G)*_phi1(a) + G*_phi2(a), 1e-12, None)
        return -2.5 * np.log10(p)

    r     = pd.to_numeric(dfM["r_au"],    errors="coerce").to_numpy(float)
    d     = pd.to_numeric(dfM["delta_au"],errors="coerce").to_numpy(float)
    alpha = pd.to_numeric(dfM["alpha_deg"],errors="coerce").to_numpy(float)

    dfM["dist_term"]  = 5.0 * np.log10(r * d)
    dfM["phase_term"] = _phase_HG(alpha, G_DEFAULT)
    dfM["mag_geo"]    = pd.to_numeric(dfM["mag"], errors="coerce") - dfM["dist_term"] - dfM["phase_term"]

    ok = np.isfinite(dfM["mag_geo"].to_numpy(float))
    dfM["mag_geo_bandcenter"] = np.nan
    if ok.any():
        dfM.loc[ok, "mag_geo_bandcenter"] = (
            dfM.loc[ok, "mag_geo"]
            - dfM.loc[ok].groupby("band")["mag_geo"].transform("median")
        )

    meta = {"n_obs": n_total, "n_matched": matched, "n_unmatched": n_unmatched,
            "G_DEFAULT": G_DEFAULT, "STEP_MINUTES": STEP_MINUTES}
    return dfM, meta


def geo_correct(df1, provid):
    return step5_geometry_horizons_range(
        df1, PROVID=provid, OUTDIR=".",
        HORIZONS_LOCATION=HORIZONS_LOCATION, G_DEFAULT=HG_G_DEFAULT,
        STEP_MINUTES=10, PAD_MINUTES=10, FAIL_ON_UNMATCHED=False,
    )


# ============================================================
# Styled fold plot
# ============================================================
def plot_fold(ax, t_hr, mag, bands, P_hr, title, mag_label, two_cycles=False):
    phase = (t_hr / float(P_hr)) % 1.0
    for b in [b for b in BAND_ORDER if b in np.unique(bands).tolist()]:
        m = bands == b
        ax.scatter(phase[m], mag[m], s=22, label=b, color=band_color(b),
                   alpha=0.8, edgecolors="none", zorder=3)
        if two_cycles:
            ax.scatter(phase[m] + 1.0, mag[m], s=22, color=band_color(b),
                       alpha=0.45, edgecolors="none", zorder=2)
    ax.invert_yaxis()
    ax.set_xlabel("Phase (0–1)" if not two_cycles else "Phase (0–2)",
                  fontsize=9, color="#94a3b8")
    ax.set_ylabel(mag_label, fontsize=9, color="#94a3b8")
    ax.set_title(title, fontsize=11, fontweight=600, pad=8)
    ax.set_xlim(0.0, 2.0 if two_cycles else 1.0)


# ============================================================
# App start
# ============================================================
st.markdown(
    '<div class="app-header">'
    '<h2>ATLAST Asteroid Rotation Dashboard</h2>'
    '<span class="app-subtitle">76 asteroids · pipeline v54 · 2025 cohort · 2026-03-10</span>'
    '</div>',
    unsafe_allow_html=True,
)

if not MASTER_PATH.exists():
    st.error(f"Missing required file: {MASTER_PATH}")
    st.stop()

master = load_master(MASTER_PATH)

RELIABLE_COUNT = int((master[C_RELIABILITY].astype(str).str.lower() == "reliable").sum()) \
    if C_RELIABILITY in master.columns else 0

# -------------------------
# Sidebar — Mode
# -------------------------
st.sidebar.markdown("## Mode")
mode = st.sidebar.radio("View", ["Asteroid Viewer", "Population Explorer"], index=0,
                        label_visibility="collapsed")


# ============================================================
# MODE 1: ASTEROID VIEWER
# ============================================================
if mode == "Asteroid Viewer":

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Asteroid")

    if "reliable_only" not in st.session_state:
        st.session_state["reliable_only"] = False

    # Filter BEFORE search so toggling doesn't lose the selected asteroid
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
        st.sidebar.warning("No asteroids match your current search/filter.")
        st.stop()

    selected = st.sidebar.selectbox("Selected Asteroid", designations, index=0, key="selected_asteroid")

    row = master[master[C_DESIG].astype(str) == str(selected)]
    row = row.iloc[0].to_dict() if len(row) else {}
    rel = reliability_short(str(row.get(C_RELIABILITY, "")))

    P_adopt = float(row.get(C_PERIOD, np.nan))
    if not (np.isfinite(P_adopt) and P_adopt > 0):
        P_adopt = 5.0


    # ---- Fold Controls ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Fold Controls")

    # Handle asteroid switch: reset period and clear photometry cache
    if st.session_state.get("fold_period_for") != selected:
        old = st.session_state.get("fold_period_for")
        if old:
            for k in list(st.session_state.keys()):
                if k.startswith(f"_phot_{old}"):
                    del st.session_state[k]
        st.session_state["fold_period_for"]     = selected
        st.session_state["_reset_to_adopted"]   = True

    candidates = build_period_candidates(row)

    # Slider range covers adopted P and all candidate harmonics
    all_bounds = [P_adopt]
    for cand in candidates:
        p = cand["period"]
        all_bounds.extend([p / 2.0, p, p * 2.0])
    lo   = max(1e-6, min(all_bounds) * 0.9)
    hi   = max(all_bounds) * 1.1
    step = float((hi - lo) / 2000.0) if hi > lo else 1e-6   # finer: 2000 steps

    # ------------------------------------------------------------------
    # SLIDER STATE RULES — same as before, with numeric input added
    # ------------------------------------------------------------------
    if st.session_state.pop("_reset_to_adopted", False):
        target = float(np.clip(P_adopt, lo, hi))
        target = round(round((target - lo) / step) * step + lo, 8)
        target = float(np.clip(target, lo, hi))
        st.session_state["fold_period_slider"] = target

    # Candidate button press? Override slider position.
    if "_set_period_to" in st.session_state:
        p_set = st.session_state.pop("_set_period_to")
        target = float(np.clip(p_set, lo, hi))
        target = round(round((target - lo) / step) * step + lo, 8)
        target = float(np.clip(target, lo, hi))
        st.session_state["fold_period_slider"] = target

    P_calc = st.sidebar.slider(
        "Fold Period (hours)",
        min_value=float(lo),
        max_value=float(hi),
        step=step,
        key="fold_period_slider",
    )

    # Numeric input — precise control
    P_numeric = st.sidebar.number_input(
        "Exact period (hours)",
        min_value=0.001,
        max_value=999.0,
        value=float(P_calc),
        step=0.001,
        format="%.6f",
        key="fold_period_number",
    )
    # If numeric input differs from slider, use numeric
    if abs(P_numeric - P_calc) > 1e-7:
        P_calc = P_numeric

    st.session_state["fold_period"] = float(P_calc)

    if st.sidebar.button("↩ Reset To Adopted Period", use_container_width=True):
        st.session_state["_reset_to_adopted"] = True
        st.rerun()

    LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
    sel_bands_sidebar = st.sidebar.multiselect("Bands", LSST_BANDS, default=["g", "r", "i"])
    two_cycles = st.sidebar.checkbox("Show two cycles (0–2)", value=False)

    row_limit = BQ_DEFAULT_ROW_LIMIT

    # ---- Main tabs ----
    tab_photo, tab_char = st.tabs(["📈  Photometry", "📋  Characterisation"])

    # ------------------------------------------------------------------
    # Two-layer cache (same logic as before)
    # ------------------------------------------------------------------
    cache_key = f"_phot_{selected}"

    if cache_key not in st.session_state:
        with st.spinner(f"Loading photometry for {selected} ..."):
            try:
                df_raw, bq_meta = bq_fetch_cached(str(selected), row_limit)
            except Exception:
                df_raw, bq_meta = None, {}

        if df_raw is not None and len(df_raw) >= 5:
            df1 = make_df1_from_bq(df_raw)
            with st.spinner("Running geometry correction (Horizons) ..."):
                try:
                    df_geo, meta5 = geo_correct(df1, str(selected))
                except Exception as e:
                    df_geo = df1.copy()
                    df_geo["mag_geo"] = df_geo["mag_geo_bandcenter"] = np.nan
                    meta5 = {"error": str(e)}
        else:
            df_geo, meta5 = None, {}

        st.session_state[cache_key] = {
            "bq_meta": bq_meta,
            "df_geo":  df_geo,
            "meta5":   meta5,
        }

    cached  = st.session_state[cache_key]
    bq_meta = cached["bq_meta"]
    df_geo  = cached["df_geo"]
    meta5   = cached["meta5"]

    with tab_photo:
        # ---- Object header with badges ----
        prefer_2p = bool(row.get(C_PREFER_2P, False))
        review    = bool(row.get(C_REVIEW, False))

        badge_html = f'<div class="badge-row">'
        badge_html += f'<span style="font-size:1.35rem;font-weight:700;color:#e2e8f0;">{selected}</span>'
        badge_html += reliability_badge_html(rel)
        if prefer_2p:
            badge_html += '<span class="badge badge-2p">2P preferred</span>'
        if review:
            badge_html += '<span class="badge badge-review">⚠ Needs review</span>'
        badge_html += '</div>'
        st.markdown(badge_html, unsafe_allow_html=True)

        if prefer_2p and row.get(C_PREFER_2P_R):
            st.info(f"Pipeline 2P preference: *{row[C_PREFER_2P_R]}*")

        with st.expander("BigQuery diagnostics & cost", expanded=False):
            if bq_meta.get("cache_hit") or bq_meta.get("actual_cost_usd", 1) == 0:
                st.success("Served from cache — $0.00 billed this load")
            else:
                cost = bq_meta.get("actual_cost_usd", 0)
                scanned = bq_meta.get("actual_bytes_human", "—")
                st.caption(f"Scanned: {scanned} · Est. cost: ${cost:.6f}")
            st.json(bq_meta)

        if bq_meta.get("may_be_truncated"):
            st.warning(f"Returned {bq_meta['returned_rows']} rows — hit row limit. "
                       "Results may be truncated.")

        if df_geo is None or len(df_geo) == 0:
            st.info("No photometry found in BigQuery for this asteroid.")
            st.stop()

        if df_geo["mag_geo_bandcenter"].notna().sum() >= 5:
            mag_col, mag_label = "mag_geo_bandcenter", "Corrected (band-centred)"
        elif df_geo["mag_geo"].notna().sum() >= 5:
            mag_col, mag_label = "mag_geo", "Corrected"
        else:
            mag_col, mag_label = "mag", "Raw mag"

        # Smart band defaults: use sidebar selection, but fall back to available bands
        df_geo["band"] = df_geo["band"].map(normalize_lsst_band)
        avail     = sorted(set(df_geo["band"].dropna().astype(str).unique()) & set(LSST_BANDS))
        sel_bands = [b for b in sel_bands_sidebar if b in avail]
        if not sel_bands:
            sel_bands = avail if avail else sorted(df_geo["band"].dropna().unique().tolist())

        dfp = df_geo[df_geo["band"].isin(sel_bands)].dropna(subset=["t_hr", mag_col, "band"])
        n_nights = resolve_nights(dfp)

        # ---- Summary metrics ----
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Adopted Period (h)",    format_float(P_adopt, 6))
        s2.metric("Fold Period (h)",        format_float(P_calc,  6))
        s3.metric("Observations",          f"{len(dfp):,}")
        s4.metric("Nights",               "—" if n_nights is None else str(n_nights))

        st.caption(f"Folding **{mag_label}** magnitudes · bands: {', '.join(sel_bands)}")

        # ---- Clickable period candidate buttons ----
        alt_candidates = [c for c in candidates if not c.get("is_adopted")]
        if alt_candidates:
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.markdown("**Period Candidates** — click to fold")
            # Build columns: one per candidate, max 5 per row
            row_cands = alt_candidates
            while row_cands:
                batch = row_cands[:5]
                row_cands = row_cands[5:]
                cols = st.columns(len(batch))
                for col, cand in zip(cols, batch):
                    p = cand["period"]
                    note = cand.get("note") or ""
                    label = f"{cand['label']}: {p:.4f} h"
                    if col.button(label, key=f"cand_{p}", use_container_width=True):
                        st.session_state["_set_period_to"] = p
                        st.rerun()
                    if note:
                        col.caption(note)
                    # Sub-harmonic buttons
                    sub_cols = col.columns(3)
                    for sc, (mult, lbl) in zip(sub_cols, [(0.5, "P/2"), (1.0, "P"), (2.0, "2P")]):
                        pv = round(p * mult, 6)
                        if sc.button(lbl, key=f"cand_{p}_{lbl}", use_container_width=True):
                            st.session_state["_set_period_to"] = pv
                            st.rerun()

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # ---- Three-panel fold ----
        t_hr  = dfp["t_hr"].to_numpy(float)
        t_day = dfp["t_day"].to_numpy(float)
        mag   = pd.to_numeric(dfp[mag_col], errors="coerce").to_numpy(float)
        bands = dfp["band"].to_numpy(str)

        P_half = 0.5 * P_calc
        P_two  = 2.0 * P_calc

        st.markdown("#### Three-Panel Fold &nbsp;(P/2 · P · 2P)")
        for col, P_hr, title in zip(
            st.columns(3),
            [P_half, P_calc, P_two],
            [f"P/2 = {P_half:.4f} h", f"P = {P_calc:.4f} h", f"2P = {P_two:.4f} h"],
        ):
            with col:
                fig, ax = plt.subplots(figsize=(5.2, 3.8))
                plot_fold(ax, t_hr, mag, bands, P_hr, title, mag_label, two_cycles)
                ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
                fig.tight_layout(pad=1.0)
                st.pyplot(fig, clear_figure=True)

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # ---- Mag vs Time ----
        st.markdown("#### Magnitude vs Time")
        fig, ax = plt.subplots(figsize=(10.5, 3.8))
        for b in [b for b in BAND_ORDER if b in np.unique(bands).tolist()]:
            m = bands == b
            ax.scatter(t_day[m], mag[m], s=18, label=b, color=band_color(b),
                       alpha=0.8, edgecolors="none")
        ax.invert_yaxis()
        ax.set_xlabel("Days Since First Observation", fontsize=9, color="#94a3b8")
        ax.set_ylabel(mag_label, fontsize=9, color="#94a3b8")
        ax.set_title("Magnitude vs Time", fontsize=11, fontweight=600, pad=8)
        ax.legend(fontsize=8, ncol=6, loc="upper right", framealpha=0.7)
        fig.tight_layout(pad=1.0)
        st.pyplot(fig, clear_figure=True)

    # ------------------------------------------------------------------
    # Characterisation tab
    # ------------------------------------------------------------------
    with tab_char:
        badge_html2 = f'<div class="badge-row">'
        badge_html2 += f'<span style="font-size:1.35rem;font-weight:700;color:#e2e8f0;">{selected}</span>'
        badge_html2 += reliability_badge_html(rel)
        badge_html2 += '</div>'
        st.markdown(badge_html2, unsafe_allow_html=True)
        st.caption("Values from MASTER_rotation_summary_v2026-03-10.csv (pipeline v54)")

        # Row 1 — period
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Adopted Period (h)",  format_float(row.get(C_PERIOD), 6))
        k2.metric("Amplitude (mag)",     format_float(row.get(C_AMPLITUDE), 3))
        k3.metric("Axial Elongation",    format_float(row.get(C_AXIAL), 3))
        k4.metric("H Mag",               format_float(row.get(C_HMAG), 3))

        # Row 2 — 2P decision
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Pipeline prefers 2P", "Yes ✓" if prefer_2p else "No")
        b2.metric("ΔBIC (2P vs P)",      format_float(row.get(C_DBIC_2P), 2))
        b3.metric("OE Ratio",            format_float(row.get(C_OE_RATIO), 3))
        b4.metric("Amp Ratio (2P/P)",    format_float(row.get(C_AMP_RATIO), 3))

        # Row 3 — bootstrap
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bootstrap Frac (base)", format_float(row.get(C_BOOT_BASE), 3))
        c2.metric("Bootstrap Frac (2P)",   format_float(row.get(C_BOOT_2P), 3))
        c3.metric("Observations",          f"{int(row.get(C_NOBS, 0)):,}" if pd.notna(row.get(C_NOBS)) else "—")
        c4.metric("Arc (days)",            format_float(row.get(C_ARC), 2))

        if row.get(C_PREFER_2P_R):
            st.info(f"2P preference reason: *{row[C_PREFER_2P_R]}*")
        if prefer_2p:
            p2_implied = round(P_adopt * 2, 6)
            st.success(f"Pipeline suggests true period may be **2P = {p2_implied:.6f} h** — "
                       f"click the candidate button above to fold at this value.")

        # Colour indices
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown("#### Colour Indices")
        ci1, ci2, ci3 = st.columns(3)
        ci1.metric("g − r", format_float(row.get(C_GR), 4))
        ci2.metric("g − i", format_float(row.get(C_GI), 4))
        ci3.metric("r − i", format_float(row.get(C_RI), 4))

        # Period candidate table
        if candidates:
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.markdown("#### Period Candidate Summary")
            rows_tbl = []
            for cand in candidates:
                p = cand["period"]
                rows_tbl.append({
                    "Source":      cand["label"] + (" ★" if cand.get("is_adopted") else ""),
                    "Period (h)":  f"{p:.6f}",
                    "P/2 (h)":     f"{p/2:.6f}",
                    "2P (h)":      f"{p*2:.6f}",
                    "Note":        cand.get("note") or "—",
                })
            st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True, hide_index=True)

        # Full raw row (collapsed)
        with st.expander("Full pipeline data row", expanded=False):
            st.json({k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                     for k, v in row.items()})


# ============================================================
# MODE 2: POPULATION EXPLORER
# ============================================================
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Population Filters")

    rel_options  = sorted(master[C_RELIABILITY].dropna().astype(str).unique().tolist()) \
        if C_RELIABILITY in master.columns else ["reliable", "ambiguous", "insufficient"]
    default_rels = rel_options
    selected_rels = st.sidebar.multiselect("Reliability", rel_options, default=default_rels)
    if not selected_rels:
        selected_rels = rel_options

    pmin = float(master[C_PERIOD].min()) if master[C_PERIOD].notna().any() else 0.0
    pmax = float(master[C_PERIOD].max()) if master[C_PERIOD].notna().any() else 100.0
    p_lo, p_hi = st.sidebar.slider("Period range (hours)",
                                    float(max(0.0, pmin)), float(max(1.0, pmax)),
                                    (float(max(0.0, pmin)), float(max(1.0, pmax))))

    prefer2p_filter = st.sidebar.checkbox("Pipeline-preferred 2P only", value=False)

    row_limit   = BQ_DEFAULT_ROW_LIMIT

    df_f = master.copy()
    if C_RELIABILITY in df_f.columns:
        df_f = df_f[df_f[C_RELIABILITY].astype(str).isin(selected_rels)]
    df_f = df_f[df_f[C_PERIOD].between(p_lo, p_hi, inclusive="both")]
    if prefer2p_filter and C_PREFER_2P in df_f.columns:
        df_f = df_f[df_f[C_PREFER_2P].astype(str) == "True"]

    st.sidebar.caption(f"{len(df_f):,} asteroids match filters")

    st.markdown("### Population Overview")
    if len(df_f) == 0:
        st.warning("No asteroids match current filters.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total (filtered)",  f"{len(df_f):,}")
    m2.metric("Reliable",          f"{int((df_f[C_RELIABILITY].astype(str)=='reliable').sum()):,}")
    m3.metric("Ambiguous",         f"{int((df_f[C_RELIABILITY].astype(str)=='ambiguous').sum()):,}")
    m4.metric("Pipeline prefers 2P", f"{int((df_f[C_PREFER_2P].astype(str)=='True').sum()):,}")

    col_l, col_r = st.columns(2)

    REL_COLORS = {"reliable": "#4ade80", "ambiguous": "#fbbf24", "insufficient": "#f87171"}

    with col_l:
        st.markdown("#### Period Distribution")
        periods = df_f[C_PERIOD].dropna().to_numpy(float)
        periods = periods[np.isfinite(periods)]
        if len(periods):
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.hist(periods, bins=30, color="#38bdf8", edgecolor="#1e293b", alpha=0.85)
            ax.set_xlabel("Adopted Period (hours)", fontsize=9, color="#94a3b8")
            ax.set_ylabel("Count", fontsize=9, color="#94a3b8")
            ax.set_title("Rotation Period Histogram", fontsize=11, fontweight=600, pad=8)
            fig.tight_layout(pad=1.0)
            st.pyplot(fig, clear_figure=True)

    with col_r:
        st.markdown("#### Period vs Amplitude")
        x = df_f[C_PERIOD].to_numpy(float)
        y = df_f[C_AMPLITUDE].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum():
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            for rel_val, colour in REL_COLORS.items():
                m = ok & (df_f[C_RELIABILITY].astype(str) == rel_val).to_numpy()
                if m.sum():
                    ax.scatter(x[m], y[m], s=28, color=colour, label=rel_val,
                               alpha=0.85, edgecolors="none", zorder=3)
            ax.set_xlabel("Adopted Period (hours)", fontsize=9, color="#94a3b8")
            ax.set_ylabel("Amplitude (mag)", fontsize=9, color="#94a3b8")
            ax.set_title("Period vs Amplitude", fontsize=11, fontweight=600, pad=8)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
            fig.tight_layout(pad=1.0)
            st.pyplot(fig, clear_figure=True)

        # Tip for navigating to individual asteroids
        st.caption("💡 To inspect an asteroid, switch to **Asteroid Viewer** and search by designation.")

    # Bootstrap frac scatter
    if C_BOOT_BASE in df_f.columns and C_BOOT_2P in df_f.columns:
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown("#### Bootstrap Fraction: Base vs 2P")
        st.caption("Points above the diagonal have higher bootstrap support at 2P than at the base period. "
                   "Asteroids in that region are stronger 2P candidates.")
        xb = df_f[C_BOOT_BASE].to_numpy(float)
        yb = df_f[C_BOOT_2P].to_numpy(float)
        ok2 = np.isfinite(xb) & np.isfinite(yb)
        if ok2.sum():
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            for rel_val, colour in REL_COLORS.items():
                m = ok2 & (df_f[C_RELIABILITY].astype(str) == rel_val).to_numpy()
                if m.sum():
                    ax.scatter(xb[m], yb[m], s=28, color=colour, label=rel_val,
                               alpha=0.8, edgecolors="none", zorder=3)
            ax.axline((0, 0), slope=1, color="#475569", lw=1, linestyle="--", zorder=1)
            ax.set_xlabel("Bootstrap frac (base period)", fontsize=9, color="#94a3b8")
            ax.set_ylabel("Bootstrap frac (2P)", fontsize=9, color="#94a3b8")
            ax.set_title("Base vs 2P Bootstrap Fraction", fontsize=11, fontweight=600, pad=8)
            ax.legend(fontsize=8, loc="lower right", framealpha=0.7)
            fig.tight_layout(pad=1.0)
            st.pyplot(fig, clear_figure=True)

    # Master table
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("#### Master Table")
    show_cols = [c for c in [
        C_DESIG, C_PERIOD, C_AMPLITUDE, C_RELIABILITY,
        C_PREFER_2P, C_DBIC_2P, C_OE_RATIO,
        C_BOOT_BASE, C_BOOT_2P, C_NOBS, C_ARC,
    ] if c in df_f.columns]
    st.dataframe(df_f[show_cols].reset_index(drop=True), use_container_width=True, height=460)

    st.download_button(
        "Download Filtered CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="master_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
