# app.py — Iteration 1b
# ==========================================================
# ATLAST Asteroid Rotation Dashboard
# Updated to use MASTER_rotation_summary_v2026-03-10.csv
# (76 asteroids, pipeline v54, 2025 cohort)
#
# Column mapping vs old CSV:
#   "Adopted period (hr)"  → "Period"
#   "Arc (days)"           → "Arc"
#   "LS peak period (hr)"  → (derived from Additional periods where relevant)
#   "2P candidate (hr)"    → parsed from "Additional periods (hr)" pipe-string
#   "ΔBIC(2P−P)"           → "adopt_delta_BIC_2P_vs_P"
#   "Bootstrap top_frac"   → "adopt_boot_frac_base"
#
# Iteration 1b additions:
#   - Period Candidate Ladder (adopted + all additional periods from pipeline)
#   - Per-candidate P/2 · P · 2P one-click buttons
#   - dBIC shown per additional period candidate
#   - 2P-preference flag and reason displayed prominently
#   - "Needs human review" warning badge
#   - Characterisation tab uses new richer columns
# ==========================================================

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


def reliability_html(rel):
    r = reliability_short(rel)
    colours = {
        "reliable":    "#22c55e",
        "ambiguous":   "#f59e0b",
        "insufficient":"#ef4444",
        "unknown":     "#64748b",
    }
    return f'<span style="color:{colours[r]};font-weight:800;">{r.capitalize()}</span>'


# ============================================================
# New CSV: parse pipe-separated additional periods
# ============================================================
def parse_pipe_list(val) -> list[float]:
    """Parse '4.433' or '2.611|2.933|5.222' → [4.433] or [2.611, 2.933, 5.222]"""
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

    Order:
      1. Adopted period (Period column)  — always first
      2. Additional periods (pipe-list)  — each with its dBIC note
      3. Boot harmonic winner            — if flagged and distinct

    Deduplication: skip any period within 0.1% of an already-seen one.
    """
    candidates: list[dict] = []
    seen: list[float] = []

    def _is_dup(p: float) -> bool:
        return any(abs(p - s) / max(s, 1e-9) < 0.001 for s in seen)

    def _add(label: str, period: float | None, note: str | None = None, is_adopted: bool = False):
        if period is None:
            return
        if _is_dup(period):
            return
        seen.append(period)
        candidates.append({
            "label": label,
            "period": period,
            "note": note,
            "is_adopted": is_adopted,
        })

    # 1. Adopted
    _add("Adopted", _safe_period(row.get(C_PERIOD)), is_adopted=True)

    # 2. Additional periods from pipeline (with per-period dBIC)
    add_periods = parse_pipe_list(row.get(C_ADDITIONAL))
    add_dbics   = parse_pipe_list(row.get(C_ADD_DBIC))

    for i, p in enumerate(add_periods):
        dbic_note = None
        if i < len(add_dbics):
            dbic_note = f"dBIC = {add_dbics[i]:.2f}"
        _add(f"Alt {i+1}", p, note=dbic_note)

    # 3. Bootstrap harmonic winner (if distinct)
    if row.get(C_BOOT_HW_FLAG):
        _add("Boot harmonic", _safe_period(row.get(C_BOOT_HW_P)), note="bootstrap winner")

    return candidates


def render_period_candidate_ladder(
    candidates: list[dict],
    current_fold_period: float,
    state_key: str = "fold_period",
):
    """
    Render clickable P/2 · P · 2P buttons per candidate in the sidebar.
    Active period (within 0.05%) gets a ✦ marker.
    """
    if not candidates:
        return

    st.sidebar.markdown("**Period Candidates**")
    st.sidebar.caption("Click any button to fold at that period")

    TOL = 0.0005

    def _active(p: float) -> bool:
        return p > 0 and abs(p - current_fold_period) / max(current_fold_period, 1e-9) < TOL

    def _lbl(p: float) -> str:
        return f"{p:.4f} h{'  ✦' if _active(p) else ''}"

    for cand in candidates:
        P_c  = cand["period"]
        note = cand.get("note")
        badge = " ★" if cand.get("is_adopted") else ""

        header = f"**{cand['label']}{badge}** — {P_c:.6f} h"
        if note:
            header += f"  `{note}`"
        st.sidebar.markdown(header)

        half_p = round(P_c / 2.0, 6)
        two_p  = round(P_c * 2.0, 6)

        c1, c2, c3 = st.sidebar.columns(3)
        with c1:
            if st.button(_lbl(half_p), key=f"btn_{cand['label']}_half", use_container_width=True):
                st.session_state[state_key] = half_p
                st.rerun()
        with c2:
            if st.button(_lbl(P_c), key=f"btn_{cand['label']}_P", use_container_width=True):
                st.session_state[state_key] = P_c
                st.rerun()
        with c3:
            if st.button(_lbl(two_p), key=f"btn_{cand['label']}_two", use_container_width=True):
                st.session_state[state_key] = two_p
                st.rerun()

        st.sidebar.caption(f"P/2={half_p:.4f}  ·  P={P_c:.4f}  ·  2P={two_p:.4f}")

    st.sidebar.markdown("---")


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


def _choose_window_days(arc_val, buffer_days, min_days, max_days):
    if arc_val is not None and np.isfinite(float(arc_val)) and float(arc_val) > 0:
        w = int(np.ceil(float(arc_val) + float(buffer_days)))
    else:
        w = 400
    return int(max(min_days, min(max_days, w)))


def bq_load_photometry_for_provid(provid, *, window_days, row_limit):
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
      AND obstime >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @win DAY)
    ORDER BY obstime
    LIMIT {row_limit}
    """
    params = [
        bigquery.ScalarQueryParameter("prov", "STRING", provid),
        bigquery.ScalarQueryParameter("win",  "INT64",  window_days),
    ]
    bq_meta = {"provid": provid, "source_table": source,
                "window_days": window_days, "row_limit": row_limit}

    try:
        dry = client.query(query, location=BQ_LOCATION,
                           job_config=bigquery.QueryJobConfig(
                               query_parameters=params, dry_run=True, use_query_cache=False))
        est = int(getattr(dry, "total_bytes_processed", 0) or 0)
        bq_meta.update({"dry_run_ok": True,
                         "estimated_bytes_human": bytes_to_human(est),
                         "estimated_cost_usd": est_usd_cost(est)})
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
                     "actual_cost_usd": est_usd_cost(actual),
                     "cache_hit": bool(getattr(job, "cache_hit", False)),
                     "job_id": getattr(job, "job_id", None),
                     "returned_rows": len(df),
                     "may_be_truncated": len(df) >= row_limit})
    return df, bq_meta


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


def plot_fold(ax, t_hr, mag, bands, P_hr, title, mag_label, two_cycles=False):
    phase = (t_hr / float(P_hr)) % 1.0
    for b in sorted(np.unique(bands).tolist()):
        m = bands == b
        ax.scatter(phase[m], mag[m], s=10, label=str(b))
        if two_cycles:
            ax.scatter(phase[m] + 1.0, mag[m], s=10)
    ax.invert_yaxis()
    ax.set_xlabel("Phase (0–1)" if not two_cycles else "Phase (0–2)")
    ax.set_ylabel(mag_label)
    ax.set_title(title)
    ax.set_xlim(0.0, 2.0 if two_cycles else 1.0)


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

# -------------------------
# Sidebar — Mode
# -------------------------
st.sidebar.markdown("## Mode")
mode = st.sidebar.radio("View", ["Asteroid Viewer", "Population Explorer"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("## BigQuery Controls")

row_limit    = st.sidebar.slider("Max rows per query",   1000, BQ_MAX_ROW_LIMIT, BQ_DEFAULT_ROW_LIMIT, 1000)
buffer_days  = st.sidebar.slider("Window buffer (days)", 0,    120,  30,  5)
min_window   = st.sidebar.slider("Min window (days)",    30,   400,  60, 10)
max_window   = st.sidebar.slider("Max window (days)",    100, 2000, 800, 50)


# ============================================================
# MODE 1: ASTEROID VIEWER
# ============================================================
if mode == "Asteroid Viewer":
    st.caption("Photometry queried from BigQuery, folded with on-the-fly geometry correction (Horizons).")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Asteroid")

    if "reliable_only" not in st.session_state:
        st.session_state["reliable_only"] = False   # 76 objects — show all by default

    q = st.sidebar.text_input("Search", value="", placeholder="E.g., 2025 ME15")

    df_pick = master.copy()
    if q.strip():
        df_pick = df_pick[df_pick[C_DESIG].astype(str).str.contains(q.strip(), case=False, na=False)]
    if st.session_state.get("reliable_only") and C_RELIABILITY in df_pick.columns:
        df_pick = df_pick[df_pick[C_RELIABILITY].astype(str).map(reliability_short) == "reliable"]

    df_pick = df_pick.sort_values(C_DESIG)
    designations = df_pick[C_DESIG].astype(str).tolist()

    if not designations:
        st.sidebar.warning("No asteroids match your current search.")
        st.stop()

    selected = st.sidebar.selectbox("Selected Asteroid", designations, index=0, key="selected_asteroid")
    st.sidebar.checkbox(f"Reliable only ({RELIABLE_COUNT})", key="reliable_only")

    row = master[master[C_DESIG].astype(str) == str(selected)]
    row = row.iloc[0].to_dict() if len(row) else {}
    rel = reliability_short(str(row.get(C_RELIABILITY, "")))

    P_adopt = float(row.get(C_PERIOD, np.nan))
    if not (np.isfinite(P_adopt) and P_adopt > 0):
        P_adopt = 5.0

    arc_val     = row.get(C_ARC, np.nan)
    window_days = _choose_window_days(arc_val, buffer_days, min_window, max_window)

    # ---- Fold Controls ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Fold Controls")

    if st.session_state.get("fold_period_for") != selected:
        st.session_state["fold_period"]     = float(P_adopt)
        st.session_state["fold_period_for"] = selected

    candidates = build_period_candidates(row)

    # Slider bounds: span all candidate harmonics
    all_bounds = [P_adopt]
    for cand in candidates:
        p = cand["period"]
        all_bounds.extend([p / 2.0, p, p * 2.0])
    lo = max(1e-6, min(all_bounds) * 0.9)
    hi = max(all_bounds) * 1.1

    P_calc = st.sidebar.slider(
        "Fold Period (hours)",
        min_value=float(lo),
        max_value=float(hi),
        value=float(st.session_state.get("fold_period", P_adopt)),
        step=float((hi - lo) / 600.0) if hi > lo else 1e-6,
        key="fold_period_slider",
    )
    st.session_state["fold_period"] = float(P_calc)

    if st.sidebar.button("↩ Reset To Adopted Period", use_container_width=True):
        st.session_state["fold_period"] = float(P_adopt)
        st.rerun()

    st.sidebar.markdown("---")
    render_period_candidate_ladder(
        candidates,
        current_fold_period=float(st.session_state.get("fold_period", P_adopt)),
        state_key="fold_period",
    )
    P_calc = float(st.session_state.get("fold_period", P_adopt))

    LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
    sel_bands_sidebar = st.sidebar.multiselect("Bands", LSST_BANDS, default=["g", "r", "i"])
    two_cycles = st.sidebar.checkbox("Show two cycles (0–2)", value=False)

    # ---- Main tabs ----
    tab_photo, tab_char = st.tabs(["Photometry", "Characterisation"])

    with tab_photo:
        # Header with reliability badge + 2P flag + human review warning
        prefer_2p = bool(row.get(C_PREFER_2P, False))
        review    = bool(row.get(C_REVIEW, False))

        header_parts = [
            f"### Fold Preview: **{selected}**",
            "&nbsp;&nbsp;•&nbsp;&nbsp;",
            reliability_html(rel),
        ]
        if prefer_2p:
            reason = str(row.get(C_PREFER_2P_R, "")) or "pipeline prefers 2P"
            header_parts += [
                "&nbsp;&nbsp;•&nbsp;&nbsp;",
                f'<span style="color:#3b82f6;font-weight:700;" title="{reason}">2P preferred</span>',
            ]
        if review:
            header_parts += [
                "&nbsp;&nbsp;•&nbsp;&nbsp;",
                '<span style="color:#ef4444;font-weight:700;">⚠ Needs review</span>',
            ]
        st.markdown(" ".join(header_parts), unsafe_allow_html=True)

        if prefer_2p and row.get(C_PREFER_2P_R):
            st.info(f"Pipeline 2P preference: *{row[C_PREFER_2P_R]}*")

        with st.spinner(f"Querying BigQuery (window={window_days}d, limit={row_limit}) ..."):
            df_raw, bq_meta = bq_load_photometry_for_provid(
                str(selected), window_days=window_days, row_limit=row_limit)

        with st.expander("BigQuery Cost & Query Diagnostics", expanded=False):
            st.json(bq_meta)

        if bq_meta.get("may_be_truncated"):
            st.warning(f"Returned {bq_meta['returned_rows']} rows — hit row limit. "
                       "Increase the slider in BigQuery Controls if needed.")

        if df_raw is None or len(df_raw) == 0:
            st.info("No photometry found in BigQuery for this asteroid under the current time window.")
            st.stop()

        df1 = make_df1_from_bq(df_raw)
        if len(df1) < 5:
            st.warning("Very few usable points after cleaning.")
            st.dataframe(df1.head(50), use_container_width=True)
            st.stop()

        with st.spinner("Running geometry correction (Horizons) ..."):
            try:
                df_geo, meta5 = geo_correct(df1, str(selected))
            except Exception as e:
                st.error("Geometry correction failed — using raw mags.")
                st.exception(e)
                df_geo = df1.copy()
                df_geo["mag_geo"] = df_geo["mag_geo_bandcenter"] = np.nan
                meta5 = {}

        if df_geo["mag_geo_bandcenter"].notna().sum() >= 5:
            mag_col, mag_label = "mag_geo_bandcenter", "mag_geo_bandcenter (corrected, band-centred)"
        elif df_geo["mag_geo"].notna().sum() >= 5:
            mag_col, mag_label = "mag_geo", "mag_geo (corrected)"
        else:
            mag_col, mag_label = "mag", "mag (raw)"

        df_geo["band"] = df_geo["band"].map(normalize_lsst_band)
        avail     = set(df_geo["band"].dropna().astype(str).unique())
        sel_bands = [b for b in sel_bands_sidebar if b in avail] or sorted(avail)

        dfp = df_geo[df_geo["band"].isin(sel_bands)].dropna(subset=["t_hr", mag_col, "band"])
        n_nights = resolve_nights(dfp)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Adopted Period (h)",    format_float(P_adopt, 6))
        s2.metric("Fold Period (h)",        format_float(P_calc,  6))
        s3.metric("Observations returned", f"{len(dfp):,}")
        s4.metric("Nights",               "—" if n_nights is None else str(n_nights))

        st.caption(f"Folding: **{mag_label}** · window_days={window_days}")

        t_hr  = dfp["t_hr"].to_numpy(float)
        t_day = dfp["t_day"].to_numpy(float)
        mag   = pd.to_numeric(dfp[mag_col], errors="coerce").to_numpy(float)
        bands = dfp["band"].to_numpy(str)

        P_half = 0.5 * P_calc
        P_two  = 2.0 * P_calc

        st.markdown("#### Three-Panel Fold (P/2 • P • 2P)")
        for col, P_hr, title in zip(
            st.columns(3),
            [P_half, P_calc, P_two],
            [f"P/2 = {P_half:.6f} h", f"P = {P_calc:.6f} h", f"2P = {P_two:.6f} h"],
        ):
            with col:
                fig, ax = plt.subplots(figsize=(5.2, 3.6))
                plot_fold(ax, t_hr, mag, bands, P_hr, title, mag_label, two_cycles)
                ax.legend(fontsize=7)
                st.pyplot(fig, clear_figure=True)

        st.markdown("#### Magnitude vs Time")
        fig, ax = plt.subplots(figsize=(10.5, 3.6))
        for b in sorted(np.unique(bands).tolist()):
            m = bands == b
            ax.scatter(t_day[m], mag[m], s=10, label=b)
        ax.invert_yaxis()
        ax.set_xlabel("Days Since First Observation")
        ax.set_ylabel(mag_label)
        ax.set_title("Magnitude vs Time")
        ax.legend(fontsize=8, ncol=6)
        st.pyplot(fig, clear_figure=True)

    with tab_char:
        st.markdown(
            f"### Characterisation: **{selected}** &nbsp;&nbsp;•&nbsp;&nbsp; {reliability_html(rel)}",
            unsafe_allow_html=True,
        )
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
                       f"use the candidate ladder to fold at this value.")

        # Colour indices
        st.markdown("#### Colour Indices")
        ci1, ci2, ci3 = st.columns(3)
        ci1.metric("g − r", format_float(row.get(C_GR), 4))
        ci2.metric("g − i", format_float(row.get(C_GI), 4))
        ci3.metric("r − i", format_float(row.get(C_RI), 4))

        # Period candidate table
        if candidates:
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
    st.caption("All 76 asteroids · 2025 cohort · pipeline v54")

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

    with col_l:
        st.markdown("#### Period Distribution")
        periods = df_f[C_PERIOD].dropna().to_numpy(float)
        periods = periods[np.isfinite(periods)]
        if len(periods):
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.hist(periods, bins=30)
            ax.set_xlabel("Adopted Period (hours)")
            ax.set_ylabel("Count")
            ax.set_title("Rotation Period Histogram")
            st.pyplot(fig, clear_figure=True)

    with col_r:
        st.markdown("#### Period vs Amplitude")
        x = df_f[C_PERIOD].to_numpy(float)
        y = df_f[C_AMPLITUDE].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum():
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            # colour by reliability
            cmap = {"reliable": "#22c55e", "ambiguous": "#f59e0b", "insufficient": "#ef4444"}
            for rel_val, colour in cmap.items():
                m = ok & (df_f[C_RELIABILITY].astype(str) == rel_val).to_numpy()
                if m.sum():
                    ax.scatter(x[m], y[m], s=14, color=colour, label=rel_val, alpha=0.85)
            ax.set_xlabel("Adopted Period (hours)")
            ax.set_ylabel("Amplitude (mag)")
            ax.set_title("Period vs Amplitude")
            ax.legend(fontsize=8)
            st.pyplot(fig, clear_figure=True)

    # Bootstrap frac scatter (new — uses richer columns)
    if C_BOOT_BASE in df_f.columns and C_BOOT_2P in df_f.columns:
        st.markdown("#### Bootstrap Fraction: Base vs 2P")
        xb = df_f[C_BOOT_BASE].to_numpy(float)
        yb = df_f[C_BOOT_2P].to_numpy(float)
        ok2 = np.isfinite(xb) & np.isfinite(yb)
        if ok2.sum():
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.scatter(xb[ok2], yb[ok2], s=14, alpha=0.8)
            ax.axline((0, 0), slope=1, color="gray", lw=0.8, linestyle="--")
            ax.set_xlabel("Bootstrap frac (base period)")
            ax.set_ylabel("Bootstrap frac (2P)")
            ax.set_title("Base vs 2P Bootstrap Fraction")
            st.pyplot(fig, clear_figure=True)

    # Master table
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
