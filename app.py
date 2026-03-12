# app.py — Iteration 1
# ==========================================================
# ATLAST Asteroid Rotation Dashboard
#
# Iteration 1 changes vs baseline:
#   - Period Candidate Ladder replaces the bare fold-period slider
#   - Shows LS Peak, Adopted, and 2P Candidate from master CSV
#   - Each candidate has P/2 · P · 2P one-click buttons
#   - Slider still present for fine-tuning after a button click
#   - ΔBIC shown next to 2P candidate so users know its confidence
#   - "Additional Periods" section for any extra pipeline candidates
#     (populated from "LS peak period (hr)" alt harmonics if present)
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
# Local file(s)
# -------------------------
MASTER_PATH = Path("master_results_clean.csv")  # required


# -------------------------
# BigQuery config
# -------------------------
BQ_PROJECT   = "lsst-484623"
BQ_LOCATION  = "US"
BQ_DATASET   = "atlast_photometry"
BQ_TABLE     = "public_obs_x05"

BQ_DEFAULT_ROW_LIMIT = 20000
BQ_MAX_ROW_LIMIT     = 200000

BQ_USD_PER_TB = 5.0

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
# BigQuery client
# -------------------------
def get_bq_client():
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

    try:
        _ = client.query("SELECT 1", location=BQ_LOCATION).result()
    except Exception as e:
        st.error("BigQuery client smoke-test failed (cannot run SELECT 1).")
        st.exception(e)
        st.stop()

    st.session_state["_bq_client"] = client
    return client


def _bq_job_debug_dict(job):
    return {
        "job_id": getattr(job, "job_id", None),
        "project": getattr(job, "project", None),
        "location": BQ_LOCATION,
        "state": getattr(job, "state", None),
        "errors": getattr(job, "errors", None),
        "error_result": getattr(job, "error_result", None),
        "cache_hit": bool(getattr(job, "cache_hit", False)),
        "total_bytes_processed": int(getattr(job, "total_bytes_processed", 0) or 0),
    }


# -------------------------
# Band normalization
# -------------------------
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


# -------------------------
# Load master
# -------------------------
def load_master(path: Path) -> pd.DataFrame:
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


def make_df1_from_bq(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["obstime_dt"] = pd.to_datetime(df["obstime"], errors="coerce", utc=True)
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["rmsmag"] = pd.to_numeric(df.get("rmsmag", np.nan), errors="coerce")
    df["band"] = df.get("band", "x").map(normalize_lsst_band)
    df = df.dropna(subset=["obstime_dt", "mag", "band"]).reset_index(drop=True)
    if len(df) == 0:
        return df
    df = df.sort_values("obstime_dt")
    t0 = df["obstime_dt"].min()
    df["t_hr"] = (df["obstime_dt"] - t0).dt.total_seconds() / 3600.0
    df["t_day"] = (df["obstime_dt"] - t0).dt.total_seconds() / 86400.0
    df["night_utc"] = df["obstime_dt"].dt.strftime("%Y-%m-%d")
    return df


# -------------------------
# BigQuery photometry fetch
# -------------------------
def _choose_window_days(arc_days_value, buffer_days: int, min_days: int, max_days: int) -> int:
    if arc_days_value is not None and np.isfinite(float(arc_days_value)) and float(arc_days_value) > 0:
        w = int(np.ceil(float(arc_days_value) + float(buffer_days)))
    else:
        w = 400
    w = max(int(min_days), min(int(max_days), int(w)))
    return int(w)


def bq_load_photometry_for_provid(provid, *, window_days, row_limit):
    client = get_bq_client()
    source_table = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    row_limit = int(max(1, min(int(row_limit), int(BQ_MAX_ROW_LIMIT))))

    query = f"""
    SELECT
      provid,
      obstime,
      band,
      SAFE_CAST(mag AS FLOAT64)    AS mag,
      SAFE_CAST(rmsmag AS FLOAT64) AS rmsmag
    FROM `{source_table}`
    WHERE provid = @prov
      AND mag IS NOT NULL
      AND obstime >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @win DAY)
    ORDER BY obstime
    LIMIT {row_limit}
    """

    params = [
        bigquery.ScalarQueryParameter("prov", "STRING", provid),
        bigquery.ScalarQueryParameter("win", "INT64", int(window_days)),
    ]

    bq_meta = {
        "provid": provid,
        "source_table": source_table,
        "location": BQ_LOCATION,
        "window_days": int(window_days),
        "row_limit": int(row_limit),
    }

    try:
        dry_cfg = bigquery.QueryJobConfig(query_parameters=params, dry_run=True, use_query_cache=False)
        dry_job = client.query(query, job_config=dry_cfg, location=BQ_LOCATION)
        est_bytes = int(getattr(dry_job, "total_bytes_processed", 0) or 0)
        bq_meta.update({
            "dry_run_ok": True,
            "estimated_bytes_processed": est_bytes if est_bytes else None,
            "estimated_bytes_human": bytes_to_human(est_bytes) if est_bytes else "—",
            "estimated_est_cost_usd": est_usd_cost(est_bytes) if est_bytes else None,
        })
    except Exception as e:
        bq_meta.update({"dry_run_ok": False})
        st.error("BigQuery DRY RUN failed.")
        st.json(bq_meta)
        st.exception(e)
        raise

    run_cfg = bigquery.QueryJobConfig(query_parameters=params, use_query_cache=True)
    job = client.query(query, job_config=run_cfg, location=BQ_LOCATION)

    try:
        df = job.to_dataframe(create_bqstorage_client=False)
    except (Forbidden, BadRequest, NotFound) as e:
        bq_meta.update({"job_debug": _bq_job_debug_dict(job)})
        st.error("BigQuery query failed.")
        st.json(bq_meta)
        st.exception(e)
        raise

    actual_bytes = int(getattr(job, "total_bytes_processed", 0) or 0)
    bq_meta.update({
        "actual_bytes_processed": actual_bytes if actual_bytes else None,
        "actual_bytes_human": bytes_to_human(actual_bytes) if actual_bytes else "—",
        "actual_est_cost_usd": est_usd_cost(actual_bytes) if actual_bytes else None,
        "cache_hit": bool(getattr(job, "cache_hit", False)),
        "job_id": getattr(job, "job_id", None),
    })

    bq_meta["returned_rows"] = int(len(df))
    bq_meta["may_be_truncated"] = bool(len(df) >= row_limit)

    return df, bq_meta


# ======================================================================
# Geometry correction (Horizons)
# ======================================================================
def step5_geometry_horizons_range(
    df1,
    *,
    PROVID,
    OUTDIR,
    HORIZONS_LOCATION="X05",
    G_DEFAULT=0.15,
    STEP_MINUTES=10,
    PAD_MINUTES=10,
    FAIL_ON_UNMATCHED=False,
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
        raise ValueError(f"STEP 5: df1 missing columns: {missing}")

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

    def query_horizons_range_smallbody(desig, start_utc, stop_utc, step_minutes=10, location="X05"):
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
                raise KeyError(f"Horizons response missing '{k}'.")
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
        eph_b = query_horizons_range_smallbody(
            PROVID,
            tmin.strftime("%Y-%m-%d %H:%M"),
            tmax.strftime("%Y-%m-%d %H:%M"),
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
        "night_key": str("night_id" if "night_id" in dfG.columns else "night_utc"),
        "n_blocks": int(len(blocks)),
        "n_obs": int(n_total),
        "n_matched": int(matched),
        "n_unmatched": int(n_unmatched),
    }

    return dfM, step5_meta


def geo_correct(df1, provid):
    df_geo, meta = step5_geometry_horizons_range(
        df1,
        PROVID=provid,
        OUTDIR=".",
        HORIZONS_LOCATION=HORIZONS_LOCATION,
        G_DEFAULT=HG_G_DEFAULT,
        STEP_MINUTES=10,
        PAD_MINUTES=10,
        FAIL_ON_UNMATCHED=False,
        save_tables=False,
        save_plots=False,
        show_plots=False,
        verbose=False,
    )
    return df_geo, meta


# ======================================================================
# NEW (Iteration 1): Period Candidate Ladder helper
# ======================================================================

def _safe_period(val) -> float | None:
    """Return a positive finite float or None."""
    try:
        v = float(val)
        if np.isfinite(v) and v > 0:
            return round(v, 6)
    except Exception:
        pass
    return None


def build_period_candidates(row: dict) -> list[dict]:
    """
    Build an ordered list of period candidates from a master CSV row.
    Each entry: {"label": str, "period": float, "note": str | None}

    Priority order:
      1. Adopted period  (always first)
      2. LS peak period  (if different from adopted)
      3. 2P candidate    (if present, with ΔBIC note)

    Deduplication: skip any period within 0.1% of an already-included period.
    """
    candidates = []
    seen: list[float] = []

    def _is_dup(p: float) -> bool:
        return any(abs(p - s) / max(s, 1e-9) < 0.001 for s in seen)

    def _add(label: str, period: float | None, note: str | None = None):
        if period is None:
            return
        if _is_dup(period):
            return
        seen.append(period)
        candidates.append({"label": label, "period": period, "note": note})

    _add("Adopted", _safe_period(row.get("Adopted period (hr)")))
    _add("LS Peak", _safe_period(row.get("LS peak period (hr)")))

    p2 = _safe_period(row.get("2P candidate (hr)"))
    dbic = row.get("ΔBIC(2P−P)", None)
    dbic_str = None
    if dbic is not None:
        try:
            dbic_str = f"ΔBIC = {float(dbic):.1f}"
        except Exception:
            pass
    _add("2P Candidate", p2, note=dbic_str)

    return candidates


def render_period_candidate_ladder(
    candidates: list[dict],
    current_fold_period: float,
    state_key: str = "fold_period",
):
    """
    Render the period candidate ladder inside the sidebar.

    For each candidate period P_c, show three buttons: P/2 · P · 2P.
    Clicking any button sets st.session_state[state_key] and triggers st.rerun().
    The active button (within 0.05% of current_fold_period) is visually highlighted
    via an asterisk in the label — Streamlit doesn't allow per-button colour in sidebar.
    """
    if not candidates:
        return

    st.sidebar.markdown("**Period Candidates**")
    st.sidebar.caption("Click P/2 · P · 2P to fold at that period")

    TOLERANCE = 0.0005  # 0.05% match → flag as active

    def _is_active(p: float) -> bool:
        if p <= 0:
            return False
        return abs(p - current_fold_period) / max(current_fold_period, 1e-9) < TOLERANCE

    def _btn_label(p: float) -> str:
        marker = " ✦" if _is_active(p) else ""
        return f"{p:.4f} h{marker}"

    for cand in candidates:
        P_c = cand["period"]
        label = cand["label"]
        note = cand.get("note")

        # Compact header line
        header = f"**{label}** — {P_c:.6f} h"
        if note:
            header += f"  `{note}`"
        st.sidebar.markdown(header)

        half_p = round(P_c / 2.0, 6)
        two_p  = round(P_c * 2.0, 6)

        col1, col2, col3 = st.sidebar.columns(3)

        with col1:
            if st.button(_btn_label(half_p), key=f"btn_{label}_half", use_container_width=True):
                st.session_state[state_key] = half_p
                st.rerun()

        with col2:
            if st.button(_btn_label(P_c), key=f"btn_{label}_P", use_container_width=True):
                st.session_state[state_key] = P_c
                st.rerun()

        with col3:
            if st.button(_btn_label(two_p), key=f"btn_{label}_two", use_container_width=True):
                st.session_state[state_key] = two_p
                st.rerun()

        st.sidebar.caption(f"↑ P/2 = {half_p:.4f} h  ·  P = {P_c:.4f} h  ·  2P = {two_p:.4f} h")

    st.sidebar.markdown("---")


# ======================================================================
# App start
# ======================================================================
st.markdown("## ATLAST Asteroid Rotation Dashboard")

if not MASTER_PATH.exists():
    st.error(f"Missing required file: {MASTER_PATH}")
    st.stop()

master = load_master(MASTER_PATH)

if "Designation" not in master.columns:
    for c in ["provid", "PROVID", "designation", "name", "object_id"]:
        if c in master.columns:
            master = master.rename(columns={c: "Designation"})
            break

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

RELIABLE_COUNT = int(
    (master.get("Reliability", pd.Series([], dtype=str))
     .astype(str).str.lower() == "reliable").sum()
) if "Reliability" in master.columns else 0

# -------------------------
# Sidebar — Mode
# -------------------------
st.sidebar.markdown("## Mode")
mode = st.sidebar.radio("View", ["Asteroid Viewer", "Population Explorer"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("## BigQuery Controls")

row_limit = st.sidebar.slider("Max rows per asteroid query", 1000, BQ_MAX_ROW_LIMIT, BQ_DEFAULT_ROW_LIMIT, 1000)
buffer_days = st.sidebar.slider("Window buffer (days)", 0, 120, 30, 5)
min_window  = st.sidebar.slider("Min window (days)", 30, 400, 60, 10)
max_window  = st.sidebar.slider("Max window (days)", 100, 2000, 800, 50)


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

    df_pick = master.copy()
    if q.strip():
        df_pick = df_pick[df_pick["Designation"].astype(str).str.contains(q.strip(), case=False, na=False)]

    if bool(st.session_state.get("reliable_only", False)) and ("Reliability" in df_pick.columns):
        rel_s = df_pick["Reliability"].astype(str).map(reliability_short)
        df_pick = df_pick[rel_s == "reliable"]

    df_pick = df_pick.sort_values("Designation")
    designations = df_pick["Designation"].astype(str).tolist()

    if not designations:
        st.sidebar.warning(
            "No reliable-period asteroids match your current search."
            if bool(st.session_state.get("reliable_only", False))
            else "No asteroids match your current search."
        )
        st.stop()

    selected = st.sidebar.selectbox("Selected Asteroid", options=designations, index=0, key="selected_asteroid")
    st.sidebar.checkbox(f"Reliable Periods only ({RELIABLE_COUNT:,})", key="reliable_only")

    row = master[master["Designation"].astype(str) == str(selected)]
    row = row.iloc[0].to_dict() if len(row) else {}
    rel = reliability_short(str(row.get("Reliability", "")))

    P_adopt = float(row.get("Adopted period (hr)", np.nan))
    if not (np.isfinite(P_adopt) and P_adopt > 0):
        P_adopt = 5.0

    arc_days   = row.get("Arc (days)", np.nan)
    window_days = _choose_window_days(arc_days, buffer_days=buffer_days, min_days=min_window, max_days=max_window)

    # ------------------------------------------------------------------
    # Sidebar — Fold Controls (Iteration 1: Period Candidate Ladder)
    # ------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Fold Controls")

    # Reset fold period when asteroid changes
    if st.session_state.get("fold_period_for") != selected:
        st.session_state["fold_period"] = float(P_adopt)
        st.session_state["fold_period_for"] = selected

    # Build candidates from master row
    candidates = build_period_candidates(row)

    # Determine slider bounds: cover all candidate harmonics + adopted
    all_periods_for_bounds = [P_adopt]
    for cand in candidates:
        p = cand["period"]
        all_periods_for_bounds.extend([p / 2.0, p, p * 2.0])
    lo = max(1e-6, min(all_periods_for_bounds) * 0.9)
    hi = max(all_periods_for_bounds) * 1.1

    # Fine-tune slider (updates live; candidate buttons will override it)
    P_calc = st.sidebar.slider(
        "Fold Period (hours)",
        min_value=float(lo),
        max_value=float(hi),
        value=float(st.session_state.get("fold_period", P_adopt)),
        step=float((hi - lo) / 600.0) if hi > lo else 1e-6,
        key="fold_period_slider",
    )
    # Keep session state in sync with slider
    st.session_state["fold_period"] = float(P_calc)

    if st.sidebar.button("↩ Reset To Adopted Period", use_container_width=True):
        st.session_state["fold_period"] = float(P_adopt)
        st.rerun()

    st.sidebar.markdown("---")

    # Period Candidate Ladder
    render_period_candidate_ladder(
        candidates,
        current_fold_period=float(st.session_state.get("fold_period", P_adopt)),
        state_key="fold_period",
    )
    # Read back after possible button rerun
    P_calc = float(st.session_state.get("fold_period", P_adopt))

    LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
    sel_bands_sidebar = st.sidebar.multiselect("Bands", options=LSST_BANDS, default=["g", "r", "i"])
    two_cycles = st.sidebar.checkbox("Show two cycles (0–2)", value=False)

    # ------------------------------------------------------------------
    # Main content
    # ------------------------------------------------------------------
    tab_photo, tab_char = st.tabs(["Photometry", "Characterisation"])

    with tab_photo:
        st.markdown(
            f"### Geometry-Corrected Fold Preview: **{selected}** &nbsp;&nbsp;•&nbsp;&nbsp; {reliability_html(rel)}",
            unsafe_allow_html=True,
        )

        with st.spinner(f"Querying BigQuery photometry (window_days={window_days}, limit={row_limit}) ..."):
            df_raw, bq_meta = bq_load_photometry_for_provid(str(selected), window_days=window_days, row_limit=row_limit)

        with st.expander("BigQuery Cost & Query Diagnostics", expanded=False):
            st.json(bq_meta)

        if bq_meta.get("may_be_truncated", False):
            st.warning(
                f"Returned {bq_meta.get('returned_rows')} rows and hit the row limit ({row_limit}). "
                "Increase the row limit in the sidebar if you expect more data."
            )

        if df_raw is None or len(df_raw) == 0:
            st.info("No photometry rows found in BigQuery for this asteroid under the current time window.")
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
                st.error("Geometry correction failed (Horizons). Showing raw mags instead.")
                st.exception(e)
                df_geo = df1.copy()
                df_geo["mag_geo"] = np.nan
                df_geo["mag_geo_bandcenter"] = np.nan
                meta5 = {}

        if "mag_geo_bandcenter" in df_geo.columns and df_geo["mag_geo_bandcenter"].notna().sum() >= 5:
            mag_col   = "mag_geo_bandcenter"
            mag_label = "mag_geo_bandcenter (corrected, band-centered)"
        elif "mag_geo" in df_geo.columns and df_geo["mag_geo"].notna().sum() >= 5:
            mag_col   = "mag_geo"
            mag_label = "mag_geo (corrected)"
        else:
            mag_col   = "mag"
            mag_label = "mag (raw)"

        df_geo["band"] = df_geo["band"].map(normalize_lsst_band)
        avail = set(df_geo["band"].dropna().astype(str).unique().tolist())
        sel_bands = [b for b in sel_bands_sidebar if b in avail]
        if not sel_bands:
            sel_bands = sorted(list(avail))

        dfp = df_geo[df_geo["band"].isin(sel_bands)].copy()
        dfp = dfp.dropna(subset=["t_hr", mag_col, "band"])
        n_nights = resolve_nights(dfp)

        # Metrics row
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Adopted Period (hours)", format_float(row.get("Adopted period (hr)", np.nan), 6))
        s2.metric("Fold Period (hours)", format_float(P_calc, 6))
        s3.metric("Observations (returned)", f"{len(dfp):,}")
        s4.metric("Nights (photometry)", "—" if n_nights is None else str(int(n_nights)))

        st.caption(f"Folding uses: **{mag_label}**  |  Query window_days={window_days}")

        t_hr   = dfp["t_hr"].to_numpy(float)
        t_day  = dfp["t_day"].to_numpy(float)
        mag    = pd.to_numeric(dfp[mag_col], errors="coerce").to_numpy(float)
        bands  = dfp["band"].to_numpy(str)

        P_half = 0.5 * float(P_calc)
        P_two  = 2.0 * float(P_calc)

        # ------------------------------------------------------------------
        # Three-panel fold — unchanged from baseline, but now the fold period
        # can be driven by the candidate ladder in the sidebar
        # ------------------------------------------------------------------
        st.markdown("#### Three-Panel Fold (P/2 • P • 2P)")
        cols = st.columns(3)
        periods = [P_half, float(P_calc), P_two]
        titles  = [
            f"P/2 = {P_half:.6f} h",
            f"P = {float(P_calc):.6f} h",
            f"2P = {P_two:.6f} h",
        ]

        for col, P_hr, title in zip(cols, periods, titles):
            with col:
                fig, ax = plt.subplots(figsize=(5.2, 3.6))
                plot_fold(ax, t_hr=t_hr, mag=mag, bands=bands,
                          P_hr=P_hr, title=title, mag_label=mag_label,
                          two_cycles=two_cycles)
                ax.legend(fontsize=7)
                st.pyplot(fig, clear_figure=True)

        # ------------------------------------------------------------------
        # Magnitude vs Time
        # ------------------------------------------------------------------
        st.markdown("#### Magnitude vs Time")
        fig, ax = plt.subplots(figsize=(10.5, 3.6))
        for b in sorted(np.unique(bands).tolist()):
            m = (bands == b)
            ax.scatter(t_day[m], mag[m], s=10, label=b)
        ax.invert_yaxis()
        ax.set_xlabel("Days Since First Observation")
        ax.set_ylabel(mag_label)
        ax.set_title("Magnitude vs Time")
        ax.legend(fontsize=8, ncol=6)
        st.pyplot(fig, clear_figure=True)

    # ------------------------------------------------------------------
    # Characterisation tab
    # ------------------------------------------------------------------
    with tab_char:
        st.markdown(
            f"### Characterisation: **{selected}** &nbsp;&nbsp;•&nbsp;&nbsp; {reliability_html(rel)}",
            unsafe_allow_html=True,
        )
        st.caption("All values from master_results_clean.csv (Step 13 Summary Exports).")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Adopted Period (hours)", format_float(row.get("Adopted period (hr)", np.nan), 6))
        k2.metric("LS Peak Period (hours)", format_float(row.get("LS peak period (hr)", np.nan), 6))
        k3.metric("Adopted K", "—" if pd.isna(row.get("Adopted K", np.nan)) else str(int(row.get("Adopted K"))))
        k4.metric("Amplitude (Mag)", format_float(row.get("Amplitude (Fourier)", np.nan), 3))

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Axial Elongation", format_float(row.get("Axial Elongation", np.nan), 3))
        b2.metric("2P Candidate (hours)", format_float(row.get("2P candidate (hr)", np.nan), 6))
        b3.metric("ΔBIC (2P − P)", format_float(row.get("ΔBIC(2P−P)", np.nan), 2))
        b4.metric("Bootstrap Top Frac", format_float(row.get("Bootstrap top_frac", np.nan), 3))

        st.markdown("#### Color Indices")
        c1, c2, c3 = st.columns(3)
        c1.metric("g − r", format_float(row.get("g - r", np.nan), 4))
        c2.metric("g − i", format_float(row.get("g - i", np.nan), 4))
        c3.metric("r − i", format_float(row.get("r - i", np.nan), 4))

        # Period candidate summary table (new in Iteration 1)
        if candidates:
            st.markdown("#### Period Candidate Summary")
            rows_table = []
            for cand in candidates:
                p = cand["period"]
                rows_table.append({
                    "Source": cand["label"],
                    "Period (h)": f"{p:.6f}",
                    "P/2 (h)": f"{p/2:.6f}",
                    "2P (h)": f"{p*2:.6f}",
                    "Note": cand.get("note") or "—",
                })
            st.dataframe(pd.DataFrame(rows_table), use_container_width=True, hide_index=True)


# ==========================================================
# MODE 2: POPULATION EXPLORER  (unchanged from baseline)
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
        "Adopted Period Range (hours)",
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

    st.sidebar.caption(f"{len(df_f):,} asteroids match filters")

    st.markdown("### Population Overview (Filtered)")
    if len(df_f) == 0:
        st.warning("No asteroids match your current population filters.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Asteroids (Filtered)", f"{len(df_f):,}")
    if "Reliability" in df_f.columns:
        c2.metric("Reliable",     f"{int((df_f['Reliability'].astype(str) == 'reliable').sum()):,}")
        c3.metric("Ambiguous",    f"{int((df_f['Reliability'].astype(str) == 'ambiguous').sum()):,}")
        c4.metric("Insufficient", f"{int((df_f['Reliability'].astype(str) == 'insufficient').sum()):,}")
    else:
        c2.metric("Reliable", "—"); c3.metric("Ambiguous", "—"); c4.metric("Insufficient", "—")

    if "Adopted period (hr)" in df_f.columns and "Amplitude (Fourier)" in df_f.columns:
        st.markdown("#### Rotation Period vs Amplitude")
        x = df_f["Adopted period (hr)"].to_numpy(float)
        y = df_f["Amplitude (Fourier)"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 0:
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.scatter(x[m], y[m], s=10)
            ax.set_xlabel("Adopted Period (hours)")
            ax.set_ylabel("Amplitude (Fourier, Mag)")
            ax.set_title("Rotation Period vs Amplitude")
            st.pyplot(fig, clear_figure=True)

    if "Adopted period (hr)" in df_f.columns:
        st.markdown("#### Adopted Period Distribution")
        periods = df_f["Adopted period (hr)"].to_numpy(float)
        periods = periods[np.isfinite(periods)]
        if len(periods) > 0:
            fig, ax = plt.subplots(figsize=(8.5, 4.0))
            ax.hist(periods, bins=50)
            ax.set_xlabel("Adopted Period (hours)")
            ax.set_ylabel("Count")
            ax.set_title("Adopted Period Histogram")
            st.pyplot(fig, clear_figure=True)

    st.markdown("#### Master Table (Filtered)")
    show_cols = [
        "Designation", "Adopted period (hr)", "LS peak period (hr)",
        "2P candidate (hr)", "ΔBIC(2P−P)",
        "Amplitude (Fourier)", "Axial Elongation",
        "Reliability", "Bootstrap top_frac", "Number of Observations", "Arc (days)",
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
