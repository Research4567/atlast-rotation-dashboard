# ==========================================================
# ATLAST Rotation Dashboard (DEMO)
# - Master table from local CSV
# - Photometry fetched ON DEMAND from BigQuery (minimal cols)
# - Geometry correction computed ON THE FLY (JPL Horizons)
# - Coarse Horizons grid (demo mode: 10-min step)
# ==========================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from google.cloud import bigquery
from google.oauth2 import service_account

from astropy.time import Time
from astroquery.jplhorizons import Horizons


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="ATLAST Asteroid Rotation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

MASTER_PATH = Path("master_results_clean.csv")

# BigQuery
BQ_PROJECT = "lsst-484623"
BQ_DATASET = "asteroid_institute_mpc_replica"
BQ_TABLE   = "public_obs_sbn"
BQ_STN     = "X05"
BQ_ROW_LIMIT = 20000

# Horizons
HORIZONS_LOCATION = "X05"
HG_G_DEFAULT = 0.15


# ==========================================================
# BIGQUERY AUTH (STREAMLIT CLOUD SAFE) — HARD FAIL IF MISSING
# ==========================================================
@st.cache_resource
def get_bq_client() -> bigquery.Client:
    if "gcp_service_account" not in st.secrets:
        st.error(
            "Missing Streamlit secret: [gcp_service_account]. "
            "Streamlit Cloud → App → Settings → Secrets."
        )
        st.stop()

    sa = dict(st.secrets["gcp_service_account"])
    if ("client_email" not in sa) or ("private_key" not in sa):
        st.error(
            "Your [gcp_service_account] secret is present but incomplete. "
            "It must include at least client_email and private_key."
        )
        st.stop()

    creds = service_account.Credentials.from_service_account_info(sa)
    return bigquery.Client(project=BQ_PROJECT, credentials=creds)


# ==========================================================
# BIGQUERY PHOTOMETRY LOADER
# ==========================================================
@st.cache_data(show_spinner=False, ttl=3600)
def bq_load_photometry_for_provid(provid: str) -> pd.DataFrame:
    client = get_bq_client()

    q = f"""
    SELECT
      provid,
      obstime,
      band,
      SAFE_CAST(mag AS FLOAT64)    AS mag,
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

    return client.query(q, job_config=job_config).to_dataframe()


# ==========================================================
# STEP 5 — GEOMETRY CORRECTION (COARSE DEMO VERSION)
# ==========================================================
def step5_geometry_horizons_range(
    df1: pd.DataFrame,
    *,
    PROVID: str,
    STEP_MINUTES: int = 10,   # DEMO
    PAD_MINUTES: int = 10,    # DEMO
) -> pd.DataFrame:
    if df1 is None or len(df1) == 0:
        raise ValueError("Empty photometry.")

    df = df1.copy()

    # Ensure UTC datetime
    df["obstime_dt"] = pd.to_datetime(df["obstime_dt"], errors="coerce", utc=True)
    df = df.dropna(subset=["obstime_dt"]).sort_values("obstime_dt").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid obstime_dt after parsing.")

    # Observation JD
    t_utc = Time(df["obstime_dt"].dt.to_pydatetime(), scale="utc")
    df["jd_utc_obs"] = t_utc.jd.astype(float)

    # Query Horizons per UTC date block
    blocks = df["obstime_dt"].dt.date.unique()
    eph_parts: list[pd.DataFrame] = []

    for block in blocks:
        sub = df[df["obstime_dt"].dt.date == block]
        tmin = sub["obstime_dt"].min() - pd.Timedelta(minutes=PAD_MINUTES)
        tmax = sub["obstime_dt"].max() + pd.Timedelta(minutes=PAD_MINUTES)

        obj = Horizons(
            id=PROVID,
            id_type="smallbody",
            location=HORIZONS_LOCATION,
            epochs={
                "start": tmin.strftime("%Y-%m-%d %H:%M"),
                "stop":  tmax.strftime("%Y-%m-%d %H:%M"),
                "step":  f"{int(STEP_MINUTES)}m",
            },
        )

        eph = obj.ephemerides().to_pandas()

        eph_parts.append(pd.DataFrame({
            "jd_utc_eph": eph["datetime_jd"].astype(float),
            "r": eph["r"].astype(float),
            "delta": eph["delta"].astype(float),
            "alpha": eph["alpha"].astype(float),
        }))

    eph_df = (
        pd.concat(eph_parts, ignore_index=True)
        .drop_duplicates(subset=["jd_utc_eph"])
        .sort_values("jd_utc_eph")
        .reset_index(drop=True)
    )

    # Merge nearest ephemeris row
    df = df.sort_values("jd_utc_obs").reset_index(drop=True)
    eph_df = eph_df.sort_values("jd_utc_eph").reset_index(drop=True)

    tol_days = max(2e-3, (STEP_MINUTES / 1440.0) * 1.2)

    df = pd.merge_asof(
        df,
        eph_df,
        left_on="jd_utc_obs",
        right_on="jd_utc_eph",
        direction="nearest",
        tolerance=tol_days,
    )

    # HG phase function
    def phase_HG(alpha_deg: np.ndarray, G: float = 0.15) -> np.ndarray:
        a = np.deg2rad(alpha_deg)
        phi1 = np.exp(-3.33 * np.power(np.tan(a / 2.0), 0.63))
        phi2 = np.exp(-1.87 * np.power(np.tan(a / 2.0), 1.22))
        p = (1.0 - G) * phi1 + G * phi2
        p = np.clip(p, 1e-12, None)
        return -2.5 * np.log10(p)

    df["dist_term"] = 5.0 * np.log10(df["r"] * df["delta"])
    df["phase_term"] = phase_HG(df["alpha"].to_numpy(float), G=HG_G_DEFAULT)
    df["mag_geo"] = df["mag"].to_numpy(float) - df["dist_term"].to_numpy(float) - df["phase_term"].to_numpy(float)

    # Band-center
    df["band"] = df["band"].astype(str).str.strip().str.lower()
    df["mag_geo_bandcenter"] = df["mag_geo"] - df.groupby("band")["mag_geo"].transform("median")

    return df


# ==========================================================
# LOAD MASTER
# ==========================================================
if not MASTER_PATH.exists():
    st.error("Missing required file: master_results_clean.csv")
    st.stop()

master = pd.read_csv(MASTER_PATH)

if "Designation" not in master.columns:
    st.error("master_results_clean.csv must contain a 'Designation' column.")
    st.stop()

st.title("ATLAST Asteroid Rotation Dashboard (Demo Fold Tool)")

selected = st.selectbox("Select Asteroid", master["Designation"].astype(str))

# Adopted period
if "Adopted period (hr)" in master.columns:
    row = master[master["Designation"].astype(str) == str(selected)]
    if len(row) and pd.notna(row.iloc[0]["Adopted period (hr)"]):
        P_adopt = float(row.iloc[0]["Adopted period (hr)"])
    else:
        P_adopt = 5.0
else:
    P_adopt = 5.0


# ==========================================================
# PHOTOMETRY + FOLD
# ==========================================================
st.header(f"Fold Preview — {selected}")

with st.spinner("Loading photometry from BigQuery..."):
    df_raw = bq_load_photometry_for_provid(str(selected))

if df_raw is None or len(df_raw) == 0:
    st.warning("No photometry found for this object (stn filter + provid match).")
    st.stop()

df_raw["obstime_dt"] = pd.to_datetime(df_raw["obstime"], errors="coerce", utc=True)
df_raw["band"] = df_raw["band"].astype(str).str.strip().str.lower()
df_raw["mag"] = pd.to_numeric(df_raw["mag"], errors="coerce")
df_raw = df_raw.dropna(subset=["obstime_dt", "mag"]).sort_values("obstime_dt").reset_index(drop=True)

t0 = df_raw["obstime_dt"].min()
df_raw["t_hr"] = (df_raw["obstime_dt"] - t0).dt.total_seconds() / 3600.0

with st.spinner("Running coarse geometry correction (Horizons)..."):
    df_geo = step5_geometry_horizons_range(
        df_raw,
        PROVID=str(selected),
        STEP_MINUTES=10,
        PAD_MINUTES=10,
    )

mag_col = "mag_geo_bandcenter"
if mag_col not in df_geo.columns or df_geo[mag_col].notna().sum() < 5:
    st.warning("Geometry correction did not produce enough corrected points; falling back to raw mag.")
    mag_col = "mag"

# Period slider
lo = max(1e-6, P_adopt / 2.0)
hi = max(lo * 1.001, P_adopt * 2.0)

P = st.slider("Fold Period (hr)", float(lo), float(hi), float(P_adopt))

phase = (df_geo["t_hr"].to_numpy(float) / float(P)) % 1.0

fig, ax = plt.subplots(figsize=(7, 4))
bands = df_geo["band"].to_numpy(str)

for b in sorted(np.unique(bands).tolist()):
    m = (bands == b)
    ax.scatter(phase[m], df_geo.loc[m, mag_col].to_numpy(float), s=10, label=b)

ax.invert_yaxis()
ax.set_xlabel("Phase")
ax.set_ylabel(mag_col)
ax.legend(fontsize=8, ncol=6)
st.pyplot(fig, clear_figure=True)
