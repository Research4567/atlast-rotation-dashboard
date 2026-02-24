# ==========================================================
# ATLAST Rotation Dashboard
# - Master table from local CSV
# - Photometry fetched ON DEMAND from BigQuery (minimal cols)
# - Geometry correction computed ON THE FLY (JPL Horizons)
# - Coarse Horizons grid (demo mode: 10-min step)
# ==========================================================

from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from google.cloud import bigquery
from google.oauth2 import service_account

from astropy.time import Time
from astroquery.jplhorizons import Horizons

from __future__ import annotations

from pathlib import Path
import os
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

BQ_PROJECT = "lsst-484623"
BQ_DATASET = "asteroid_institute_mpc_replica"
BQ_TABLE   = "public_obs_sbn"
BQ_STN     = "X05"
BQ_ROW_LIMIT = 20000

HORIZONS_LOCATION = "X05"
HG_G_DEFAULT = 0.15


# ==========================================================
# BIGQUERY AUTH (STREAMLIT CLOUD SAFE)
# ==========================================================
@st.cache_resource
from google.oauth2 import service_account
from google.cloud import bigquery
import streamlit as st

@st.cache_resource
def get_bq_client() -> bigquery.Client:
    # HARD fail with a clear message if secrets aren't present
    if "gcp_service_account" not in st.secrets:
        st.error(
            "Missing Streamlit secret: [gcp_service_account]. "
            "Go to Streamlit Cloud → App → Settings → Secrets and paste the service account TOML."
        )
        st.stop()

    sa = dict(st.secrets["gcp_service_account"])

    # Optional sanity check
    if "client_email" not in sa or "private_key" not in sa:
        st.error(
            "Your [gcp_service_account] secret is present but incomplete. "
            "It must include at least client_email and private_key."
        )
        st.stop()

    creds = service_account.Credentials.from_service_account_info(sa)
    return bigquery.Client(project=BQ_PROJECT, credentials=creds)


# ==========================================================
# STEP 5 — GEOMETRY CORRECTION (COARSE DEMO VERSION)
# ==========================================================
def step5_geometry_horizons_range(
    df1: pd.DataFrame,
    *,
    PROVID: str,
    STEP_MINUTES: int = 10,     # ← DEMO CHANGE
    PAD_MINUTES: int = 10,      # ← DEMO CHANGE
):
    if df1 is None or len(df1) == 0:
        raise ValueError("Empty photometry.")

    df = df1.copy()
    df["obstime_dt"] = pd.to_datetime(df["obstime_dt"], utc=True)
    df = df.sort_values("obstime_dt").reset_index(drop=True)

    t_utc = Time(df["obstime_dt"].dt.to_pydatetime(), scale="utc")
    df["jd_utc_obs"] = t_utc.jd.astype(float)

    blocks = df["obstime_dt"].dt.date.unique()

    eph_parts = []

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
                "step":  f"{STEP_MINUTES}m",
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
        pd.concat(eph_parts)
        .drop_duplicates("jd_utc_eph")
        .sort_values("jd_utc_eph")
        .reset_index(drop=True)
    )

    df = df.sort_values("jd_utc_obs")
    eph_df = eph_df.sort_values("jd_utc_eph")

    df = pd.merge_asof(
        df,
        eph_df,
        left_on="jd_utc_obs",
        right_on="jd_utc_eph",
        direction="nearest",
        tolerance=max(2e-3, (STEP_MINUTES/1440)*1.2),
    )

    def phase_HG(alpha_deg, G=0.15):
        a = np.deg2rad(alpha_deg)
        phi1 = np.exp(-3.33 * np.tan(a/2)**0.63)
        phi2 = np.exp(-1.87 * np.tan(a/2)**1.22)
        return -2.5*np.log10((1-G)*phi1 + G*phi2)

    df["dist_term"] = 5*np.log10(df["r"] * df["delta"])
    df["phase_term"] = phase_HG(df["alpha"], G=HG_G_DEFAULT)
    df["mag_geo"] = df["mag"] - df["dist_term"] - df["phase_term"]

    df["mag_geo_bandcenter"] = (
        df["mag_geo"] -
        df.groupby("band")["mag_geo"].transform("median")
    )

    return df


# ==========================================================
# BIGQUERY PHOTOMETRY LOADER
# ==========================================================
@st.cache_data(ttl=3600)
def bq_load_photometry_for_provid(provid: str) -> pd.DataFrame:
    client = get_bq_client()

    query = f"""
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
    LIMIT {BQ_ROW_LIMIT}
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("stn", "STRING", BQ_STN),
            bigquery.ScalarQueryParameter("prov", "STRING", provid),
        ]
    )

    return client.query(query, job_config=job_config).to_dataframe()


# ==========================================================
# LOAD MASTER
# ==========================================================
if not MASTER_PATH.exists():
    st.error("master_results_clean.csv not found.")
    st.stop()

master = pd.read_csv(MASTER_PATH)

st.title("ATLAST Asteroid Rotation Dashboard")

selected = st.selectbox("Select Asteroid", master["Designation"].astype(str))

P_adopt = float(master.loc[
    master["Designation"].astype(str)==selected,
    "Adopted period (hr)"
].values[0])


# ==========================================================
# PHOTOMETRY TAB
# ==========================================================
st.header(f"Fold Preview — {selected}")

with st.spinner("Loading photometry..."):
    df_raw = bq_load_photometry_for_provid(selected)

if len(df_raw) == 0:
    st.warning("No photometry found.")
    st.stop()

df_raw["obstime_dt"] = pd.to_datetime(df_raw["obstime"], utc=True)
df_raw["band"] = df_raw["band"].str.lower()
df_raw = df_raw.dropna(subset=["mag"])

t0 = df_raw["obstime_dt"].min()
df_raw["t_hr"] = (df_raw["obstime_dt"] - t0).dt.total_seconds()/3600

with st.spinner("Running coarse geometry correction..."):
    df_geo = step5_geometry_horizons_range(
        df_raw,
        PROVID=selected,
        STEP_MINUTES=10,
        PAD_MINUTES=10,
    )

mag_col = "mag_geo_bandcenter"

P = st.slider("Fold Period (hr)", P_adopt/2, P_adopt*2, P_adopt)

phase = (df_geo["t_hr"]/P) % 1

fig, ax = plt.subplots(figsize=(7,4))

for b in sorted(df_geo["band"].unique()):
    mask = df_geo["band"]==b
    ax.scatter(phase[mask], df_geo.loc[mask, mag_col], s=10, label=b)

ax.invert_yaxis()
ax.set_xlabel("Phase")
ax.set_ylabel("Corrected Mag")
ax.legend()

st.pyplot(fig)


