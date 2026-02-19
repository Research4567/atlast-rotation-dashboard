import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ATLAST Rotation Dashboard", layout="wide")
st.title("ATLAST Asteroid Rotation Dashboard")

# -------------------------
# Load master + overrides
# -------------------------
MASTER_FILE = "master_results_clean.csv"
OVERRIDES_FILE = "overrides.csv"

@st.cache_data(show_spinner=False)
def load_master():
    return pd.read_csv(MASTER_FILE)

@st.cache_data(show_spinner=False)
def load_overrides():
    if os.path.exists(OVERRIDES_FILE):
        return pd.read_csv(OVERRIDES_FILE)
    return pd.DataFrame(columns=["provid", "triage_manual", "P_manual_hr", "notes"])

try:
    df = load_master()
except Exception as e:
    st.error(f"Could not load {MASTER_FILE}: {e}")
    st.stop()

ovr = load_overrides()

# Ensure expected columns exist
for c in ["provid", "triage_auto", "triage_manual", "triage_final"]:
    if c not in df.columns:
        st.error(f"Missing column in {MASTER_FILE}: {c}")
        st.stop()

# Apply overrides (if any)
if len(ovr) > 0:
    ovr["provid"] = ovr["provid"].astype(str)
    df["provid"] = df["provid"].astype(str)
    df = df.merge(ovr, on="provid", how="left", suffixes=("", "_ovr"))

    # triage_final: manual override wins if provided
    manual = df["triage_manual_ovr"].fillna("").astype(str).str.strip()
    df["triage_final"] = df["triage_final"].astype(str)
    df.loc[manual != "", "triage_final"] = manual[manual != ""]

    # optional: period override
    if "P_manual_hr_ovr" in df.columns:
        pman = pd.to_numeric(df["P_manual_hr_ovr"], errors="coerce")
        if "P_final_hr" in df.columns:
            df.loc[pman.notna(), "P_final_hr"] = pman[pman.notna()]

    # optional: notes
    if "notes_ovr" in df.columns:
        df["notes"] = df["notes_ovr"]

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Triage Filters")

triage_options = sorted(df["triage_final"].dropna().astype(str).unique().tolist())
selected_triage = st.sidebar.multiselect(
    "Class",
    triage_options,
    default=triage_options
)

# Some numeric filters (only show if columns exist)
def slider_if(col, label, default_min=None):
    if col not in df.columns:
        return None
    v = pd.to_numeric(df[col], errors="coerce")
    v = v.dropna()
    if len(v) == 0:
        return None
    mn, mx = float(v.min()), float(v.max())
    if default_min is None:
        default_min = mn
    return st.sidebar.slider(label, mn, mx, float(default_min))

min_obs = slider_if("N_obs", "Min observations (N_obs)", default_min=0)
min_boot = slider_if("bootstrap_top_frac", "Min bootstrap_top_frac", default_min=0.0)

# Apply filters
df_f = df.copy()
df_f = df_f[df_f["triage_final"].astype(str).isin(selected_triage)]

if min_obs is not None:
    df_f = df_f[pd.to_numeric(df_f["N_obs"], errors="coerce") >= min_obs]
if min_boot is not None:
    df_f = df_f[pd.to_numeric(df_f["bootstrap_top_frac"], errors="coerce") >= min_boot]

st.subheader("Scoreboard")

# Show a clean table
show_cols = [c for c in [
    "provid", "triage_final", "P_final_hr", "P_p50_hr", "sigma_sec",
    "N_obs", "arc_days", "bootstrap_top_frac", "bootstrap_n_unique",
    "g_r", "g_i", "r_i"
] if c in df_f.columns]

st.dataframe(
    df_f.sort_values(["triage_final", "bootstrap_top_frac"], ascending=[True, False])[show_cols],
    use_container_width=True,
    height=420
)

# -------------------------
# Pick asteroid
# -------------------------
st.sidebar.header("Asteroid")
asteroids = df_f["provid"].astype(str).tolist()
if len(asteroids) == 0:
    st.warning("No asteroids match your filters.")
    st.stop()

selected = st.sidebar.selectbox("Choose asteroid", asteroids)
row = df[df["provid"].astype(str) == str(selected)].iloc[0]

st.header(f"{selected}")

st.subheader("Key results")
key_cols = [c for c in [
    "triage_final", "P_final_hr", "P_p16_hr", "P_p50_hr", "P_p84_hr", "sigma_sec",
    "bootstrap_top_frac", "bootstrap_top_P", "bootstrap_n_unique",
    "N_obs", "arc_days", "g_r", "g_i", "r_i",
    "pipeline_version", "run_id_utc", "last_updated_utc",
] if c in row.index]
st.table(row[key_cols])

# -------------------------
# Plots (optional for now)
# -------------------------
st.subheader("Diagnostic plots")

# If later you upload images into repo at outputs/objects/<provid>/
obj_dir = Path("outputs") / "objects" / str(selected).replace(" ", "_")
candidates = [
    ("periodogram.png", "Periodogram"),
    ("fold_panel.png", "Fold comparison (P/2, P, 2P)"),
    ("bootstrap.png", "Bootstrap"),
    ("residuals.png", "Residuals"),
]
found_any = False
for fname, title in candidates:
    p = obj_dir / fname
    if p.exists():
        found_any = True
        st.image(str(p), caption=title, use_container_width=True)

if not found_any:
    st.info("No plots found yet. (Later: upload images to outputs/objects/<provid>/)")
