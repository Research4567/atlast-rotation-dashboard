import os
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ATLAST Rotation Dashboard", layout="wide")
st.title("ATLAST Asteroid Rotation Dashboard")

MASTER_FILE = "master_results_clean.csv"
OVERRIDES_FILE = "overrides.csv"

# ==========================
# Load Data
# ==========================
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

# ==========================
# Apply Overrides
# ==========================
if len(ovr) > 0:
    ovr["provid"] = ovr["provid"].astype(str)
    df["provid"] = df["provid"].astype(str)

    df = df.merge(ovr, on="provid", how="left", suffixes=("", "_ovr"))

    manual = df["triage_manual_ovr"].fillna("").astype(str).str.strip()
    df.loc[manual != "", "triage_final"] = manual[manual != ""]

    if "P_manual_hr_ovr" in df.columns:
        pman = pd.to_numeric(df["P_manual_hr_ovr"], errors="coerce")
        df.loc[pman.notna(), "P_final_hr"] = pman[pman.notna()]

# ==========================
# Sidebar Filters
# ==========================
st.sidebar.header("Triage Filters")

triage_options = sorted(df["triage_final"].unique())
selected_triage = st.sidebar.multiselect(
    "Class",
    triage_options,
    default=triage_options
)

top_only = st.sidebar.checkbox("Top reliable only (confidence ≥ 80)", value=False)

df_f = df[df["triage_final"].isin(selected_triage)]

if top_only and "confidence" in df_f.columns:
    df_f = df_f[df_f["confidence"] >= 80]

# ==========================
# Scoreboard
# ==========================
st.subheader("Scoreboard")

display_cols = [
    c for c in [
        "provid",
        "triage_final",
        "confidence",
        "P_final_hr",
        "sigma_sec",
        "N_obs",
        "arc_days",
        "bootstrap_top_frac",
        "bootstrap_n_unique",
        "g_r",
        "g_i",
        "r_i"
    ] if c in df_f.columns
]

# ✅ Safe sorting: only use confidence if it exists
sort_cols = ["triage_final"]
ascending = [True]
if "confidence" in df_f.columns:
    sort_cols.append("confidence")
    ascending.append(False)

df_sorted = df_f.sort_values(sort_cols, ascending=ascending)

st.dataframe(
    df_sorted[display_cols],
    use_container_width=True,
    height=450
)

st.download_button(
    "Download current scoreboard (CSV)",
    data=df_sorted[display_cols].to_csv(index=False).encode("utf-8"),
    file_name="scoreboard_filtered.csv",
    mime="text/csv",
)

# ==========================
# Select Asteroid
# ==========================
st.sidebar.header("Asteroid")
asteroids = df_f["provid"].tolist()

if len(asteroids) == 0:
    st.warning("No asteroids match filters.")
    st.stop()

selected = st.sidebar.selectbox("Choose asteroid", asteroids)
row = df[df["provid"] == selected].iloc[0]

st.header(selected)

# ==========================
# Key Results
# ==========================
st.subheader("Key Results")

key_cols = [
    c for c in [
        "triage_final",
        "confidence",
        "P_final_hr",
        "P_p16_hr",
        "P_p50_hr",
        "P_p84_hr",
        "sigma_sec",
        "bootstrap_top_frac",
        "bootstrap_top_P",
        "bootstrap_n_unique",
        "N_obs",
        "arc_days",
        "g_r",
        "g_i",
        "r_i",
        "pipeline_version",
        "run_id_utc",
        "last_updated_utc"
    ] if c in row.index
]

st.table(row[key_cols])

# ==========================
# Diagnostic Plots
# ==========================
st.subheader("Diagnostic Plots")

obj_dir = Path("outputs") / "objects" / selected.replace(" ", "_")

plots = [
    ("periodogram.png", "Periodogram"),
    ("fold_panel.png", "Fold comparison (P/2, P, 2P)"),
    ("bootstrap.png", "Bootstrap"),
]

found = False
for fname, title in plots:
    p = obj_dir / fname
    if p.exists():
        found = True
        st.image(str(p), caption=title, use_container_width=True)

if not found:
    st.info("No plots found yet. Upload images to outputs/objects/<provid>/")




