import streamlit as st
import pandas as pd
from pathlib import Path
import os

st.set_page_config(page_title="ATLAST Rotation Dashboard", layout="wide")

st.title("ATLAST Asteroid Rotation Dashboard")

# ==========================
# Load master results
# ==========================
@st.cache_data
def load_master():
    return pd.read_csv("master_results.csv")

try:
    df = load_master()
except Exception as e:
    st.error(f"Could not load master_results.csv: {e}")
    st.stop()

# ==========================
# Sidebar Filters
# ==========================
st.sidebar.header("Triage Filters")

triage_options = df["triage_class"].unique().tolist()
selected_triage = st.sidebar.multiselect(
    "Select classification",
    triage_options,
    default=triage_options
)

min_obs = st.sidebar.slider("Minimum observations", 0, int(df["N_obs"].max()), 0)
min_snr = st.sidebar.slider("Minimum SNR amplitude", 0.0, float(df["snr_amp"].max()), 0.0)

df_f = df[
    (df["triage_class"].isin(selected_triage)) &
    (df["N_obs"] >= min_obs) &
    (df["snr_amp"] >= min_snr)
]

st.subheader("Scoreboard")

st.dataframe(
    df_f.sort_values("triage_class"),
    use_container_width=True
)

# ==========================
# Select asteroid
# ==========================
st.sidebar.header("Asteroid Selection")

asteroids = df_f["provid"].tolist()
if len(asteroids) == 0:
    st.warning("No asteroids match filters.")
    st.stop()

selected = st.sidebar.selectbox("Choose asteroid", asteroids)

row = df[df["provid"] == selected].iloc[0]

st.header(f"Asteroid: {selected}")

# ==========================
# Key Results Table
# ==========================
st.subheader("Key Results")

key_cols = [
    "triage_class",
    "P_final_hr",
    "P_alt_hr",
    "dBIC_2nd",
    "sigma_sec",
    "snr_amp",
    "N_obs",
    "N_nights",
    "night_dom_frac"
]

st.table(row[key_cols])

# ==========================
# Plot Display
# ==========================
st.subheader("Diagnostic Plots")

object_folder = f"outputs/objects/{selected.replace(' ', '_')}"

periodogram_path = f"{object_folder}/periodogram.png"
fold_path = f"{object_folder}/fold_panel.png"
bootstrap_path = f"{object_folder}/bootstrap.png"

if os.path.exists(periodogram_path):
    st.image(periodogram_path, caption="Periodogram")

if os.path.exists(fold_path):
    st.image(fold_path, caption="Fold Comparison (P/2, P, 2P)")

if os.path.exists(bootstrap_path):
    st.image(bootstrap_path, caption="Bootstrap Distribution")

# ==========================
# What’s New (optional future feature)
# ==========================
st.markdown("---")
st.subheader("What's New")

new_reliable = df[df["triage_class"] == "reliable"]

st.write(f"Total reliable periods: {len(new_reliable)}")