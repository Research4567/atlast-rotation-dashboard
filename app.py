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
# Photometry Section
# ==========================
st.subheader("Photometry Explorer")

PHOTO_FILE = "bq-results.csv"  # must exist in repo OR load from URL

@st.cache_data(show_spinner=False)
def load_photometry():
    return pd.read_csv(PHOTO_FILE)

try:
    df_photo = load_photometry()
except Exception as e:
    st.warning("Photometry file not found yet.")
    st.stop()

# Filter to selected asteroid
df_obj = df_photo[df_photo["provid"].astype(str) == selected].copy()

if len(df_obj) == 0:
    st.warning("No photometry rows for this asteroid.")
    st.stop()

# Convert time
df_obj["obstime"] = pd.to_datetime(df_obj["obstime"], errors="coerce")
df_obj = df_obj.dropna(subset=["obstime", "mag"])

# Sidebar controls
st.sidebar.header("Photometry Controls")

bands = sorted(df_obj["band"].unique())
sel_bands = st.sidebar.multiselect("Bands", bands, default=bands)

max_rms = st.sidebar.slider(
    "Max rmsmag",
    float(df_obj["rmsmag"].min()),
    float(df_obj["rmsmag"].max()),
    float(df_obj["rmsmag"].quantile(0.9))
)

df_plot = df_obj[
    (df_obj["band"].isin(sel_bands)) &
    (df_obj["rmsmag"] <= max_rms)
].copy()

# ------------------
# Raw Lightcurve
# ------------------
st.markdown("### Folded Lightcurve")

P_calc = float(row["P_final_hr"])   # your calculated/adopted period (hours)

# --- quick controls ---
c1, c2, c3 = st.columns(3)

# store the current period in session_state so buttons can change it
if "P_current" not in st.session_state:
    st.session_state.P_current = P_calc

if c1.button("Reset to calculated P"):
    st.session_state.P_current = P_calc

if c2.button("Use P/2"):
    st.session_state.P_current = st.session_state.P_current / 2.0

if c3.button("Use 2P"):
    st.session_state.P_current = st.session_state.P_current * 2.0

st.caption(f"Calculated rotation period (P_final_hr): **{P_calc:.6f} h**")

# --- slider bound around calculated period ---
P = st.slider(
    "Fold period (hours)",
    min_value=float(P_calc * 0.25),
    max_value=float(P_calc * 4.0),
    value=float(st.session_state.P_current),
    step=float(max(P_calc * 0.001, 1e-4)),
)

# keep session state synced with slider moves
st.session_state.P_current = P

# Convert time to hours relative
t0 = df_plot["obstime"].min()
t_hr = (df_plot["obstime"] - t0).dt.total_seconds() / 3600.0
phase = (t_hr / P) % 1.0

fig2, ax2 = plt.subplots()

for b in sel_bands:
    m = df_plot["band"] == b
    ax2.scatter(phase[m], df_plot["mag"][m], s=8, label=b)
    ax2.scatter(phase[m] + 1.0, df_plot["mag"][m], s=8)

ax2.invert_yaxis()
ax2.set_xlabel("Phase")
ax2.set_ylabel("Magnitude")
ax2.legend()

st.pyplot(fig2)

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






