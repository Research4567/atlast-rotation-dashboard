import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="ATLAST Rotation Dashboard", layout="wide")
st.title("ATLAST Asteroid Rotation Dashboard")

MASTER_FILE = "master_results_clean.csv"
OVERRIDES_FILE = "overrides.csv"
PHOTO_FILE = "bq-results.csv"  # you uploaded this to the repo (18 MB)

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

@st.cache_data(show_spinner=False)
def load_photometry():
    return pd.read_csv(PHOTO_FILE, low_memory=False)

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

    manual = df.get("triage_manual_ovr", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    df.loc[manual != "", "triage_final"] = manual[manual != ""]

    pman = pd.to_numeric(df.get("P_manual_hr_ovr", np.nan), errors="coerce")
    if "P_final_hr" in df.columns:
        df.loc[pman.notna(), "P_final_hr"] = pman[pman.notna()]

# ==========================
# Sidebar Filters
# ==========================
st.sidebar.header("Triage Filters")

triage_options = sorted(df["triage_final"].dropna().astype(str).unique())
selected_triage = st.sidebar.multiselect("Class", triage_options, default=triage_options)

top_only = st.sidebar.checkbox("Top reliable only (confidence ≥ 80)", value=False)

df_f = df[df["triage_final"].astype(str).isin(selected_triage)].copy()
if top_only and "confidence" in df_f.columns:
    df_f = df_f[pd.to_numeric(df_f["confidence"], errors="coerce") >= 80].copy()

# ==========================
# Scoreboard
# ==========================
st.subheader("Scoreboard")

display_cols = [
    c for c in [
        "provid", "triage_final", "confidence",
        "P_final_hr", "sigma_sec",
        "N_obs", "arc_days",
        "bootstrap_top_frac", "bootstrap_n_unique",
        "g_r", "g_i", "r_i"
    ] if c in df_f.columns
]

sort_cols = ["triage_final"]
ascending = [True]
if "confidence" in df_f.columns:
    sort_cols.append("confidence")
    ascending.append(False)

df_sorted = df_f.sort_values(sort_cols, ascending=ascending)

st.dataframe(df_sorted[display_cols], use_container_width=True, height=420)

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
asteroids = df_sorted["provid"].astype(str).tolist()

if len(asteroids) == 0:
    st.warning("No asteroids match filters.")
    st.stop()

selected = st.sidebar.selectbox("Choose asteroid", asteroids)
row = df[df["provid"].astype(str) == str(selected)].iloc[0]

st.header(selected)

# ==========================
# Key Results
# ==========================
st.subheader("Key Results (official pipeline outputs)")

key_cols = [
    c for c in [
        "triage_final", "confidence",
        "P_final_hr", "P_p16_hr", "P_p50_hr", "P_p84_hr",
        "sigma_sec",
        "bootstrap_top_frac", "bootstrap_top_P", "bootstrap_n_unique",
        "N_obs", "arc_days",
        "g_r", "g_i", "r_i",
        "pipeline_version", "run_id_utc", "last_updated_utc"
    ] if c in row.index
]
st.table(row[key_cols])

# ==========================
# Photometry Explorer
# ==========================
st.subheader("Photometry Explorer (exploratory folds only)")

try:
    df_photo = load_photometry()
except Exception as e:
    st.warning(f"Photometry file not found or failed to load: {e}")
    st.stop()

needed = {"provid", "obstime", "mag", "rmsmag", "band"}
missing = [c for c in needed if c not in df_photo.columns]
if missing:
    st.error(f"Photometry file missing columns: {missing}")
    st.stop()

# Filter to selected asteroid
df_obj = df_photo[df_photo["provid"].astype(str) == str(selected)].copy()
if len(df_obj) == 0:
    st.warning("No photometry rows for this asteroid.")
    st.stop()

df_obj["obstime"] = pd.to_datetime(df_obj["obstime"], errors="coerce", utc=True).dt.tz_convert(None)
df_obj["mag"] = pd.to_numeric(df_obj["mag"], errors="coerce")
df_obj["rmsmag"] = pd.to_numeric(df_obj["rmsmag"], errors="coerce")
df_obj["band"] = df_obj["band"].astype(str)

df_obj = df_obj.dropna(subset=["obstime", "mag", "rmsmag", "band"])
df_obj = df_obj.sort_values("obstime")

# Sidebar controls for photometry
st.sidebar.header("Photometry Controls")
bands = sorted(df_obj["band"].unique().tolist())
sel_bands = st.sidebar.multiselect("Bands", bands, default=bands)

rms_min = float(df_obj["rmsmag"].min())
rms_max = float(df_obj["rmsmag"].max())
rms_default = float(df_obj["rmsmag"].quantile(0.90))
max_rms = st.sidebar.slider("Max rmsmag", rms_min, rms_max, rms_default)

df_plot = df_obj[(df_obj["band"].isin(sel_bands)) & (df_obj["rmsmag"] <= max_rms)].copy()

if len(df_plot) < 10:
    st.warning("Too few photometry points after filters. Increase max rmsmag or select more bands.")
    st.stop()

# ------------------
# Raw Lightcurve
# ------------------
st.markdown("### Raw Lightcurve (time domain)")

fig_raw, ax_raw = plt.subplots()
for b in sel_bands:
    d = df_plot[df_plot["band"] == b]
    ax_raw.scatter(d["obstime"], d["mag"], s=10, label=b)

ax_raw.invert_yaxis()
ax_raw.set_xlabel("Time")
ax_raw.set_ylabel("Magnitude")
ax_raw.legend()
st.pyplot(fig_raw, clear_figure=True)

# ------------------
# Fold quality metric
# ------------------
def fold_quality_score(t_hr: np.ndarray, mag: np.ndarray, P_hr: float, nbins: int = 30) -> float:
    """
    Simple fold quality metric:
    - compute phase
    - bin phases
    - within-bin robust scatter (MAD)
    - return 0–100 score where higher is better
    """
    if not np.isfinite(P_hr) or P_hr <= 0:
        return np.nan

    phase = (t_hr / P_hr) % 1.0
    bins = np.linspace(0, 1, nbins + 1)
    scatters = []

    for i in range(nbins):
        m = (phase >= bins[i]) & (phase < bins[i + 1])
        if m.sum() < 5:
            continue
        y = mag[m]
        med = np.nanmedian(y)
        mad = np.nanmedian(np.abs(y - med))
        scatters.append(1.4826 * mad)  # approx sigma

    if len(scatters) < 5:
        return np.nan

    s = float(np.nanmedian(scatters))
    # Convert scatter to a score: smaller scatter => higher score
    # Scale relative to overall mag scatter:
    base = float(np.nanstd(mag))
    if not np.isfinite(base) or base <= 0:
        base = 0.1
    ratio = s / base
    score = 100 * max(0.0, 1.0 - ratio)  # ratio=0 => 100, ratio>=1 => 0
    return float(np.clip(score, 0, 100))

# ------------------
# Folded Lightcurve controls
# ------------------
st.markdown("### Folded Lightcurve (phase domain)")

P_calc = float(row["P_final_hr"]) if "P_final_hr" in row.index and pd.notna(row["P_final_hr"]) else np.nan
if not np.isfinite(P_calc) or P_calc <= 0:
    st.warning("No valid P_final_hr for this asteroid.")
    st.stop()

# session state to support buttons
if "P_current" not in st.session_state:
    st.session_state.P_current = P_calc

c1, c2, c3 = st.columns(3)
if c1.button("Reset to calculated P"):
    st.session_state.P_current = P_calc
if c2.button("Use P/2"):
    st.session_state.P_current = st.session_state.P_current / 2.0
if c3.button("Use 2P"):
    st.session_state.P_current = st.session_state.P_current * 2.0

st.caption(f"Calculated rotation period (P_final_hr): **{P_calc:.6f} h**")

P = st.slider(
    "Fold period (hours)",
    min_value=float(P_calc * 0.25),
    max_value=float(P_calc * 4.0),
    value=float(st.session_state.P_current),
    step=float(max(P_calc * 0.001, 1e-4)),
)
st.session_state.P_current = P

# Fold
t0 = df_plot["obstime"].min()
t_hr = (df_plot["obstime"] - t0).dt.total_seconds().to_numpy() / 3600.0

# Detrend per band (helps folding)
df_fold = df_plot.copy()
df_fold["mag_detrend"] = df_fold["mag"] - df_fold.groupby("band")["mag"].transform("median")

phase = (t_hr / P) % 1.0

# fold-quality
score = fold_quality_score(t_hr, df_fold["mag_detrend"].to_numpy(), P, nbins=30)
st.metric("Fold Quality Score (0–100, higher is better)", f"{score:.1f}" if np.isfinite(score) else "n/a")

fig_fold, ax_fold = plt.subplots()
for b in sel_bands:
    m = df_fold["band"] == b
    ax_fold.scatter(phase[m], df_fold.loc[m, "mag_detrend"], s=10, label=b)
    ax_fold.scatter(phase[m] + 1.0, df_fold.loc[m, "mag_detrend"], s=10)

ax_fold.invert_yaxis()
ax_fold.set_xlabel("Phase")
ax_fold.set_ylabel("Detrended mag (mag - median per band)")
ax_fold.legend()
st.pyplot(fig_fold, clear_figure=True)

# ==========================
# Diagnostic Plots (official images)
# ==========================
st.subheader("Diagnostic Plots (official, from pipeline outputs)")

obj_dir = Path("outputs") / "objects" / str(selected).replace(" ", "_")
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
    st.info("No official plot images found yet. Upload images to outputs/objects/<provid>/")
