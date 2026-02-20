import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional (periodogram)
try:
    from astropy.timeseries import LombScargle
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False

# --------------------------
# App config
# --------------------------
st.set_page_config(page_title="ATLAST Rotation Dashboard", layout="wide")
st.title("ATLAST Asteroid Rotation Dashboard")

# --------------------------
# Files / directories
# --------------------------
MASTER_FILE = Path("master_results_clean.csv")
OVERRIDES_FILE = Path("overrides.csv")
PHOTO_FILE = Path("bq-results.csv")          # repo photometry
PIPELINE_ROOT = Path("outputs")              # where you store per-object outputs
OBJECTS_DIR = PIPELINE_ROOT / "objects"      # outputs/objects/<provid>/

# --------------------------
# Helpers
# --------------------------
def _as_str(x) -> str:
    return "" if x is None else str(x)

def _safe_num(x):
    return pd.to_numeric(x, errors="coerce")

def _norm_provid(p: str) -> str:
    # keep consistent naming for folder lookups
    return _as_str(p).strip()

def _obj_dir(provid: str) -> Path:
    # NOTE: you can adjust this mapping to match your repo structure
    return OBJECTS_DIR / _norm_provid(provid).replace(" ", "_")

def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def _read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None

# --------------------------
# Load data
# --------------------------
@st.cache_data(show_spinner=False)
def load_master() -> pd.DataFrame:
    return pd.read_csv(MASTER_FILE)

@st.cache_data(show_spinner=False)
def load_overrides() -> pd.DataFrame:
    if OVERRIDES_FILE.exists():
        return pd.read_csv(OVERRIDES_FILE)
    return pd.DataFrame(columns=["provid", "triage_manual", "P_manual_hr", "notes"])

@st.cache_data(show_spinner=False)
def load_photometry() -> pd.DataFrame:
    return pd.read_csv(PHOTO_FILE, low_memory=False)

try:
    df_master = load_master()
except Exception as e:
    st.error(f"Could not load {MASTER_FILE}: {e}")
    st.stop()

df_master = df_master.copy()
df_master["provid"] = df_master["provid"].astype(str)

df_ovr = load_overrides().copy()
if "provid" in df_ovr.columns:
    df_ovr["provid"] = df_ovr["provid"].astype(str)

# --------------------------
# Apply overrides (clean + predictable)
# --------------------------
if len(df_ovr) > 0 and "provid" in df_ovr.columns:
    df = df_master.merge(df_ovr, on="provid", how="left")

    # manual triage override
    if "triage_manual" in df.columns:
        m = df["triage_manual"].fillna("").astype(str).str.strip()
        if "triage_final" in df.columns:
            df.loc[m != "", "triage_final"] = m[m != ""]
        else:
            df["triage_final"] = np.where(m != "", m, df.get("triage_final", ""))

    # manual period override
    if "P_manual_hr" in df.columns and "P_final_hr" in df.columns:
        pman = _safe_num(df["P_manual_hr"])
        df.loc[pman.notna(), "P_final_hr"] = pman[pman.notna()]
else:
    df = df_master

# If triage_final missing, make it exist
if "triage_final" not in df.columns:
    df["triage_final"] = df.get("reliability", "unknown")

# --------------------------
# Sidebar Filters
# --------------------------
st.sidebar.header("Triage Filters")

triage_options = sorted(df["triage_final"].dropna().astype(str).unique())
selected_triage = st.sidebar.multiselect("Class", triage_options, default=triage_options)

top_only = st.sidebar.checkbox("Top reliable only (confidence ≥ 80)", value=False)

df_f = df[df["triage_final"].astype(str).isin(selected_triage)].copy()
if top_only and "confidence" in df_f.columns:
    df_f = df_f[_safe_num(df_f["confidence"]) >= 80].copy()

# --------------------------
# Scoreboard
# --------------------------
st.subheader("Scoreboard")

display_cols = [
    c for c in [
        "provid", "triage_final", "confidence",
        "P_final_hr", "sigma_sec",
        "N_obs", "arc_days",
        "bootstrap_top_frac", "bootstrap_n_unique",
        "g_r", "g_i", "r_i",
        "run_id_utc", "pipeline_version"
    ] if c in df_f.columns
]

sort_cols = ["triage_final"]
ascending = [True]
if "confidence" in df_f.columns:
    sort_cols.append("confidence")
    ascending.append(False)

df_sorted = df_f.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

st.dataframe(df_sorted[display_cols], use_container_width=True, height=380)

st.download_button(
    "Download current scoreboard (CSV)",
    data=df_sorted[display_cols].to_csv(index=False).encode("utf-8"),
    file_name="scoreboard_filtered.csv",
    mime="text/csv",
)

# --------------------------
# Sidebar asteroid selector
# --------------------------
st.sidebar.header("Asteroid")
asteroid_list = df_sorted["provid"].astype(str).tolist()

if len(asteroid_list) == 0:
    st.warning("No asteroids match filters.")
    st.stop()

if "selected_asteroid" not in st.session_state:
    st.session_state["selected_asteroid"] = asteroid_list[0]

# keep selection valid if filters change
if st.session_state["selected_asteroid"] not in asteroid_list:
    st.session_state["selected_asteroid"] = asteroid_list[0]

selected = st.sidebar.selectbox(
    "Choose asteroid",
    asteroid_list,
    index=asteroid_list.index(st.session_state["selected_asteroid"])
)
st.session_state["selected_asteroid"] = selected

row = df[df["provid"].astype(str) == str(selected)].iloc[0]
st.header(selected)

# --------------------------
# Tabs
# --------------------------
tab_key, tab_photo, tab_pipeline = st.tabs(["Key results", "Photometry explorer", "Pipeline diagnostics"])

# ==========================
# Key Results
# ==========================
with tab_key:
    st.subheader("Key Results (pipeline outputs)")

    key_cols = [
        c for c in [
            "triage_final", "confidence",
            "P_final_hr", "P_p16_hr", "P_p50_hr", "P_p84_hr",
            "sigma_sec",
            "bootstrap_top_frac", "bootstrap_n_unique",
            "N_obs", "arc_days",
            "g_r", "g_i", "r_i",
            "run_id_utc", "pipeline_version", "last_updated_utc"
        ] if c in row.index
    ]
    if key_cols:
        st.table(row[key_cols])
    else:
        st.info("No key columns found in master file for this object.")

    # Quick adopted-vs-2P evidence if present
    extra_cols = [c for c in ["2P candidate (hr)", "ΔBIC(2P−P)", "LS peak period (hr)"] if c in df.columns]
    if extra_cols:
        st.caption("Extra period-evidence fields (if present in master CSV)")
        st.table(row[extra_cols])

# ==========================
# Photometry Explorer
# ==========================
with tab_photo:
    st.subheader("Photometry Explorer (exploratory folds + periodogram)")

    if not PHOTO_FILE.exists():
        st.info(f"No photometry file found at {PHOTO_FILE}. Upload it to enable this tab.")
        st.stop()

    try:
        df_photo = load_photometry()
    except Exception as e:
        st.error(f"Photometry failed to load: {e}")
        st.stop()

    needed = {"provid", "obstime", "mag", "rmsmag", "band"}
    missing = [c for c in needed if c not in df_photo.columns]
    if missing:
        st.error(f"Photometry file missing columns: {missing}")
        st.stop()

    df_obj = df_photo[df_photo["provid"].astype(str) == str(selected)].copy()
    if len(df_obj) == 0:
        st.warning("No photometry rows for this asteroid.")
        st.stop()

    df_obj["obstime"] = pd.to_datetime(df_obj["obstime"], errors="coerce", utc=True).dt.tz_convert(None)
    df_obj["mag"] = _safe_num(df_obj["mag"])
    df_obj["rmsmag"] = _safe_num(df_obj["rmsmag"])
    df_obj["band"] = df_obj["band"].astype(str).str.strip().str.lower()

    df_obj = df_obj.dropna(subset=["obstime", "mag", "rmsmag", "band"]).sort_values("obstime")

    # Controls
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

    # time array (hours since first obs)
    t0 = df_plot["obstime"].min()
    t_hr = (df_plot["obstime"] - t0).dt.total_seconds().to_numpy() / 3600.0

    # detrend per band for exploratory viewing (not your official pipeline reduction)
    df_plot["mag_detrend"] = df_plot["mag"] - df_plot.groupby("band")["mag"].transform("median")

    # Raw LC
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

    # Fold-quality metric
    def fold_quality_score(t_hr: np.ndarray, mag: np.ndarray, P_hr: float, nbins: int = 30) -> float:
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
            scatters.append(1.4826 * mad)
        if len(scatters) < 5:
            return np.nan
        s = float(np.nanmedian(scatters))
        base = float(np.nanstd(mag))
        if not np.isfinite(base) or base <= 0:
            base = 0.1
        ratio = s / base
        score = 100 * max(0.0, 1.0 - ratio)
        return float(np.clip(score, 0, 100))

    # Fold controls
    st.markdown("### Folded Lightcurve (phase domain)")

    P_calc = float(row["P_final_hr"]) if "P_final_hr" in row.index and pd.notna(row["P_final_hr"]) else np.nan
    if not np.isfinite(P_calc) or P_calc <= 0:
        st.warning("No valid P_final_hr for this asteroid.")
        st.stop()

    state_key = f"P_current__{selected}"
    base_key = f"P_base__{selected}"

    if base_key not in st.session_state:
        st.session_state[base_key] = P_calc
    else:
        # keep base synced to current object’s official P
        st.session_state[base_key] = P_calc

    if state_key not in st.session_state:
        st.session_state[state_key] = P_calc

    c1, c2, c3 = st.columns(3)
    if c1.button("Reset to P_final"):
        st.session_state[state_key] = st.session_state[base_key]
    if c2.button("Use P/2 (of P_final)"):
        st.session_state[state_key] = 0.5 * st.session_state[base_key]
    if c3.button("Use 2P (of P_final)"):
        st.session_state[state_key] = 2.0 * st.session_state[base_key]

    st.caption(f"Pipeline adopted period (P_final_hr): **{P_calc:.6f} h**")

    P = st.slider(
        "Fold period (hours)",
        min_value=float(P_calc * 0.5),
        max_value=float(P_calc * 2.0),
        value=float(st.session_state[state_key]),
        step=float(max(P_calc * 0.001, 1e-4)),
    )
    st.session_state[state_key] = P

    phase = (t_hr / P) % 1.0
    score = fold_quality_score(t_hr, df_plot["mag_detrend"].to_numpy(), P, nbins=30)
    st.metric("Fold Quality Score (0–100)", f"{score:.1f}" if np.isfinite(score) else "n/a")

    fig_fold, ax_fold = plt.subplots()
    for b in sel_bands:
        m = df_plot["band"] == b
        ax_fold.scatter(phase[m], df_plot.loc[m, "mag_detrend"], s=10, label=b)
        ax_fold.scatter(phase[m] + 1.0, df_plot.loc[m, "mag_detrend"], s=10)
    ax_fold.invert_yaxis()
    ax_fold.set_xlabel("Phase")
    ax_fold.set_ylabel("Detrended mag (mag - median per band)")
    ax_fold.legend()
    st.pyplot(fig_fold, clear_figure=True)

    # Periodogram (exploratory)
    st.markdown("### Periodogram (exploratory Lomb–Scargle)")
    if not HAS_ASTROPY:
        st.info("Install astropy to enable Lomb–Scargle periodogram (pip install astropy).")
    else:
        y = df_plot["mag_detrend"].to_numpy(float)
        # weights: prefer rmsmag if sensible
        dy = df_plot["rmsmag"].to_numpy(float)
        dy = np.where(np.isfinite(dy) & (dy > 0), dy, np.nan)
        med_dy = np.nanmedian(dy) if np.isfinite(np.nanmedian(dy)) else 0.1
        dy = np.where(np.isfinite(dy), dy, med_dy)

        # Frequency grid: around P_final span
        Pmin = max(0.05, 0.25 * P_calc)   # avoid absurdly small
        Pmax = 4.0 * P_calc
        fmin = 1.0 / Pmax
        fmax = 1.0 / Pmin
        freq = np.linspace(fmin, fmax, 3000)

        ls = LombScargle(t_hr, y, dy=dy)
        power = ls.power(freq)

        period = 1.0 / freq
        fig_pg, ax_pg = plt.subplots()
        ax_pg.plot(period, power)
        ax_pg.set_xscale("log")
        ax_pg.set_xlabel("Period (hr) [log]")
        ax_pg.set_ylabel("LS power")
        ax_pg.axvline(P_calc, linestyle="--")
        ax_pg.axvline(0.5 * P_calc, linestyle=":")
        ax_pg.axvline(2.0 * P_calc, linestyle=":")
        ax_pg.set_title("Exploratory LS periodogram (band-detrended mags)")
        st.pyplot(fig_pg, clear_figure=True)

# ==========================
# Pipeline Diagnostics (official outputs)
# ==========================
with tab_pipeline:
    st.subheader("Pipeline diagnostics (official artifacts)")

    obj_dir = _obj_dir(selected)
    st.caption(f"Looking in: {obj_dir}")

    # Common official plots
    st.markdown("### Official diagnostic images")
    plots = [
        ("periodogram.png", "Periodogram"),
        ("fold_panel.png", "Fold comparison (P/2, P, 2P)"),
        ("bootstrap.png", "Bootstrap summary"),
        ("step13_folded_final.png", "Final folded LC (Step 13)"),
        ("step13_residuals_vs_time.png", "Residuals vs time (Step 13)"),
        ("step13_residual_hist.png", "Residual histogram (Step 13)"),
    ]

    found_any = False
    for fname, title in plots:
        p = obj_dir / fname
        if p.exists():
            found_any = True
            st.image(str(p), caption=title, use_container_width=True)

    if not found_any:
        st.info("No official plot images found yet. Place PNGs in outputs/objects/<provid>/")

    # CSV outputs (if you save them per object)
    st.markdown("### Official tables (CSV if present)")

    # adjust these names to match what your pipeline writes
    candidates_csv = obj_dir / "step11_fourier_validator.csv"
    fam_csv = obj_dir / "step11_families.csv"                 # optional if you choose to save
    boot_winners_csv = obj_dir / "step12_bootstrap_winner_fractions.csv"
    step13_summary_csv = obj_dir / "step13_final_summary.csv"

    colA, colB = st.columns(2)

    with colA:
        df_cand = _read_csv_if_exists(candidates_csv)
        if df_cand is not None and len(df_cand):
            st.caption("Step 11: Candidate table (top 20 by BIC)")
            show = df_cand.copy()
            for c in ["P_hr", "BIC", "wrms", "K_best", "amp_ptp_mag"]:
                if c in show.columns:
                    show[c] = _safe_num(show[c])
            show = show.sort_values(["BIC", "wrms"], ascending=[True, True]).head(20)
            st.dataframe(show, use_container_width=True, height=280)
            st.download_button("Download Step 11 candidate table", df_cand.to_csv(index=False).encode("utf-8"),
                               file_name=f"{selected}_step11_candidates.csv", mime="text/csv")
        else:
            st.info("No step11_fourier_validator.csv found for this object (optional).")

        df_sum = _read_csv_if_exists(step13_summary_csv)
        if df_sum is not None and len(df_sum):
            st.caption("Step 13: Final summary row")
            st.dataframe(df_sum, use_container_width=True)

    with colB:
        df_boot = _read_csv_if_exists(boot_winners_csv)
        if df_boot is not None and len(df_boot):
            st.caption("Step 12: Bootstrap winner fractions")
            show = df_boot.copy()
            for c in ["P_hr", "wins", "frac"]:
                if c in show.columns:
                    show[c] = _safe_num(show[c])
            show = show.sort_values("frac", ascending=False)
            st.dataframe(show, use_container_width=True, height=280)
            st.download_button("Download bootstrap winner fractions", df_boot.to_csv(index=False).encode("utf-8"),
                               file_name=f"{selected}_bootstrap_winners.csv", mime="text/csv")
        else:
            st.info("No step12_bootstrap_winner_fractions.csv found for this object (optional).")

        df_fam = _read_csv_if_exists(fam_csv)
        if df_fam is not None and len(df_fam):
            st.caption("Step 11: Family table")
            st.dataframe(df_fam, use_container_width=True)
        else:
            st.info("No step11_families.csv found (optional). If you want it, write fam_df to CSV in Step 11.")
