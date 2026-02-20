# app.py
# ==========================================================
# ATLAST Rotation Dashboard — Simple/Research UI (republish)
# Includes:
#   - Header row with Simple/Research toggle
#   - 3-panel fold (P/2 vs P vs 2P) ALWAYS (simple trust-builder)
#   - Periodogram with alias highlights (P, P/2, 2P + bootstrap alias peaks)
#   - Research mode extras: bootstrap histogram + candidate table + residuals + downloads
#   - Fix: P/2 and 2P never “compound” (always relative to base adopted P)
#   - Fix: selecting a row updates the whole page (session_state selection)
# ==========================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional: Lomb-Scargle periodogram
try:
    from astropy.timeseries import LombScargle  # type: ignore
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False

# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="ATLAST Rotation Dashboard", layout="wide")

# --- Mode toggle (header row) ---
h1, h2 = st.columns([0.75, 0.25])
with h1:
    st.title("ATLAST Asteroid Rotation Dashboard")
with h2:
    mode = st.radio(
        "Mode",
        ["Simple", "Research"],
        horizontal=True,
        label_visibility="collapsed",
        key="ui_mode",
    )
is_research = (mode == "Research")

# -------------------------
# Config paths
# -------------------------
# Master summary table across objects
MASTER_FILE = "master_results_clean.csv"
OVERRIDES_FILE = "overrides.csv"

# Photometry per object:
# Recommended structure:
#   data/
#     photometry/
#       <Designation>/
#         bq-results.csv
#         step11_period_table.csv                         (optional)
#         step12_bootstrap_winner_fractions.csv           (optional)
#         step13_final_summary.csv                        (optional)
DATA_ROOT = Path("data")
PHOT_ROOT = DATA_ROOT / "photometry"

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return _read_csv(path)
    except Exception:
        return None
    return None

def _obj_dir(designation: str) -> Path:
    return PHOT_ROOT / str(designation)

def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float(default)

def _safe_int(x, default=np.nan) -> int | float:
    try:
        v = int(x)
        return v
    except Exception:
        return default

def _rel_err(a, b) -> float:
    return abs(a - b) / max(abs(b), 1e-12)

def _clean_band(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _download_df_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )

# -------------------------
# Load master + overrides
# -------------------------
@st.cache_data(show_spinner=False)
def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_FILE)
    if "Designation" not in df.columns:
        # best effort: common alternatives
        for c in ["provid", "PROVID", "designation", "name", "object_id"]:
            if c in df.columns:
                df = df.rename(columns={c: "Designation"})
                break
    return df

@st.cache_data(show_spinner=False)
def load_overrides() -> pd.DataFrame:
    if not Path(OVERRIDES_FILE).exists():
        return pd.DataFrame()
    return pd.read_csv(OVERRIDES_FILE)

master = load_master()
overrides = load_overrides()

# -------------------------
# Selection UI
# -------------------------
st.markdown("### Object selection")

# Persist selection across reruns
if "selected_designation" not in st.session_state:
    st.session_state.selected_designation = None

left, right = st.columns([0.55, 0.45])

with left:
    # Filter/search
    q = st.text_input("Search (Designation contains)", value="", placeholder="e.g., 2025 MA46")
    df_view = master.copy()

    if q.strip():
        df_view = df_view[df_view["Designation"].astype(str).str.contains(q.strip(), case=False, na=False)]

    # Show master table (select a row -> update whole page)
    # Use a stable index to map selection
    df_view = df_view.reset_index(drop=True)
    st.caption("Click a row (or use the selector on the right).")

    # Streamlit dataframe selection API differs by version; use a robust pattern:
    event = st.dataframe(
        df_view,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="master_table",
    )

    try:
        sel_rows = event.selection.rows  # type: ignore[attr-defined]
    except Exception:
        sel_rows = []

    if sel_rows:
        picked = df_view.iloc[int(sel_rows[0])]["Designation"]
        st.session_state.selected_designation = str(picked)

with right:
    # Fallback selector
    designations = master["Designation"].astype(str).tolist()
    default_idx = 0
    if st.session_state.selected_designation in designations:
        default_idx = designations.index(st.session_state.selected_designation)

    picked2 = st.selectbox(
        "Or pick an object",
        options=designations,
        index=default_idx if designations else 0,
        key="designation_selectbox",
    )
    st.session_state.selected_designation = str(picked2) if picked2 else st.session_state.selected_designation

designation = st.session_state.selected_designation
if not designation:
    st.stop()

# Pull the selected summary row
row = master[master["Designation"].astype(str) == str(designation)]
row = row.iloc[0].to_dict() if len(row) else {}

# -------------------------
# Resolve adopted/base period + K
# -------------------------
# Prefer explicit adopted period column names; fall back to whatever you have.
P_adopt = np.nan
for c in ["Adopted period (hr)", "P_adopt_hr", "P_final_hr", "P_hr_adopt", "P_hr"]:
    if c in row and pd.notna(row[c]):
        P_adopt = _safe_float(row[c], np.nan)
        break

K_adopt = np.nan
for c in ["Adopted K", "K_adopt", "K_best", "K"]:
    if c in row and pd.notna(row[c]):
        K_adopt = _safe_int(row[c], np.nan)
        break

# IMPORTANT: base period is fixed (prevents compounding)
P_base = float(P_adopt) if np.isfinite(P_adopt) and P_adopt > 0 else np.nan

# -------------------------
# Load per-object photometry + aux tables
# -------------------------
obj_dir = _obj_dir(designation)
photo_path = obj_dir / "bq-results.csv"
df_photo = _read_csv_if_exists(photo_path)

period_table_path = obj_dir / "step11_period_table.csv"
df_pt = _read_csv_if_exists(period_table_path)

boot_path = obj_dir / "step12_bootstrap_winner_fractions.csv"
df_boot = _read_csv_if_exists(boot_path)

step13_path = obj_dir / "step13_final_summary.csv"
df_step13 = _read_csv_if_exists(step13_path)

# -------------------------
# Key results panel
# -------------------------
st.markdown("---")
st.markdown("## Key results")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Designation", str(designation))
k2.metric("Adopted P (hr)", f"{P_base:.6f}" if np.isfinite(P_base) else "—")
k3.metric("Adopted K", f"{int(K_adopt)}" if np.isfinite(K_adopt) else "—")

# Reliability (if present)
reliab = row.get("Reliability", row.get("reliability", "—"))
k4.metric("Reliability", str(reliab) if pd.notna(reliab) else "—")

# Optional: show Step13 row if present
if df_step13 is not None and len(df_step13):
    with st.expander("Step 13 final summary row (from step13_final_summary.csv)", expanded=False):
        st.dataframe(df_step13, use_container_width=True)
        _download_df_button(df_step13, f"{designation}_step13_final_summary.csv", "Download Step 13 summary CSV")

# -------------------------
# Photometry tab
# -------------------------
st.markdown("---")
st.markdown("## Photometry & validation")

if df_photo is None or len(df_photo) == 0:
    st.error(f"Missing photometry file: {photo_path}")
    st.stop()

# ---- Infer columns
dfp = df_photo.copy()

# Try to detect time column in hours
# Prefer your pipeline column names; fall back to converting MJD to hours relative
TIME_CANDIDATES = ["t_hr", "t_hours", "t", "time_hr", "time_hours"]
time_col = next((c for c in TIME_CANDIDATES if c in dfp.columns), None)

if time_col is None:
    # Common: mjd / obstime_mjd
    if "mjd" in dfp.columns:
        t = pd.to_numeric(dfp["mjd"], errors="coerce").to_numpy(float)
        t0 = np.nanmin(t) if np.isfinite(t).any() else 0.0
        dfp["t_hr"] = (t - t0) * 24.0
        time_col = "t_hr"
    elif "obstime_mjd" in dfp.columns:
        t = pd.to_numeric(dfp["obstime_mjd"], errors="coerce").to_numpy(float)
        t0 = np.nanmin(t) if np.isfinite(t).any() else 0.0
        dfp["t_hr"] = (t - t0) * 24.0
        time_col = "t_hr"
    else:
        st.error("Could not find a time column. Expected t_hr or mjd/obstime_mjd.")
        st.stop()

# Choose a magnitude column for plotting/folding
MAG_CANDIDATES = [
    "mag_geo_bandcenter", "mag_geo", "mag_detrend", "mag", "mag_corr"
]
mag_col = next((c for c in MAG_CANDIDATES if c in dfp.columns), None)
if mag_col is None:
    st.error(f"Could not find a magnitude column. Tried: {MAG_CANDIDATES}")
    st.stop()

# Band column
if "band" not in dfp.columns:
    dfp["band"] = "x"
dfp["band"] = _clean_band(dfp["band"])

# Use-only finite
df_plot = dfp[[time_col, mag_col, "band"]].copy()
df_plot[time_col] = pd.to_numeric(df_plot[time_col], errors="coerce")
df_plot[mag_col] = pd.to_numeric(df_plot[mag_col], errors="coerce")
df_plot = df_plot.dropna(subset=[time_col, mag_col])

t_hr = df_plot[time_col].to_numpy(float)
mag = df_plot[mag_col].to_numpy(float)
bands = df_plot["band"].to_numpy(str)

# -------------------------
# Period control
# -------------------------
st.markdown("### Period control")

if not np.isfinite(P_base):
    st.warning("No adopted/base period found in master table. The fold panels will still show if you set a period below.")
    P_base = float(np.nanmedian([p for p in [row.get("LS peak period (hr)", np.nan)] if pd.notna(p)])) if pd.notna(row.get("LS peak period (hr)", np.nan)) else np.nan

# In Simple mode: fixed to adopted P for trust/clarity.
# In Research mode: allow a slider to explore around base P.
P_calc = P_base

if is_research and np.isfinite(P_base) and P_base > 0:
    pct = st.slider("Explore period around adopted P (±%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    lo = max(1e-6, P_base * (1.0 - pct / 100.0))
    hi = P_base * (1.0 + pct / 100.0)
    P_calc = st.slider(
        "Fold period (hr)",
        min_value=float(lo),
        max_value=float(hi),
        value=float(P_base),
        step=float((hi - lo) / 400.0) if hi > lo else 1e-6,
    )
elif not np.isfinite(P_calc) or P_calc <= 0:
    P_calc = st.number_input("Set fold period (hr)", min_value=1e-6, value=5.0)

# NOTE: P/2 and 2P are ALWAYS derived from base adopted period, not chained clicks.
P_half = 0.5 * float(P_calc)
P_two = 2.0 * float(P_calc)

# -------------------------
# 3-panel fold (P/2 vs P vs 2P)
# -------------------------
st.markdown("### Quick validation (3-panel fold: P/2 vs P vs 2P)")

def plot_fold(ax, t_hr, mag, bands, P_hr, title):
    phase = (t_hr / float(P_hr)) % 1.0
    uniq = sorted(np.unique(bands).tolist())
    for b in uniq:
        m = (bands == b)
        ax.scatter(phase[m], mag[m], s=10, label=b)
        # show second cycle for readability
        ax.scatter(phase[m] + 1.0, mag[m], s=10)
    ax.invert_yaxis()
    ax.set_xlabel("Phase")
    ax.set_ylabel(mag_col)
    ax.set_title(title)

cols = st.columns(3)
periods = [P_half, P_calc, P_two]
titles = [f"P/2 = {P_half:.6f} h", f"P = {P_calc:.6f} h", f"2P = {P_two:.6f} h"]

for col, P_hr, title in zip(cols, periods, titles):
    with col:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        plot_fold(ax, t_hr=t_hr, mag=mag, bands=bands, P_hr=P_hr, title=title)
        ax.legend(fontsize=7)
        st.pyplot(fig, clear_figure=True)

# -------------------------
# Periodogram with alias highlights
# -------------------------
st.markdown("### Periodogram")

alias_periods: list[float] = []

# Load bootstrap winners (optional) for alias peaks
if df_boot is not None and {"P_hr", "frac"}.issubset(df_boot.columns):
    tmp = df_boot.copy()
    tmp["P_hr"] = pd.to_numeric(tmp["P_hr"], errors="coerce")
    tmp["frac"] = pd.to_numeric(tmp["frac"], errors="coerce")
    tmp = tmp.dropna(subset=["P_hr", "frac"]).sort_values("frac", ascending=False)

    # take top periods excluding ~P itself, keep up to 3
    for p in tmp["P_hr"].head(8).to_numpy(float):
        if np.isfinite(p) and np.isfinite(P_calc) and P_calc > 0:
            if _rel_err(p, P_calc) > 0.002:
                alias_periods.append(float(p))
    alias_periods = alias_periods[:3]

if not HAS_ASTROPY:
    st.info("Astropy not found in this environment, so Lomb–Scargle periodogram is disabled.")
else:
    # Build a Lomb-Scargle on (t_hr, mag)
    # Use a period grid around P_calc for speed + trust-building clarity
    if np.isfinite(P_calc) and P_calc > 0:
        # Wide grid for simple mode, narrower if research slider already controls
        pmin = max(0.2, 0.25 * P_calc)
        pmax = 4.0 * P_calc
    else:
        pmin, pmax = 0.2, 48.0

    n_grid = 4000 if is_research else 2500
    periods = np.linspace(pmin, pmax, n_grid)
    freqs = 1.0 / periods

    # LombScargle expects time in same unit; we use hours
    ls = LombScargle(t_hr, mag, nterms=1)
    power = ls.power(freqs)

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.plot(periods, power, linewidth=1.2)
    ax.set_xlabel("Period (hr)")
    ax.set_ylabel("LS power")
    ax.set_title("Lomb–Scargle periodogram")

    # Alias highlights
    if np.isfinite(P_calc) and P_calc > 0:
        ax.axvline(P_calc, linestyle="--", linewidth=1.2, label="P")
        ax.axvline(0.5 * P_calc, linestyle=":", linewidth=1.2, label="P/2")
        ax.axvline(2.0 * P_calc, linestyle=":", linewidth=1.2, label="2P")

    for j, p in enumerate(alias_periods):
        ax.axvline(p, linestyle="-.", linewidth=1.0, label=f"boot alias {j+1}")

    ax.set_xlim(pmin, pmax)
    ax.legend(fontsize=8, ncol=6)
    st.pyplot(fig, clear_figure=True)

# -------------------------
# Research diagnostics
# -------------------------
if is_research:
    st.markdown("---")
    st.markdown("## Research diagnostics")

    # ---- Bootstrap histogram + top winners table
    st.markdown("### Bootstrap winner distribution")
    if df_boot is not None and {"P_hr", "wins", "frac"}.issubset(df_boot.columns):
        tmp = df_boot.copy()
        tmp["P_hr"] = pd.to_numeric(tmp["P_hr"], errors="coerce")
        tmp["wins"] = pd.to_numeric(tmp["wins"], errors="coerce")
        tmp["frac"] = pd.to_numeric(tmp["frac"], errors="coerce")
        tmp = tmp.dropna(subset=["P_hr", "wins"]).sort_values("P_hr")

        # Expand wins into samples for histogram (OK for N_BOOT ~ few hundred)
        wins_int = tmp["wins"].fillna(0).astype(int).to_numpy()
        samples = np.repeat(tmp["P_hr"].to_numpy(float), wins_int)

        bins = st.slider("Histogram bins", min_value=10, max_value=120, value=40, step=1)
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        if len(samples):
            ax.hist(samples, bins=bins)
        ax.set_xlabel("Winning period (hr)")
        ax.set_ylabel("Count")
        ax.set_title("Bootstrap winners histogram")
        st.pyplot(fig, clear_figure=True)

        st.caption("Top bootstrap winners")
        st.dataframe(
            tmp.sort_values("wins", ascending=False).head(12)[["P_hr", "wins", "frac"]],
            use_container_width=True,
        )
        _download_df_button(tmp, f"{designation}_step12_bootstrap_winners.csv", "Download bootstrap winner table CSV")
    else:
        st.info(f"No bootstrap winner table found at: {boot_path}")

    # ---- Candidate period table
    st.markdown("### Candidate periods (Step 11 table)")
    if df_pt is not None and len(df_pt):
        pt = df_pt.copy()
        # common expected columns:
        for c in ["P_hr", "BIC", "wrms", "K_best", "amp_ptp_mag"]:
            if c in pt.columns:
                pt[c] = pd.to_numeric(pt[c], errors="coerce")
        # show best few
        show_cols = [c for c in ["P_hr", "K_best", "BIC", "wrms", "amp_ptp_mag"] if c in pt.columns]
        if show_cols:
            pt2 = pt.sort_values([c for c in ["BIC", "wrms"] if c in pt.columns], ascending=True)
            st.dataframe(pt2[show_cols].head(50), use_container_width=True)
        else:
            st.dataframe(pt.head(50), use_container_width=True)

        _download_df_button(pt, f"{designation}_step11_period_table.csv", "Download candidate period table CSV")
    else:
        st.info(f"No candidate period table found at: {period_table_path}")

    # ---- Residuals (lightweight “trust” diagnostics)
    st.markdown("### Residual diagnostics (quick)")
    # Simple detrend: subtract per-band median, then show residuals vs time after folding model is not available here.
    # If you already write Step 13 residual files, prefer loading those. Otherwise show a minimal check:
    by_band = df_plot.copy()
    by_band["mag_resid"] = by_band[mag_col] - by_band.groupby("band")[mag_col].transform("median")

    fig, ax = plt.subplots(figsize=(10.0, 3.2))
    ax.scatter(by_band[time_col].to_numpy(float), by_band["mag_resid"].to_numpy(float), s=10)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel(f"{time_col} (hr)")
    ax.set_ylabel("mag - band median (mag)")
    ax.set_title("Residuals vs time (band-median detrend)")
    st.pyplot(fig, clear_figure=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    r = by_band["mag_resid"].to_numpy(float)
    r = r[np.isfinite(r)]
    ax.hist(r, bins=30)
    ax.set_xlabel("Residual (mag)")
    ax.set_ylabel("Count")
    ax.set_title("Residual histogram (band-median detrend)")
    st.pyplot(fig, clear_figure=True)

# -------------------------
# Downloads (always useful)
# -------------------------
st.markdown("---")
st.markdown("## Downloads")

d1, d2, d3 = st.columns(3)
with d1:
    _download_df_button(df_photo, f"{designation}_bq-results.csv", "Download photometry CSV")
with d2:
    if df_pt is not None:
        _download_df_button(df_pt, f"{designation}_step11_period_table.csv", "Download candidate table CSV")
    else:
        st.caption("Candidate table not found.")
with d3:
    if df_boot is not None:
        _download_df_button(df_boot, f"{designation}_step12_bootstrap_winners.csv", "Download bootstrap winners CSV")
    else:
        st.caption("Bootstrap winners not found.")

# Footer
st.caption(
    "Simple mode shows: Key results + 3-panel fold + periodogram (with alias lines). "
    "Research mode adds: bootstrap histogram + candidate table + residual diagnostics + downloads."
)
