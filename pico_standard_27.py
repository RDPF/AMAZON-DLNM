#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ============================================================
# DLNM PM2.5 – EPISODE-BASED (TOP 5% PM2.5 DAYS)  — FULL PIPELINE (UPDATED)
#
# OUTPUT RULES (NATURE-LIKE):
#   - All figure labels in English
#   - Output root: C:\NEWVERSIONGASPARRINI\TOP5PCT_PM25\
#   - CSV column names without underscores (camelCase)
#   - Graph titles kept minimal (captions external; no huge one-line titles)
#
# OUTPUT ROOT:
#   C:\NEWVERSIONGASPARRINI\TOP5PCT_PM25\
#       RR_LAG\          (RR vs lag 0..7 per state + OVERALL)
#       RR_PM\           (RR vs PM 0..100 step 5, curves by lag 0..7)
#       RR_PM_LAG0\      (RR vs PM for lag 0 ONLY, with CI ribbon)
#       TABLES\          (CSVs: RR_LAG, RR_PM, RR_PM_LAG0, summaries, forest, etc.)
#       GUIDE.TXT        (text guide describing folder contents)
# ============================================================

import os
import re
import glob
import math
import warnings
from typing import Dict, List, Tuple

# ---------- MUST BE FIRST for Jupyter on Windows ----------
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
os.environ.setdefault("RPY2_CONSOLE_ENCODING", "cp1252")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
print("Matplotlib backend:", matplotlib.get_backend())

# Make figures closer to "Nature" layout (small fonts, vector-friendly)
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "font.family": "DejaVu Sans",
})

# ============================================================
# RPY2 BULLETPROOF UNICODE PATCH (Windows)
# ============================================================

def patch_rpy2_unicode_windows() -> bool:
    ok = True
    try:
        import rpy2.rinterface_lib.callbacks as cb

        def _safe_text(x):
            if isinstance(x, bytes):
                for enc in ("utf-8", "cp1252", "latin-1"):
                    try:
                        return x.decode(enc, errors="replace")
                    except Exception:
                        pass
                return x.decode("latin-1", errors="replace")
            return str(x)

        cb.consolewrite_print = lambda x: print(_safe_text(x), end="")
        cb.consolewrite_warnerror = lambda x: print(_safe_text(x), end="")
    except Exception as e:
        print("⚠️ console callback patch failed:", e)
        ok = False

    try:
        import rpy2.rinterface_lib.conversion as rconv
        from cffi import FFI
        _ffi = FFI()
        orig = getattr(rconv, "_cchar_to_str", None)

        def _cchar_to_str_fallback(c, encoding: str) -> str:
            if orig is not None:
                try:
                    return orig(c, encoding)
                except UnicodeDecodeError:
                    pass
                except Exception:
                    pass
            try:
                raw = _ffi.string(c)
            except Exception:
                return ""
            for enc in ("utf-8", "cp1252", "latin-1"):
                try:
                    return raw.decode(enc, errors="replace")
                except Exception:
                    pass
            return raw.decode("latin-1", errors="replace")

        rconv._cchar_to_str = _cchar_to_str_fallback
    except Exception as e:
        print("⚠️ error-message decode patch failed:", e)
        ok = False

    return ok

_ = patch_rpy2_unicode_windows()

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# R locale/encoding best-effort
try:
    ro.r('options(encoding="UTF-8")')
    ro.r('try(Sys.setlocale(category="LC_ALL", locale="C"), silent=TRUE)')
    ro.r('try(Sys.setlocale(category="LC_MESSAGES", locale="C"), silent=TRUE)')
except Exception as e:
    print("⚠️ R locale setup warning:", e)

# ============================================================
# GLOBAL CONFIG
# ============================================================

HOSP_BASE_DIR = (
    r"C:\dados3\DADOS - HOSP_ENV_LEGAL_AMAZON_DATA-20251201T163911Z-1-001"
    r"\DADOS - HOSP_ENV_LEGAL_AMAZON_DATA\Full"
)
MORT_BASE_DIR = r"C:\dados2\MORT"

EPISODES_CSV = r"C:\busca5\episodes_simple.csv"

# UPDATED OUTPUT ROOT (NATURE PIPELINE ROOT)
GLOBAL_OUT_ROOT = r"C:\NEWVERSIONGASPARRINI"
os.makedirs(GLOBAL_OUT_ROOT, exist_ok=True)

DATE_COL = "DATA"

PM_PREFIX = "WF_PM2.5_POND_"
PM_CANON = "WF_PM2.5_POND_STATE"

TEMP_COL = "TEMP_MEAN_C"
RH_COL = "RH_MEAN_PCT"
WIND_COL = "WIND_SPEED_MEAN_MPS"

TEMP_MA_COL = "TEMP_MA21"
RH_MA_COL = "RH_MA7"
WIND_MA_COL = "WIND_MA7"

PRECIP_MEAN_COL = "PRECIP_MEAN"
PRECIP_MAX_COL  = "PRECIP_MAX"
PRECIP_MIN_COL  = "PRECIP_MIN"

# Totals (kept)
HOSP_RESP_TOTAL = "HOSP_RESP_TOTAL"
HOSP_CIRC_TOTAL = "HOSP_CIRC_TOTAL"
MORT_RESP_TOTAL = "MORT_RESP_TOTAL"
MORT_CIRC_TOTAL = "MORT_CIRC_TOTAL"

# UPDATED: only 0..7
LAG_MAX = 7
TIME_DF = 7

# RR vs PM config
PM_GRID = np.arange(0, 101, 5).astype(float)  # 0..100 step 5
PM_CEN = 5.0

FIG_DPI = 300
LINE_BLUE = "#0072B2"
SHADE_BLUEGRAY = "#A6BDD7"
SHADE_ALPHA = 0.55
MAX_PATH_SAFE = 220

SCENARIO_UNIQUE = "TOP5PCT_PM25"

# ============================================================
# UTILS
# ============================================================

def log(msg: str):
    print(msg)

def assert_exists(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found: {path}")

def sanitize_filename(s: str, max_len: int = 150) -> str:
    s = str(s)
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"[\x00-\x1f]", "_", s)
    s = s.strip().strip(".")
    if len(s) > max_len:
        s = s[:max_len]
    return s

def safe_out_base(out_base: str) -> str:
    out_base = os.path.normpath(out_base)
    d = os.path.dirname(out_base)
    os.makedirs(d, exist_ok=True)
    if len(out_base) > MAX_PATH_SAFE:
        d = os.path.dirname(out_base)
        base = sanitize_filename(os.path.basename(out_base), max_len=80)
        out_base = os.path.join(d, base)
    return out_base

def save_fig_all(fig, out_base: str):
    out_base = safe_out_base(out_base)
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(out_base + ext, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved:", out_base + ".png/.pdf/.svg")

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def list_excel_files(base_dir: str, pattern: str) -> List[str]:
    assert_exists(base_dir, "Base directory")
    files = sorted(glob.glob(os.path.join(base_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files in {base_dir} matching: {pattern}")
    return files

def infer_state_from_hosp_filename(path: str) -> str:
    return os.path.basename(path).split("_HOSPITALIZATION")[0].strip().upper()

def infer_state_from_mort_filename(path: str) -> str:
    return os.path.basename(path).replace("_MORTALITY_FULL_DATA.xlsx", "").strip().upper()

def self_test_saving(OUT_ROOT: str):
    test_base = os.path.join(OUT_ROOT, "_TEST_SAVEFIG")
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.plot([0, 1, 2], [0, 1, 0], lw=2, color=LINE_BLUE)
    ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save_fig_all(fig, test_base)
    log("✅ Self-test figure writing OK.")

# ============================================================
# NATURE-STYLE COLUMN RENAMING (NO UNDERSCORES)
# ============================================================

def rename_for_nature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with column names adapted to Nature-style:
      - No underscores
      - camelCase for multi-word technical names
      - Keep short tokens like RR, PM, etc.
    """
    mapping = {
        # time / index
        "TIME_INDEX": "timeIndex",
        "DOW": "dayOfWeek",
        # meteo MAs
        "TEMP_MA21": "tempMA21",
        "RH_MA7": "rhMA7",
        "WIND_MA7": "windMA7",
        "PRECIP_MEAN": "precipMean",
        "PRECIP_MAX": "precipMax",
        "PRECIP_MIN": "precipMin",
        # PM
        "WF_PM2.5_POND_STATE": "wfPm25PondState",
        # episodes
        "EPISODE_ID": "episodeId",
        "PEAK_DATE": "peakDate",
        "STATE": "state",
        # date
        "DATA": "date",
        # summaries
        "outcome_group": "outcomeGroup",
        "n_events": "nEvents",
        "n_valid_days": "nValidDays",
        "peak_rr": "peakRR",
        "lag_peak": "lagPeak",
        "cum_rr": "cumRR",
        "cum_low": "cumLow",
        "cum_high": "cumHigh",
        "meta_tau2": "metaTau2",
        "meta_Q": "metaQ",
        "meta_df": "metaDf",
        "meta_I2": "metaI2",
        "p_overall": "pOverall",
        "p_heterogeneity": "pHeterogeneity",
        # forest
        "CIlow": "ciLow",
        "CIhigh": "ciHigh",
        "logRR": "logRR",
        "SE": "se",
        "weight_norm": "weightNorm",
        "p_value": "pValue",
        "p_value_difference": "pValueDifference",
        "isOverall": "isOverall",
    }

    new_cols = {}
    for c in df.columns:
        if c in mapping:
            new_cols[c] = mapping[c]
        else:
            sc = str(c)
            if "_" in sc:
                parts = sc.split("_")
                base = parts[0].lower()
                tail = "".join(p.capitalize() for p in parts[1:])
                new_cols[c] = base + tail
            else:
                new_cols[c] = sc
    return df.rename(columns=new_cols)

def save_csv_nature(df: pd.DataFrame, path: str):
    """
    Save DataFrame with Nature-style column names (no underscores), UTF-8 BOM,
    without altering the original df in memory.
    """
    df_out = rename_for_nature(df.copy())
    df_out.to_csv(path, index=False, encoding="utf-8-sig")

# ============================================================
# OUTCOME DISCOVERY (NEW)
# ============================================================

SEX_TOKENS = ("_FEM", "_FEMALE", "_MASC", "_MALE")
AGE_HINT_TOKENS = (
    "_AGE", "_IDADE", "_ID_", "_AGE_",
    "_0_", "_5_", "_10_", "_15_", "_20_", "_25_", "_30_", "_35_",
    "_40_", "_45_", "_50_", "_55_", "_60_", "_65_", "_70_", "_75_",
    "_80_", "_85_", "_90_", "_PLUS", "_MAIS",
)

def classify_outcome(col: str) -> str:
    c = col.upper()
    if c.endswith("_TOTAL"):
        return "TOTAL"
    if any(tok in c for tok in SEX_TOKENS):
        return "SEX"
    if any(tok in c for tok in AGE_HINT_TOKENS) or re.search(r"_\d{1,2}(_\d{1,2})?(\b|_)", c):
        return "AGE"
    return "OTHER"

def discover_outcomes_from_df(df: pd.DataFrame, dataset_type: str) -> List[str]:
    if df is None or df.empty:
        return []

    if dataset_type == "HOSP":
        prefix_ok = ("HOSP_RESP_", "HOSP_CIRC_")
    else:
        prefix_ok = ("MORT_RESP_", "MORT_CIRC_")

    cols = [c for c in df.columns if any(str(c).upper().startswith(p) for p in prefix_ok)]
    # ensure totals are included if present
    must = [HOSP_RESP_TOTAL, HOSP_CIRC_TOTAL] if dataset_type == "HOSP" else [MORT_RESP_TOTAL, MORT_CIRC_TOTAL]
    for m in must:
        if m in df.columns and m not in cols:
            cols.append(m)

    cols_sorted = sorted(
        cols,
        key=lambda x: ({"TOTAL": 0, "SEX": 1, "AGE": 2, "OTHER": 3}.get(classify_outcome(x), 9), str(x))
    )
    return cols_sorted

def build_core_outcomes(
    hosp_by_state: Dict[str, pd.DataFrame],
    mort_by_state: Dict[str, pd.DataFrame],
) -> Dict[str, List[str]]:
    hosp_cols = set()
    for _, df in hosp_by_state.items():
        hosp_cols.update(discover_outcomes_from_df(df, "HOSP"))

    mort_cols = set()
    for _, df in mort_by_state.items():
        mort_cols.update(discover_outcomes_from_df(df, "MORT"))

    hosp_list = sorted(
        list(hosp_cols),
        key=lambda x: ({"TOTAL": 0, "SEX": 1, "AGE": 2, "OTHER": 3}.get(classify_outcome(x), 9), str(x)),
    )
    mort_list = sorted(
        list(mort_cols),
        key=lambda x: ({"TOTAL": 0, "SEX": 1, "AGE": 2, "OTHER": 3}.get(classify_outcome(x), 9), str(x)),
    )

    return {"HOSP": hosp_list, "MORT": mort_list}

# ============================================================
# DATA PREP (precipitation + moving averages)
# ============================================================

def extract_precip_from_VWX(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)

    # 1) position V/W/X (indices 21..23)
    if len(cols) >= 24:
        idx_map = {PRECIP_MEAN_COL: 21, PRECIP_MAX_COL: 22, PRECIP_MIN_COL: 23}
        for target, idx in idx_map.items():
            if idx < len(cols):
                df[target] = pd.to_numeric(df[cols[idx]], errors="coerce")

    # 2) fallback by header keyword
    key_map = {
        PRECIP_MEAN_COL: ["mean", "media", "média", "avg", "average"],
        PRECIP_MAX_COL:  ["max", "máx", "maxima", "máxima", "maximum"],
        PRECIP_MIN_COL:  ["min", "mín", "minima", "mínima", "minimum"],
    }
    precip_candidates = [c for c in cols if re.search(r"(precip|chuva|rain|pluv)", str(c), flags=re.I)]
    if precip_candidates:
        for target, keys in key_map.items():
            if target in df.columns and df[target].notna().sum() > 50:
                continue
            for c in precip_candidates:
                if any(re.search(k, str(c), flags=re.I) for k in keys):
                    df[target] = pd.to_numeric(df[c], errors="coerce")
                    break

    for c in [PRECIP_MEAN_COL, PRECIP_MAX_COL, PRECIP_MIN_COL]:
        if c not in df.columns:
            df[c] = np.nan

    return df

def apply_end_of_year_filter_hosp(df: pd.DataFrame) -> pd.DataFrame:
    mask_keep = ~(
        ((df[DATE_COL].dt.month == 12) & (df[DATE_COL].dt.day >= 24)) |
        ((df[DATE_COL].dt.month == 1) & (df[DATE_COL].dt.day <= 6))
    )
    return df.loc[mask_keep].reset_index(drop=True)

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if TEMP_COL in df.columns:
        df[TEMP_MA_COL] = df[TEMP_COL].rolling(window=21, min_periods=1).mean()
    else:
        df[TEMP_MA_COL] = np.nan

    if RH_COL in df.columns:
        df[RH_MA_COL] = df[RH_COL].rolling(window=7, min_periods=1).mean()
    else:
        df[RH_MA_COL] = np.nan

    if WIND_COL in df.columns:
        df[WIND_MA_COL] = df[WIND_COL].rolling(window=7, min_periods=1).mean()
    else:
        df[WIND_MA_COL] = np.nan

    return df

def prepare_dataframe(path: str, dataset_type: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if DATE_COL not in df.columns:
        raise ValueError(f"Missing '{DATE_COL}' in {os.path.basename(path)}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    pm_cols = [c for c in df.columns if str(c).startswith(PM_PREFIX)]
    if not pm_cols:
        raise ValueError(f"No PM column starting '{PM_PREFIX}' in {os.path.basename(path)}")
    df = df.rename(columns={pm_cols[0]: PM_CANON})

    # numeric coercion: PM + meteo + all outcome-like cols
    base_cols = [PM_CANON, TEMP_COL, RH_COL, WIND_COL]
    out_cols = discover_outcomes_from_df(df, dataset_type)
    coerce_numeric(df, base_cols + out_cols)

    df = extract_precip_from_VWX(df)

    if dataset_type == "HOSP":
        df = apply_end_of_year_filter_hosp(df)

    df = add_moving_averages(df)
    return df

# ============================================================
# EPISODES: LOAD (TOP 5% PM)
# ============================================================

def load_episodes_all(csv_path: str) -> pd.DataFrame:
    assert_exists(csv_path, "Episodes CSV")

    ep = pd.read_csv(csv_path)
    ep.columns = [str(c).strip() for c in ep.columns]

    state_col = next((c for c in ep.columns if c.lower() in ("state", "uf", "estado")), None)
    if state_col is None:
        raise ValueError("Episodes CSV must contain a state/UF column (state/UF/estado).")

    peak_col = next(
        (c for c in ep.columns
         if c.lower() in ("peak_day", "peak_date", "peak",
                          "max_date", "dia_pico", "pico", "data")),
        None,
    )
    if peak_col is None:
        raise ValueError("Episodes CSV must contain a peak date column (peak_date/peak_day/pico/DATA...).")

    ep[peak_col] = pd.to_datetime(ep[peak_col], errors="coerce")
    ep["STATE"] = ep[state_col].astype(str).str.strip().str.upper()
    ep["PEAK_DATE"] = ep[peak_col]

    ep = ep.dropna(subset=["STATE", "PEAK_DATE"]).reset_index(drop=True)
    ep["EPISODE_ID"] = np.arange(1, len(ep) + 1).astype(int)
    return ep

def add_event_indicator(
    df: pd.DataFrame,
    episodes_state: pd.DataFrame,
    lag_after: int = LAG_MAX,
) -> pd.DataFrame:
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df["event_extreme"] = 0

    if episodes_state is None or episodes_state.empty:
        return df

    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()

    for _, e in episodes_state.iterrows():
        peak = pd.to_datetime(e["PEAK_DATE"])
        if pd.isna(peak):
            continue
        if peak < min_date:
            continue
        if peak + pd.Timedelta(days=lag_after) > max_date:
            continue
        mask = df[DATE_COL] == peak
        if mask.any():
            df.loc[mask, "event_extreme"] = 1

    return df

# ============================================================
# R: (A) EVENT-BASED DLNM (RR vs lag)
# ============================================================

R_CODE = r"""
suppressPackageStartupMessages({
  library(dlnm)
  library(splines)
  library(stats)
})

rr_lag_events_threshold <- function(df, outcome, lag_max, df_time = 7) {

  required <- c(outcome, "event_extreme",
                "TIME_INDEX", "DOW",
                "TEMP_MA21", "RH_MA7", "WIND_MA7")

  if (!all(required %in% names(df))) {
    miss <- setdiff(required, names(df))
    stop(paste("Missing required columns:", paste(miss, collapse = ", ")))
  }

  df2 <- df[complete.cases(df[, required]), , drop = FALSE]
  if (nrow(df2) < 100) {
    lags <- 0:lag_max
    return(data.frame(lag = lags,
                      rr = NA_real_, low = NA_real_, high = NA_real_,
                      cum_rr = NA_real_, cum_low = NA_real_, cum_high = NA_real_))
  }

  if (sum(df2$event_extreme > 0, na.rm = TRUE) < 3) {
    lags <- 0:lag_max
    return(data.frame(lag = lags,
                      rr = NA_real_, low = NA_real_, high = NA_real_,
                      cum_rr = NA_real_, cum_low = NA_real_, cum_high = NA_real_))
  }

  res <- tryCatch({

    cb_ev <- crossbasis(df2$event_extreme,
                        lag = lag_max,
                        argvar = list(fun = "lin"),
                        arglag = list(fun = "ns", df = 4))

    fit <- glm(df2[[outcome]] ~ cb_ev +
                 ns(df2[["TIME_INDEX"]], df_time) +
                 factor(df2[["DOW"]]) +
                 ns(df2[["TEMP_MA21"]], 4) +
                 ns(df2[["RH_MA7"]], 4) +
                 ns(df2[["WIND_MA7"]], 3),
               family = quasipoisson(),
               data = df2)

    cp <- crosspred(cb_ev, fit,
                    at = c(0, 1),
                    cen = 0,
                    cumul = TRUE)

    lags <- 0:lag_max
    rr   <- rep(NA_real_, length(lags))
    low  <- rep(NA_real_, length(lags))
    high <- rep(NA_real_, length(lags))

    if (!is.null(cp$matRRfit)) {
      for (i in seq_along(lags)) {
        j <- i
        rr[i]   <- cp$matRRfit[2, j]
        low[i]  <- cp$matRRlow[2, j]
        high[i] <- cp$matRRhigh[2, j]
      }
    }

    cum_rr  <- NA_real_
    cum_low <- NA_real_
    cum_high<- NA_real_
    if (!is.null(cp$cumRRfit)) {
      jcum <- lag_max + 1
      cum_rr   <- cp$cumRRfit[2, jcum]
      cum_low  <- cp$cumRRlow[2, jcum]
      cum_high <- cp$cumRRhigh[2, jcum]
    }

    out <- data.frame(lag = lags, rr = rr, low = low, high = high)
    out$cum_rr   <- cum_rr
    out$cum_low  <- cum_low
    out$cum_high <- cum_high

    out

  }, error = function(e) {
    lags <- 0:lag_max
    message("⚠️ rr_lag_events_threshold: error for outcome = ", outcome,
            " — returning RR(lag) = NA. Message: ", conditionMessage(e))
    data.frame(lag = lags,
               rr = NA_real_, low = NA_real_, high = NA_real_,
               cum_rr = NA_real_, cum_low = NA_real_, cum_high = NA_real_)
  })

  return(res)
}
"""
ro.r(R_CODE)
r_rr_event_fun = ro.globalenv["rr_lag_events_threshold"]

# ============================================================
# R: (B) CONTINUOUS PM DLNM (RR vs PM by lag)
# ============================================================

R_CODE_PM = r"""
suppressPackageStartupMessages({
  library(dlnm)
  library(splines)
  library(stats)
})

rr_x_pm_by_lag <- function(df, outcome, pm_col, lag_max,
                           df_time = 7,
                           pm_grid = seq(0, 100, by = 5),
                           pm_cen = 5) {

  required <- c(outcome, pm_col,
                "TIME_INDEX", "DOW",
                "TEMP_MA21", "RH_MA7", "WIND_MA7")

  if (!all(required %in% names(df))) {
    miss <- setdiff(required, names(df))
    stop(paste("Missing required columns:", paste(miss, collapse = ", ")))
  }

  df2 <- df[complete.cases(df[, required]), , drop = FALSE]
  if (nrow(df2) < 150) {
    return(data.frame())
  }

  if (all(is.na(df2[[pm_col]])) || sd(df2[[pm_col]], na.rm=TRUE) == 0) {
    return(data.frame())
  }

  res <- tryCatch({

    qs <- quantile(df2[[pm_col]], probs=c(0.10, 0.50, 0.90), na.rm=TRUE, names=FALSE)
    qs <- sort(unique(qs))
    if (length(qs) < 2) qs <- quantile(df2[[pm_col]], probs=c(0.25, 0.75), na.rm=TRUE, names=FALSE)

    cb_pm <- crossbasis(df2[[pm_col]],
                        lag = lag_max,
                        argvar = list(fun = "ns", knots = qs),
                        arglag = list(fun = "ns", df = 4))

    fit <- glm(df2[[outcome]] ~ cb_pm +
                 ns(df2[["TIME_INDEX"]], df_time) +
                 factor(df2[["DOW"]]) +
                 ns(df2[["TEMP_MA21"]], 4) +
                 ns(df2[["RH_MA7"]], 4) +
                 ns(df2[["WIND_MA7"]], 3),
               family = quasipoisson(),
               data = df2)

    cp <- crosspred(cb_pm, fit,
                    at  = pm_grid,
                    cen = pm_cen,
                    cumul = TRUE)

    lags <- 0:lag_max
    out <- data.frame()

    if (!is.null(cp$matRRfit)) {
      for (j in seq_along(lags)) {
        lagj <- lags[j]
        tmp <- data.frame(
          pm = pm_grid,
          lag = lagj,
          rr = cp$matRRfit[, j],
          low = cp$matRRlow[, j],
          high = cp$matRRhigh[, j]
        )
        out <- rbind(out, tmp)
      }
    }

    out

  }, error = function(e) {
    message("⚠️ rr_x_pm_by_lag error for outcome = ", outcome,
            " — returning empty. Message: ", conditionMessage(e))
    data.frame()
  })

  return(res)
}
"""
ro.r(R_CODE_PM)
r_rr_pm_fun = ro.globalenv["rr_x_pm_by_lag"]

def py_to_r_df(df: pd.DataFrame):
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)

# ============================================================
# PLOTS
# ============================================================

def plot_rr_lag_curve(rr: pd.DataFrame, out_base: str):
    if rr is None or rr.empty:
        return
    rr = rr.sort_values("lag").copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(rr["lag"], rr["rr"], lw=2.4, color=LINE_BLUE)
    ax.fill_between(rr["lag"], rr["low"], rr["high"], color=SHADE_BLUEGRAY, alpha=SHADE_ALPHA)
    ax.axhline(1.0, lw=1, color="black", alpha=0.4)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Relative risk")
    ax.set_title("")
    ax.set_xlim(0, int(rr["lag"].max()))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    save_fig_all(fig, out_base)

def plot_rr_vs_pm(rr_pm: pd.DataFrame, out_base: str, lag_max: int = LAG_MAX):
    """Existing: plots one curve per lag (0..lag_max), no ribbons per lag."""
    if rr_pm is None or rr_pm.empty:
        return

    rr_pm = rr_pm.copy()
    rr_pm["pm"] = pd.to_numeric(rr_pm["pm"], errors="coerce")
    rr_pm["lag"] = pd.to_numeric(rr_pm["lag"], errors="coerce").astype(int)
    rr_pm["rr"] = pd.to_numeric(rr_pm["rr"], errors="coerce")
    rr_pm["low"] = pd.to_numeric(rr_pm["low"], errors="coerce")
    rr_pm["high"] = pd.to_numeric(rr_pm["high"], errors="coerce")
    rr_pm = rr_pm.dropna(subset=["pm", "lag", "rr", "low", "high"])

    rr_pm = rr_pm[
        (rr_pm["pm"] >= 0) & (rr_pm["pm"] <= 100) & (rr_pm["lag"].between(0, lag_max))
    ].copy()
    if rr_pm.empty:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for lag in range(0, lag_max + 1):
        sub = rr_pm[rr_pm["lag"] == lag].sort_values("pm")
        if sub.empty:
            continue
        ax.plot(sub["pm"], sub["rr"], lw=2.0, label=f"Lag {lag}")

    ax.axhline(1.0, lw=1.0, color="black", alpha=0.4)
    ax.set_xlabel("Wildfire-related PM2.5 (µg/m³)")
    ax.set_ylabel("Relative risk")
    ax.set_title("")
    ax.set_xlim(0, 100)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(ncols=2, fontsize=8, frameon=False, loc="upper left")
    save_fig_all(fig, out_base)

def plot_rr_vs_pm_lag0_with_ci(rr_pm: pd.DataFrame, out_base: str):
    """
    Concentration–response for LAG 0 ONLY with uncertainty (95% CI ribbon).
    """
    if rr_pm is None or rr_pm.empty:
        return

    d = rr_pm.copy()
    d["pm"] = pd.to_numeric(d["pm"], errors="coerce")
    d["lag"] = pd.to_numeric(d["lag"], errors="coerce").astype(int)
    d["rr"] = pd.to_numeric(d["rr"], errors="coerce")
    d["low"] = pd.to_numeric(d["low"], errors="coerce")
    d["high"] = pd.to_numeric(d["high"], errors="coerce")
    d = d.dropna(subset=["pm", "lag", "rr", "low", "high"])
    d = d[(d["pm"].between(0, 100)) & (d["lag"] == 0)].sort_values("pm")
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(d["pm"], d["rr"], lw=2.6, color=LINE_BLUE)
    ax.fill_between(d["pm"], d["low"], d["high"], color=SHADE_BLUEGRAY, alpha=SHADE_ALPHA)
    ax.axhline(1.0, lw=1.0, color="black", alpha=0.4)

    ax.set_xlabel("Wildfire-related PM2.5 (µg/m³)")
    ax.set_ylabel("Relative risk")
    ax.set_title("")
    ax.set_xlim(0, 100)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    save_fig_all(fig, out_base)

# ============================================================
# META-ANALYSIS (DerSimonian-Laird) IN LOG-RR
# ============================================================

def meta_random_effects(log_rr: np.ndarray, se: np.ndarray):
    w_fixed = 1.0 / (se ** 2)
    mu_fixed = np.sum(w_fixed * log_rr) / np.sum(w_fixed)
    q = np.sum(w_fixed * (log_rr - mu_fixed) ** 2)
    df = len(log_rr) - 1
    if df <= 0:
        tau2 = 0.0
    else:
        c = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
        tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    w_re = 1.0 / (se ** 2 + tau2)
    mu_re = np.sum(w_re * log_rr) / np.sum(w_re)
    se_re = math.sqrt(1.0 / np.sum(w_re))
    return mu_re, se_re, tau2, q, df

def add_overall_meta_rr(rr_states: pd.DataFrame) -> pd.DataFrame:
    """OVERALL for RR(lag) (event-based), lag 0..LAG_MAX, for ALL outcomes present."""
    if rr_states is None or rr_states.empty:
        return rr_states

    rr_states = rr_states.copy()
    rr_states["lag"] = pd.to_numeric(rr_states["lag"], errors="coerce").astype(int)
    rr_states["rr"] = pd.to_numeric(rr_states["rr"], errors="coerce")
    rr_states["low"] = pd.to_numeric(rr_states["low"], errors="coerce")
    rr_states["high"] = pd.to_numeric(rr_states["high"], errors="coerce")

    rr_states = rr_states.dropna(subset=["rr", "low", "high"])
    rr_states = rr_states[rr_states["rr"] > 0].copy()

    meta_rows = []
    for (dataset, outcome, lag), sub in rr_states.groupby(["dataset", "outcome", "lag"]):
        if sub.shape[0] < 2:
            continue

        log_rr = np.log(sub["rr"].to_numpy(float))
        log_low = np.log(sub["low"].to_numpy(float))
        log_high = np.log(sub["high"].to_numpy(float))
        se = (log_high - log_low) / (2 * 1.96)

        mask = np.isfinite(log_rr) & np.isfinite(se) & (se > 0)
        if mask.sum() < 2:
            continue

        mu_re, se_re, tau2, q, df = meta_random_effects(log_rr[mask], se[mask])
        rr_pool = math.exp(mu_re)
        low_pool = math.exp(mu_re - 1.96 * se_re)
        high_pool = math.exp(mu_re + 1.96 * se_re)

        meta_rows.append({
            "state": "OVERALL",
            "dataset": dataset,
            "outcome": outcome,
            "lag": int(lag),
            "rr": rr_pool,
            "low": low_pool,
            "high": high_pool,
            "meta_tau2": tau2,
            "meta_Q": q,
            "meta_df": df,
            "meta_method": "DL",
        })

    if not meta_rows:
        return rr_states

    meta_df = pd.DataFrame(meta_rows)

    common_cols = sorted(set(rr_states.columns) | set(meta_df.columns))
    for c in common_cols:
        if c not in rr_states.columns:
            rr_states[c] = np.nan
        if c not in meta_df.columns:
            meta_df[c] = np.nan

    out = pd.concat([rr_states[common_cols], meta_df[common_cols]], ignore_index=True)
    return out

def meta_overall_rr_pm(rr_pm_states: pd.DataFrame) -> pd.DataFrame:
    """
    OVERALL for RR(PM) by lag and pm-grid.
    For each (dataset, outcome, lag, pm): random-effects DL meta in log(RR).
    """
    if rr_pm_states is None or rr_pm_states.empty:
        return pd.DataFrame()

    df = rr_pm_states.copy()
    df = df[df["state"] != "OVERALL"].copy()

    df["pm"] = pd.to_numeric(df["pm"], errors="coerce")
    df["lag"] = pd.to_numeric(df["lag"], errors="coerce").astype(int)
    df["rr"] = pd.to_numeric(df["rr"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")

    df = df.dropna(subset=["pm", "lag", "rr", "low", "high"])
    df = df[(df["rr"] > 0) & (df["low"] > 0) & (df["high"] > 0)]

    rows = []
    for (dataset, outcome, lag, pm), sub in df.groupby(["dataset", "outcome", "lag", "pm"]):
        if sub.shape[0] < 2:
            continue

        log_rr = np.log(sub["rr"].to_numpy(float))
        log_low = np.log(sub["low"].to_numpy(float))
        log_high = np.log(sub["high"].to_numpy(float))
        se = (log_high - log_low) / (2 * 1.96)

        mask = np.isfinite(log_rr) & np.isfinite(se) & (se > 0)
        if mask.sum() < 2:
            continue

        mu_re, se_re, tau2, q, dfm = meta_random_effects(log_rr[mask], se[mask])
        rr_pool = math.exp(mu_re)
        low_pool = math.exp(mu_re - 1.96 * se_re)
        high_pool = math.exp(mu_re + 1.96 * se_re)

        rows.append({
            "state": "OVERALL",
            "dataset": dataset,
            "outcome": outcome,
            "lag": int(lag),
            "pm": float(pm),
            "rr": rr_pool,
            "low": low_pool,
            "high": high_pool,
            "meta_tau2": tau2,
            "meta_Q": q,
            "meta_df": dfm,
            "meta_method": "DL",
        })

    return pd.DataFrame(rows)

# ============================================================
# DLNM WRAPPERS (A) EVENT and (B) PM
# ============================================================

def run_dlnm_events_for_state(
    df: pd.DataFrame,
    dataset_type: str,
    state_label: str,
    outcomes: List[str],
    lag_max: int = LAG_MAX,
    df_time: int = TIME_DF,
    scenario: str = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    results = []
    for outc in outcomes:
        if outc not in df.columns:
            continue

        cols_needed = [outc, "event_extreme", "TIME_INDEX", "DOW", TEMP_MA_COL, RH_MA_COL, WIND_MA_COL]
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            log(f"[{scenario}][{state_label}][{dataset_type}] missing cols for outcome {outc}: {missing}")
            continue

        d = df[cols_needed].copy()
        for c in cols_needed:
            if c == "DOW":
                d[c] = pd.to_numeric(d[c], errors="coerce").astype("Int64")
            else:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        d = d.dropna()
        if d.shape[0] < 100:
            continue

        if (d["event_extreme"] > 0).sum() < 3:
            continue

        try:
            r_df = py_to_r_df(d)
            res = r_rr_event_fun(r_df, outc, int(lag_max), int(df_time))
            with localconverter(ro.default_converter + pandas2ri.converter):
                rr_lag = ro.conversion.rpy2py(res)
        except Exception as e:
            log(f"⚠️ R run failed for {state_label}/{dataset_type}/{outc}: {e}")
            continue

        rr_lag["state"] = state_label
        rr_lag["dataset"] = dataset_type
        rr_lag["outcome"] = outc
        rr_lag["outcome_group"] = classify_outcome(outc)
        if scenario is not None:
            rr_lag["scenario"] = scenario
        results.append(rr_lag)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def run_dlnm_pm_by_lag_for_state(
    df: pd.DataFrame,
    dataset_type: str,
    state_label: str,
    outcomes: List[str],
    lag_max: int = LAG_MAX,
    df_time: int = TIME_DF,
    scenario: str = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    results = []
    for outc in outcomes:
        if outc not in df.columns:
            continue

        cols_needed = [outc, PM_CANON, "TIME_INDEX", "DOW", TEMP_MA_COL, RH_MA_COL, WIND_MA_COL]
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            continue

        d = df[cols_needed].copy()
        for c in cols_needed:
            if c == "DOW":
                d[c] = pd.to_numeric(d[c], errors="coerce").astype("Int64")
            else:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna()

        if d.shape[0] < 150:
            continue

        try:
            r_df = py_to_r_df(d)
            res = r_rr_pm_fun(
                r_df,
                outc,
                PM_CANON,
                int(lag_max),
                int(df_time),
                ro.FloatVector(PM_GRID.tolist()),
                float(PM_CEN),
            )
            with localconverter(ro.default_converter + pandas2ri.converter):
                rr_pm = ro.conversion.rpy2py(res)
        except Exception as e:
            log(f"⚠️ R run failed RR~PM for {state_label}/{dataset_type}/{outc}: {e}")
            continue

        if rr_pm is None or len(rr_pm) == 0:
            continue

        rr_pm["state"] = state_label
        rr_pm["dataset"] = dataset_type
        rr_pm["outcome"] = outc
        rr_pm["outcome_group"] = classify_outcome(outc)
        if scenario is not None:
            rr_pm["scenario"] = scenario
        results.append(rr_pm)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

# ============================================================
# SUMMARY (for forest plots) uses EVENT-based cumulative RR
# ============================================================

def build_state_summary(
    rr_states: pd.DataFrame,
    ts_hosp_state: Dict[str, pd.DataFrame],
    ts_mort_state: Dict[str, pd.DataFrame],
    states: List[str],
    core_outcomes: Dict[str, List[str]],
) -> pd.DataFrame:
    rows = []

    for dataset_type, ts_map in [("HOSP", ts_hosp_state), ("MORT", ts_mort_state)]:
        outcomes = core_outcomes.get(dataset_type, [])
        for st in states:
            if st not in ts_map or ts_map[st] is None or ts_map[st].empty:
                continue
            ts = ts_map[st]
            n_events = int((ts["event_extreme"] > 0).sum())

            for outc in outcomes:
                if outc not in ts.columns:
                    continue

                rr_sub = rr_states[
                    (rr_states["state"] == st) &
                    (rr_states["dataset"] == dataset_type) &
                    (rr_states["outcome"] == outc)
                ].copy()
                if rr_sub.empty:
                    continue

                cols_req = [outc, "event_extreme", "TIME_INDEX", "DOW", TEMP_MA_COL, RH_MA_COL, WIND_MA_COL]
                cols_req = [c for c in cols_req if c in ts.columns]
                mask = ts[cols_req].notna().all(axis=1)
                n_valid = int(mask.sum())

                rr_vals = pd.to_numeric(rr_sub["rr"], errors="coerce")
                if rr_vals.dropna().empty:
                    peak_rr = np.nan
                    lag_peak = np.nan
                else:
                    idxmax = rr_vals.idxmax()
                    peak_rr = float(rr_vals.loc[idxmax])
                    lag_peak = int(rr_sub.loc[idxmax, "lag"])

                cum_rr = cum_low = cum_high = np.nan
                if "cum_rr" in rr_sub.columns:
                    cr = pd.to_numeric(rr_sub["cum_rr"], errors="coerce").dropna()
                    cl = pd.to_numeric(rr_sub.get("cum_low", np.nan), errors="coerce").dropna()
                    ch = pd.to_numeric(rr_sub.get("cum_high", np.nan), errors="coerce").dropna()
                    if cr.size > 0:
                        cum_rr = float(cr.iloc[0])
                    if cl.size > 0:
                        cum_low = float(cl.iloc[0])
                    if ch.size > 0:
                        cum_high = float(ch.iloc[0])

                rows.append({
                    "state": st,
                    "dataset": dataset_type,
                    "outcome": outc,
                    "outcome_group": classify_outcome(outc),
                    "n_events": n_events,
                    "n_valid_days": n_valid,
                    "peak_rr": peak_rr,
                    "lag_peak": lag_peak,
                    "cum_rr": cum_rr,
                    "cum_low": cum_low,
                    "cum_high": cum_high,
                })

    return pd.DataFrame(rows)

def build_overall_summary_from_states(
    df_state_summary: pd.DataFrame,
    rr_all: pd.DataFrame,
    core_outcomes: Dict[str, List[str]],
) -> pd.DataFrame:
    rows = []

    for dataset in ["HOSP", "MORT"]:
        outcomes = core_outcomes.get(dataset, [])
        for outc in outcomes:
            sub_state = df_state_summary[
                (df_state_summary["dataset"] == dataset) &
                (df_state_summary["outcome"] == outc)
            ].copy()
            if sub_state.empty:
                continue

            n_events_total = int(sub_state["n_events"].sum())
            n_valid_total = int(sub_state["n_valid_days"].sum())

            sub_c = sub_state.dropna(subset=["cum_rr", "cum_low", "cum_high"])
            if sub_c.empty:
                cum_rr_pool = cum_low_pool = cum_high_pool = np.nan
            else:
                log_rr = np.log(sub_c["cum_rr"].to_numpy(float))
                log_low = np.log(sub_c["cum_low"].to_numpy(float))
                log_high = np.log(sub_c["cum_high"].to_numpy(float))
                se = (log_high - log_low) / (2 * 1.96)
                mask = np.isfinite(log_rr) & np.isfinite(se) & (se > 0)
                if mask.sum() >= 2:
                    mu_re, se_re, tau2, q, df_meta = meta_random_effects(log_rr[mask], se[mask])
                    cum_rr_pool = math.exp(mu_re)
                    cum_low_pool = math.exp(mu_re - 1.96 * se_re)
                    cum_high_pool = math.exp(mu_re + 1.96 * se_re)
                else:
                    cum_rr_pool = cum_low_pool = cum_high_pool = np.nan

            rr_over = rr_all[
                (rr_all["state"] == "OVERALL") &
                (rr_all["dataset"] == dataset) &
                (rr_all["outcome"] == outc)
            ].copy()
            if rr_over.empty:
                peak_rr_over = np.nan
                lag_peak_over = np.nan
            else:
                vals = pd.to_numeric(rr_over["rr"], errors="coerce")
                if vals.dropna().empty:
                    peak_rr_over = np.nan
                    lag_peak_over = np.nan
                else:
                    idxmax = vals.idxmax()
                    peak_rr_over = float(vals.loc[idxmax])
                    lag_peak_over = int(rr_over.loc[idxmax, "lag"])

            rows.append({
                "state": "OVERALL",
                "dataset": dataset,
                "outcome": outc,
                "outcome_group": classify_outcome(outc),
                "n_events": n_events_total,
                "n_valid_days": n_valid_total,
                "peak_rr": peak_rr_over,
                "lag_peak": lag_peak_over,
                "cum_rr": cum_rr_pool,
                "cum_low": cum_low_pool,
                "cum_high": cum_high_pool,
            })

    return pd.DataFrame(rows)

# ============================================================
# FOREST PLOTS (CUMULATIVE RR 0–LAG_MAX) + weights + p-values
# ============================================================

def build_forest_data(df_summary_all: pd.DataFrame, dataset: str, outcome: str) -> pd.DataFrame:
    sub_state = df_summary_all[
        (df_summary_all["dataset"] == dataset) &
        (df_summary_all["outcome"] == outcome) &
        (df_summary_all["state"] != "OVERALL")
    ].copy()

    sub_state = sub_state.dropna(subset=["cum_rr", "cum_low", "cum_high"])
    if sub_state.empty:
        return pd.DataFrame()

    log_rr = np.log(sub_state["cum_rr"].to_numpy(float))
    log_low = np.log(sub_state["cum_low"].to_numpy(float))
    log_high = np.log(sub_state["cum_high"].to_numpy(float))
    se = (log_high - log_low) / (2 * 1.96)

    mask = np.isfinite(log_rr) & np.isfinite(se) & (se > 0)
    sub_state = sub_state.loc[mask].copy()
    if sub_state.empty:
        return pd.DataFrame()

    log_rr = log_rr[mask]
    se = se[mask]

    # Random-effects (DL)
    mu_re, se_re, tau2, q, df = meta_random_effects(log_rr, se)
    rr_pool = math.exp(mu_re)
    low_pool = math.exp(mu_re - 1.96 * se_re)
    high_pool = math.exp(mu_re + 1.96 * se_re)

    # RE weights and normalized weights
    weight = 1.0 / (se ** 2 + tau2)
    weight_norm = weight / np.sum(weight) if np.sum(weight) > 0 else np.full_like(weight, np.nan)

    # p_value per state: beta vs 0
    z_state = log_rr / se
    try:
        p_value = (np.array(ro.r["pnorm"](np.abs(z_state), lower_tail=False)) * 2.0).astype(float)
    except Exception:
        from math import erfc, sqrt
        p_value = np.array([erfc(abs(z) / sqrt(2.0)) for z in z_state], dtype=float)

    # p_value_difference: beta vs pooled (approx independence)
    denom = np.sqrt(se ** 2 + se_re ** 2) if (se_re is not None and se_re > 0) else np.full_like(se, np.nan)
    z_diff = (log_rr - mu_re) / denom
    try:
        p_value_difference = (np.array(ro.r["pnorm"](np.abs(z_diff), lower_tail=False)) * 2.0).astype(float)
    except Exception:
        from math import erfc, sqrt
        p_value_difference = np.array(
            [erfc(abs(z) / sqrt(2.0)) if np.isfinite(z) else np.nan for z in z_diff], dtype=float
        )

    # heterogeneity
    if df > 0 and q > 0:
        meta_I2 = max(0.0, (q - df) / q) * 100.0
        try:
            p_heterogeneity = float(ro.r["pchisq"](q, df, lower_tail=False)[0])
        except Exception:
            p_heterogeneity = np.nan
    else:
        meta_I2 = np.nan
        p_heterogeneity = np.nan

    # pooled p-value
    if se_re and se_re > 0:
        z_pool = mu_re / se_re
        try:
            p_overall = float(ro.r["pnorm"](abs(z_pool), lower_tail=False)[0]) * 2.0
        except Exception:
            from math import erfc, sqrt
            p_overall = erfc(abs(z_pool) / sqrt(2.0))
    else:
        p_overall = np.nan

    forest = pd.DataFrame({
        "state": sub_state["state"].values,
        "dataset": dataset,
        "outcome": outcome,
        "outcome_group": sub_state.get("outcome_group", "UNKNOWN").values,
        "RR": sub_state["cum_rr"].values.astype(float),
        "CIlow": sub_state["cum_low"].values.astype(float),
        "CIhigh": sub_state["cum_high"].values.astype(float),
        "logRR": log_rr,
        "SE": se,
        "weight": weight,
        "weight_norm": weight_norm,
        "p_value": p_value,
        "p_value_difference": p_value_difference,
        "isOverall": False,
        "meta_tau2": tau2,
        "meta_Q": q,
        "meta_df": df,
        "meta_I2": meta_I2,
        "p_overall": p_overall,
        "p_heterogeneity": p_heterogeneity,
    })

    overall_row = {
        "state": "OVERALL",
        "dataset": dataset,
        "outcome": outcome,
        "outcome_group": classify_outcome(outcome),
        "RR": rr_pool,
        "CIlow": low_pool,
        "CIhigh": high_pool,
        "logRR": mu_re,
        "SE": se_re,
        "weight": 1.0 / (se_re ** 2) if se_re and se_re > 0 else np.nan,
        "weight_norm": np.nan,
        "p_value": p_overall,
        "p_value_difference": np.nan,
        "isOverall": True,
        "meta_tau2": tau2,
        "meta_Q": q,
        "meta_df": df,
        "meta_I2": meta_I2,
        "p_overall": p_overall,
        "p_heterogeneity": p_heterogeneity,
    }
    forest = pd.concat([forest, pd.DataFrame([overall_row])], ignore_index=True)
    forest = forest.sort_values(by=["isOverall", "state"], ascending=[True, True]).reset_index(drop=True)
    return forest

def plot_forest(forest: pd.DataFrame, dataset: str, outcome: str, scenario: str, out_dir_rr: str):
    if forest is None or forest.empty:
        return

    forest = forest.copy()
    forest = forest[
        (forest["RR"] > 0) & (forest["CIlow"] > 0) & (forest["CIhigh"] > 0)
    ]
    if forest.empty:
        return

    n = forest.shape[0]
    y_pos = np.arange(n)

    mask_overall = forest["isOverall"]
    mask_state = ~forest["isOverall"]

    fig, ax = plt.subplots(figsize=(6.8, 0.45 * n + 1.5))

    ax.errorbar(
        forest.loc[mask_state, "RR"],
        y_pos[mask_state],
        xerr=[
            forest.loc[mask_state, "RR"] - forest.loc[mask_state, "CIlow"],
            forest.loc[mask_state, "CIhigh"] - forest.loc[mask_state, "RR"],
        ],
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.0,
        capsize=2.5,
        markersize=4.0,
        alpha=0.9,
    )

    if mask_overall.any():
        ax.errorbar(
            forest.loc[mask_overall, "RR"],
            y_pos[mask_overall],
            xerr=[
                forest.loc[mask_overall, "RR"] - forest.loc[mask_overall, "CIlow"],
                forest.loc[mask_overall, "CIhigh"] - forest.loc[mask_overall, "RR"],
            ],
            fmt="s",
            color=LINE_BLUE,
            ecolor=LINE_BLUE,
            elinewidth=1.4,
            capsize=3.0,
            markersize=7.0,
            alpha=0.95,
        )

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(forest["state"].tolist())

    ax.set_xlabel(f"Cumulative relative risk (lag 0–{LAG_MAX}, log scale)")
    ax.set_title("")

    x_max = max(forest["CIhigh"].max() * 1.3, 1.2)
    ax.set_xlim(left=max(forest["CIlow"].min() / 1.5, 0.2), right=x_max)

    for i, row in forest.iterrows():
        rr_val = row["RR"]
        ci_l = row["CIlow"]
        ci_h = row["CIhigh"]
        text = f"{rr_val:.2f} ({ci_l:.2f}, {ci_h:.2f})"
        ax.text(x_max * 1.02, y_pos[i], text, va="center", fontsize=7.2)

    plt.tight_layout()
    out_base = os.path.join(
        out_dir_rr,
        sanitize_filename(f"FOREST_{scenario}_{dataset}_{outcome}")
    )
    save_fig_all(fig, out_base)

def make_forest_plots(
    df_summary_all: pd.DataFrame,
    scenario: str,
    out_dir_rr: str,
    out_dir_tables: str,
):
    """
    Forest plots for TOTAL + SEX + AGE + OTHER outcomes (OVERALL vs States).
    """
    if df_summary_all is None or df_summary_all.empty:
        return

    for dataset in ["HOSP", "MORT"]:
        outs = sorted(
            df_summary_all[df_summary_all["dataset"] == dataset]["outcome"]
            .dropna()
            .unique()
            .tolist()
        )
        for outc in outs:
            forest = build_forest_data(df_summary_all, dataset, outc)
            if forest is None or forest.empty:
                continue

            csv_path = os.path.join(
                out_dir_tables,
                sanitize_filename(f"FOREST_{scenario}_{dataset}_{outc}.csv"),
            )
            save_csv_nature(forest, csv_path)
            plot_forest(forest, dataset, outc, scenario, out_dir_rr)

# ============================================================
# NEW: FORESTS OVERALL vs AGE and OVERALL vs SEX
# ============================================================

def extract_family_prefix(outcome: str) -> str:
    """
    For TOTAL outcomes like HOSP_RESP_TOTAL, returns 'HOSP_RESP'.
    Fallback: everything except last token.
    """
    c = str(outcome).upper()
    if c.endswith("_TOTAL"):
        return c[:-len("_TOTAL")]
    parts = c.split("_")
    if len(parts) > 1:
        return "_".join(parts[:-1])
    return c

def make_strata_label(outcome: str, family_prefix: str, group: str) -> str:
    """
    Human-readable label for age/sex strata (used on forest y-axis).
    """
    c = str(outcome).upper()
    suf = c[len(family_prefix):].lstrip("_")

    if group == "SEX":
        if "FEM" in suf or "FEMALE" in suf:
            return "Female"
        if "MASC" in suf or "MALE" in suf:
            return "Male"
        return suf.title() if suf else "Sex"

    if group == "AGE":
        s = suf
        m = re.match(r"(\d{1,2})[_\-](\d{1,2})", s)
        if m:
            return f"Age {int(m.group(1))}–{int(m.group(2))}"
        m2 = re.match(r"(\d{1,2})(\+|PLUS)", s)
        if m2:
            return f"Age {int(m2.group(1))}+"
        m3 = re.match(r"(\d{1,2})$", s)
        if m3:
            return f"Age {int(m3.group(1))}"
        return f"Age {s.title()}" if s else "Age"
    return suf.title() if suf else str(outcome)

def build_forest_data_strata(
    df_summary_all: pd.DataFrame,
    dataset: str,
    total_outcome: str,
    strata_group: str,
) -> pd.DataFrame:
    """
    Build forest data comparing OVERALL TOTAL vs OVERALL age or sex strata.

    - Uses only rows with state == 'OVERALL' (meta across states).
    - total_outcome is e.g. 'HOSP_RESP_TOTAL'.
    - strata_group in {'AGE', 'SEX'}.
    """
    if df_summary_all is None or df_summary_all.empty:
        return pd.DataFrame()

    df = df_summary_all[
        (df_summary_all["dataset"] == dataset) &
        (df_summary_all["state"] == "OVERALL")
    ].copy()
    if df.empty:
        return pd.DataFrame()

    tot_row = df[df["outcome"] == total_outcome].copy()
    if tot_row.empty:
        return pd.DataFrame()
    tot_row = tot_row.iloc[0]

    # Extract family prefix and select strata
    family_prefix = extract_family_prefix(total_outcome)
    mask_strata = (
        (df["outcome_group"] == strata_group) &
        df["outcome"].str.upper().str.startswith(family_prefix + "_")
    )
    sub = df[mask_strata].copy()
    sub = sub.dropna(subset=["cum_rr", "cum_low", "cum_high"])
    if sub.empty:
        return pd.DataFrame()

    # Prepare TOTAL effect
    tot_rr = float(tot_row["cum_rr"])
    tot_low = float(tot_row["cum_low"])
    tot_high = float(tot_row["cum_high"])
    if not (np.isfinite(tot_rr) and np.isfinite(tot_low) and np.isfinite(tot_high)):
        return pd.DataFrame()
    if tot_rr <= 0 or tot_low <= 0 or tot_high <= 0:
        return pd.DataFrame()

    log_rr_overall = math.log(tot_rr)
    se_overall = (math.log(tot_high) - math.log(tot_low)) / (2 * 1.96)
    if not np.isfinite(se_overall) or se_overall <= 0:
        return pd.DataFrame()

    # Prepare strata effects
    rr = sub["cum_rr"].to_numpy(float)
    low = sub["cum_low"].to_numpy(float)
    high = sub["cum_high"].to_numpy(float)

    mask_pos = (rr > 0) & (low > 0) & (high > 0)
    rr = rr[mask_pos]
    low = low[mask_pos]
    high = high[mask_pos]
    sub = sub.loc[mask_pos].copy()
    if sub.empty:
        return pd.DataFrame()

    log_rr = np.log(rr)
    se = (np.log(high) - np.log(low)) / (2 * 1.96)
    mask_fin = np.isfinite(log_rr) & np.isfinite(se) & (se > 0)
    log_rr = log_rr[mask_fin]
    se = se[mask_fin]
    sub = sub.loc[mask_fin].copy()
    if sub.empty:
        return pd.DataFrame()

    # Weights
    weight_strata = 1.0 / (se ** 2)
    weight_overall = 1.0 / (se_overall ** 2)
    weights_all = np.concatenate([weight_strata, np.array([weight_overall])])
    wmask = np.isfinite(weights_all) & (weights_all > 0)
    wsum = np.sum(weights_all[wmask]) if np.any(wmask) else np.nan
    weight_norm_all = weights_all / wsum if np.isfinite(wsum) and wsum > 0 else np.full_like(weights_all, np.nan)

    # p-values for strata vs null
    z_strata = log_rr / se
    try:
        p_strata = (np.array(ro.r["pnorm"](np.abs(z_strata), lower_tail=False)) * 2.0).astype(float)
    except Exception:
        from math import erfc, sqrt
        p_strata = np.array([erfc(abs(z) / sqrt(2.0)) for z in z_strata], dtype=float)

    # TOTAL p-value vs null
    try:
        z_overall = log_rr_overall / se_overall
        p_overall = float(ro.r["pnorm"](abs(z_overall), lower_tail=False)[0]) * 2.0
    except Exception:
        from math import erfc, sqrt
        p_overall = erfc(abs(log_rr_overall / se_overall) / math.sqrt(2.0))

    # p-value for difference (strata vs TOTAL)
    denom = np.sqrt(se ** 2 + se_overall ** 2)
    z_diff = (log_rr - log_rr_overall) / denom
    try:
        p_diff = (np.array(ro.r["pnorm"](np.abs(z_diff), lower_tail=False)) * 2.0).astype(float)
    except Exception:
        from math import erfc, sqrt
        p_diff = np.array(
            [erfc(abs(z) / sqrt(2.0)) if np.isfinite(z) else np.nan for z in z_diff],
            dtype=float,
        )

    # Heterogeneity across strata (optional)
    if len(log_rr) >= 2:
        mu_re, se_re, tau2, q, df_meta = meta_random_effects(log_rr, se)
        if df_meta > 0 and q > 0:
            meta_I2 = max(0.0, (q - df_meta) / q) * 100.0
            try:
                p_heterogeneity = float(ro.r["pchisq"](q, df_meta, lower_tail=False)[0])
            except Exception:
                p_heterogeneity = np.nan
        else:
            tau2 = np.nan
            q = np.nan
            df_meta = np.nan
            meta_I2 = np.nan
            p_heterogeneity = np.nan
    else:
        tau2 = np.nan
        q = np.nan
        df_meta = np.nan
        meta_I2 = np.nan
        p_heterogeneity = np.nan

    # Build rows (strata + TOTAL)
    rows = []
    labels = [make_strata_label(outc, family_prefix, strata_group) for outc in sub["outcome"].tolist()]

    for i, (_, row_sub) in enumerate(sub.iterrows()):
        rows.append({
            "state": labels[i],
            "dataset": dataset,
            "outcome": row_sub["outcome"],
            "outcome_group": row_sub["outcome_group"],
            "RR": float(rr[i]),
            "CIlow": float(low[i]),
            "CIhigh": float(high[i]),
            "logRR": float(log_rr[i]),
            "SE": float(se[i]),
            "weight": float(weight_strata[i]),
            "weight_norm": float(weight_norm_all[i]),
            "p_value": float(p_strata[i]),
            "p_value_difference": float(p_diff[i]),
            "isOverall": False,
            "meta_tau2": float(tau2) if np.isfinite(tau2) else np.nan,
            "meta_Q": float(q) if np.isfinite(q) else np.nan,
            "meta_df": float(df_meta) if np.isfinite(df_meta) else np.nan,
            "meta_I2": float(meta_I2) if np.isfinite(meta_I2) else np.nan,
            "p_overall": float(p_overall) if np.isfinite(p_overall) else np.nan,
            "p_heterogeneity": float(p_heterogeneity) if np.isfinite(p_heterogeneity) else np.nan,
        })

    rows.append({
        "state": "OVERALL",
        "dataset": dataset,
        "outcome": total_outcome,
        "outcome_group": tot_row["outcome_group"],
        "RR": tot_rr,
        "CIlow": tot_low,
        "CIhigh": tot_high,
        "logRR": log_rr_overall,
        "SE": se_overall,
        "weight": weight_overall,
        "weight_norm": float(weight_norm_all[-1]),
        "p_value": float(p_overall),
        "p_value_difference": np.nan,
        "isOverall": True,
        "meta_tau2": float(tau2) if np.isfinite(tau2) else np.nan,
        "meta_Q": float(q) if np.isfinite(q) else np.nan,
        "meta_df": float(df_meta) if np.isfinite(df_meta) else np.nan,
        "meta_I2": float(meta_I2) if np.isfinite(meta_I2) else np.nan,
        "p_overall": float(p_overall) if np.isfinite(p_overall) else np.nan,
        "p_heterogeneity": float(p_heterogeneity) if np.isfinite(p_heterogeneity) else np.nan,
    })

    forest = pd.DataFrame(rows)
    forest = forest.sort_values(by=["isOverall", "state"], ascending=[True, True]).reset_index(drop=True)
    return forest

def make_forest_plots_strata(
    df_summary_all: pd.DataFrame,
    scenario: str,
    out_dir_rr: str,
    out_dir_tables: str,
):
    """
    NEW:
    - Forest plots OVERALL vs Age strata (FOREST_AGE_...)
    - Forest plots OVERALL vs Sex strata (FOREST_SEX_...)
    All based on OVERALL (state == 'OVERALL') cumulative RR 0–LAG_MAX.
    """
    if df_summary_all is None or df_summary_all.empty:
        return

    df_overall = df_summary_all[df_summary_all["state"] == "OVERALL"].copy()
    if df_overall.empty:
        return

    for dataset in ["HOSP", "MORT"]:
        df_d = df_overall[df_overall["dataset"] == dataset]
        if df_d.empty:
            continue

        total_outcomes = (
            df_d[df_d["outcome_group"] == "TOTAL"]["outcome"]
            .dropna()
            .unique()
            .tolist()
        )
        total_outcomes = sorted(total_outcomes)

        for tot_out in total_outcomes:
            # AGE
            forest_age = build_forest_data_strata(df_summary_all, dataset, tot_out, "AGE")
            if forest_age is not None and not forest_age.empty:
                csv_path_age = os.path.join(
                    out_dir_tables,
                    sanitize_filename(f"FOREST_AGE_{scenario}_{dataset}_{tot_out}.csv"),
                )
                save_csv_nature(forest_age, csv_path_age)
                plot_forest(
                    forest_age,
                    dataset,
                    f"{tot_out}_AGE",
                    f"{scenario}_AGE",
                    out_dir_rr,
                )

            # SEX
            forest_sex = build_forest_data_strata(df_summary_all, dataset, tot_out, "SEX")
            if forest_sex is not None and not forest_sex.empty:
                csv_path_sex = os.path.join(
                    out_dir_tables,
                    sanitize_filename(f"FOREST_SEX_{scenario}_{dataset}_{tot_out}.csv"),
                )
                save_csv_nature(forest_sex, csv_path_sex)
                plot_forest(
                    forest_sex,
                    dataset,
                    f"{tot_out}_SEX",
                    f"{scenario}_SEX",
                    out_dir_rr,
                )

# ============================================================
# GUIDE FILE
# ============================================================

def write_guide_txt(
    out_root: str,
    scenario: str,
    out_rr: str,
    out_rr_pm: str,
    out_rr_pm_lag0: str,
    out_tables: str,
):
    """
    Creates a GUIDE.TXT file in the scenario root folder describing subfolders.
    """
    guide_path = os.path.join(out_root, "GUIDE.TXT")
    lines = [
        "DLNM PM2.5 – episode-based (top 5% PM2.5 days)",
        f"Scenario: {scenario}",
        "",
        "Folder structure and content:",
        "",
        "RR_LAG\\",
        "    - Event-based DLNM relative risk vs lag (0–7 days).",
        "    - One curve per state and outcome, plus pooled OVERALL curves.",
        "    - Forest plots comparing:",
        "        * OVERALL vs States (state-level cumulative RR).",
        "        * OVERALL vs Age groups (FOREST_AGE_...).",
        "        * OVERALL vs Sex strata (FOREST_SEX_...).",
        "",
        "RR_PM\\",
        "    - Continuous concentration–response DLNM (RR vs PM2.5) by lag (0–7 days).",
        "    - Curves for each state and OVERALL (meta-analysis).",
        "",
        "RR_PM_LAG0\\",
        "    - Concentration–response DLNM for lag 0 only.",
        "    - Includes 95% confidence ribbons for each state and OVERALL.",
        "",
        "TABLES\\",
        "    - All CSV outputs used to build the figures and summaries:",
        "        * State-level RR(lag) and OVERALL meta-analysis.",
        "        * RR(PM) grids for each state and OVERALL.",
        "        * Summary tables (episodes, valid days, peak and cumulative RR).",
        "        * Forest-plot tables for states, ages and sex strata,",
        "          including weights, normalized weights, p values, and",
        "          p values for differences vs OVERALL.",
        "",
        "Notes:",
        "    - All column names in CSV files are formatted in English and camelCase,",
        "      without underscores, to match Nature-style publication standards.",
        "",
    ]
    with open(guide_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    log(f"GUIDE written to: {guide_path}")

# ============================================================
# MAIN
# ============================================================

def main():
    # --- checks ---
    assert_exists(HOSP_BASE_DIR, "HOSP_BASE_DIR")
    assert_exists(MORT_BASE_DIR, "MORT_BASE_DIR")
    assert_exists(EPISODES_CSV, "EPISODES_CSV")

    episodes_all = load_episodes_all(EPISODES_CSV)
    log(f"Episodes loaded (top 5% PM2.5): {episodes_all.shape[0]}")
    if episodes_all.empty:
        log("⚠️ No episodes found in episodes CSV. Aborting.")
        return

    hosp_files = list_excel_files(HOSP_BASE_DIR, "*_HOSPITALIZATION_FULL_DATA*.xlsx")
    mort_files = list_excel_files(MORT_BASE_DIR, "*_MORTALITY_FULL_DATA.xlsx")

    hosp_by_state: Dict[str, pd.DataFrame] = {}
    mort_by_state: Dict[str, pd.DataFrame] = {}

    for fp in hosp_files:
        st = infer_state_from_hosp_filename(fp)
        df = prepare_dataframe(fp, "HOSP")
        hosp_by_state[st] = df
        log(f"[HOSP] loaded {st}: n={df.shape[0]}")

    for fp in mort_files:
        st = infer_state_from_mort_filename(fp)
        df = prepare_dataframe(fp, "MORT")
        mort_by_state[st] = df
        log(f"[MORT] loaded {st}: n={df.shape[0]}")

    # NEW: build outcome lists dynamically (TOTAL + SEX + AGE etc.)
    CORE_OUTCOMES = build_core_outcomes(hosp_by_state, mort_by_state)
    log("\nDiscovered outcomes:")
    log(f"  HOSP outcomes: {len(CORE_OUTCOMES['HOSP'])}")
    log(f"  MORT outcomes: {len(CORE_OUTCOMES['MORT'])}")

    scenario = SCENARIO_UNIQUE
    log("\n" + "=" * 70)
    log(f"Running scenario: {scenario}")
    log("=" * 70)

    # OUTPUT DIRS
    OUT_ROOT = os.path.join(GLOBAL_OUT_ROOT, scenario)
    OUT_RR = os.path.join(OUT_ROOT, "RR_LAG")
    OUT_RR_PM = os.path.join(OUT_ROOT, "RR_PM")
    OUT_RR_PM_LAG0 = os.path.join(OUT_ROOT, "RR_PM_LAG0")
    OUT_TABLES = os.path.join(OUT_ROOT, "TABLES")
    for d in [OUT_ROOT, OUT_RR, OUT_RR_PM, OUT_RR_PM_LAG0, OUT_TABLES]:
        os.makedirs(d, exist_ok=True)

    self_test_saving(OUT_ROOT)

    ep = episodes_all.copy()
    save_csv_nature(
        ep,
        os.path.join(OUT_TABLES, f"EPISODES_USED_{scenario}.csv"),
    )

    states = sorted(ep["STATE"].unique().tolist())
    log(f"[{scenario}] States with episodes: {states}")

    ts_hosp_state: Dict[str, pd.DataFrame] = {}
    ts_mort_state: Dict[str, pd.DataFrame] = {}

    # Build time series per state with event indicator + covariates/time index/dow
    for st in states:
        eps_st = ep[ep["STATE"] == st].copy()

        if st in hosp_by_state:
            df_h = hosp_by_state[st].copy()
            df_h = add_event_indicator(df_h, eps_st, lag_after=LAG_MAX)
            df_h = df_h.sort_values(DATE_COL).reset_index(drop=True)
            df_h["TIME_INDEX"] = np.arange(df_h.shape[0]) + 1
            df_h["DOW"] = df_h[DATE_COL].dt.dayofweek
            ts_hosp_state[st] = df_h
            save_csv_nature(
                df_h,
                os.path.join(
                    OUT_TABLES,
                    sanitize_filename(f"TS_HOSP_{scenario}_{st}.csv"),
                ),
            )

        if st in mort_by_state:
            df_m = mort_by_state[st].copy()
            df_m = add_event_indicator(df_m, eps_st, lag_after=LAG_MAX)
            df_m = df_m.sort_values(DATE_COL).reset_index(drop=True)
            df_m["TIME_INDEX"] = np.arange(df_m.shape[0]) + 1
            df_m["DOW"] = df_m[DATE_COL].dt.dayofweek
            ts_mort_state[st] = df_m
            save_csv_nature(
                df_m,
                os.path.join(
                    OUT_TABLES,
                    sanitize_filename(f"TS_MORT_{scenario}_{st}.csv"),
                ),
            )

    # ------------------------------------------------------------
    # A) EVENT-BASED RR vs LAG (0..LAG_MAX)
    # ------------------------------------------------------------
    log(f"\n[{scenario}] Running EVENT-BASED DLNM per state (lag 0..{LAG_MAX})...")
    rr_tables_states = []

    for st in states:
        if st in ts_hosp_state and ts_hosp_state[st] is not None and not ts_hosp_state[st].empty:
            rr_h = run_dlnm_events_for_state(
                ts_hosp_state[st],
                "HOSP",
                st,
                outcomes=CORE_OUTCOMES["HOSP"],
                lag_max=LAG_MAX,
                df_time=TIME_DF,
                scenario=scenario,
            )
            if not rr_h.empty:
                save_csv_nature(
                    rr_h,
                    os.path.join(
                        OUT_TABLES,
                        sanitize_filename(f"RR_LAG_{scenario}_{st}_HOSP.csv"),
                    ),
                )
                rr_tables_states.append(rr_h)

                for outc in CORE_OUTCOMES["HOSP"]:
                    sub = rr_h[rr_h["outcome"] == outc].copy()
                    if not sub.empty:
                        plot_rr_lag_curve(
                            sub,
                            os.path.join(
                                OUT_RR,
                                sanitize_filename(
                                    f"RR_LAG_{scenario}_{st}_HOSP_{outc}"
                                ),
                            ),
                        )

        if st in ts_mort_state and ts_mort_state[st] is not None and not ts_mort_state[st].empty:
            rr_m = run_dlnm_events_for_state(
                ts_mort_state[st],
                "MORT",
                st,
                outcomes=CORE_OUTCOMES["MORT"],
                lag_max=LAG_MAX,
                df_time=TIME_DF,
                scenario=scenario,
            )
            if not rr_m.empty:
                save_csv_nature(
                    rr_m,
                    os.path.join(
                        OUT_TABLES,
                        sanitize_filename(f"RR_LAG_{scenario}_{st}_MORT.csv"),
                    ),
                )
                rr_tables_states.append(rr_m)

                for outc in CORE_OUTCOMES["MORT"]:
                    sub = rr_m[rr_m["outcome"] == outc].copy()
                    if not sub.empty:
                        plot_rr_lag_curve(
                            sub,
                            os.path.join(
                                OUT_RR,
                                sanitize_filename(
                                    f"RR_LAG_{scenario}_{st}_MORT_{outc}"
                                ),
                            ),
                        )

    if not rr_tables_states:
        log(f"\n[{scenario}] ⚠️ No state RR(lag) tables generated. Check inputs / number of events.")
        return

    rr_states = pd.concat(rr_tables_states, ignore_index=True)
    save_csv_nature(
        rr_states,
        os.path.join(OUT_TABLES, f"RR_LAG_{scenario}_ALL_STATES.csv"),
    )

    log(f"\n[{scenario}] Running random-effects meta-analysis (OVERALL by lag)...")
    rr_all = add_overall_meta_rr(rr_states)
    save_csv_nature(
        rr_all,
        os.path.join(OUT_TABLES, f"RR_LAG_{scenario}_ALL_WITH_META.csv"),
    )

    # Plot OVERALL RR(lag) for ALL outcomes
    for dataset in ["HOSP", "MORT"]:
        sub_overall = rr_all[
            (rr_all["state"] == "OVERALL") & (rr_all["dataset"] == dataset)
        ].copy()
        if sub_overall.empty:
            continue
        save_csv_nature(
            sub_overall,
            os.path.join(OUT_TABLES, f"RR_LAG_{scenario}_OVERALL_{dataset}.csv"),
        )

        for outc in CORE_OUTCOMES[dataset]:
            s2 = sub_overall[sub_overall["outcome"] == outc].copy()
            if not s2.empty:
                plot_rr_lag_curve(
                    s2,
                    os.path.join(
                        OUT_RR,
                        sanitize_filename(
                            f"RR_LAG_{scenario}_OVERALL_{dataset}_{outc}"
                        ),
                    ),
                )

    # ------------------------------------------------------------
    # B) CONTINUOUS PM RR vs PM (0..100 step 5) with curves by lag
    #    + lag-0 concentration–response with CI ribbon
    # ------------------------------------------------------------
    log(f"\n[{scenario}] Running CONTINUOUS-PM DLNM per state (PM grid 0..100 by 5; lag 0..{LAG_MAX})...")
    rr_pm_tables = []

    for st in states:
        if st in ts_hosp_state and ts_hosp_state[st] is not None and not ts_hosp_state[st].empty:
            rr_pm_h = run_dlnm_pm_by_lag_for_state(
                ts_hosp_state[st],
                "HOSP",
                st,
                outcomes=CORE_OUTCOMES["HOSP"],
                lag_max=LAG_MAX,
                df_time=TIME_DF,
                scenario=scenario,
            )
            if not rr_pm_h.empty:
                rr_pm_tables.append(rr_pm_h)
                save_csv_nature(
                    rr_pm_h,
                    os.path.join(
                        OUT_TABLES,
                        sanitize_filename(f"RR_PM_{scenario}_{st}_HOSP.csv"),
                    ),
                )

                # existing multi-lag plot
                for outc in CORE_OUTCOMES["HOSP"]:
                    sub = rr_pm_h[rr_pm_h["outcome"] == outc].copy()
                    if not sub.empty:
                        plot_rr_vs_pm(
                            sub,
                            os.path.join(
                                OUT_RR_PM,
                                sanitize_filename(
                                    f"RR_PM_{scenario}_{st}_HOSP_{outc}"
                                ),
                            ),
                            lag_max=LAG_MAX,
                        )

                        # NEW lag0 plot with CI ribbon
                        plot_rr_vs_pm_lag0_with_ci(
                            sub,
                            os.path.join(
                                OUT_RR_PM_LAG0,
                                sanitize_filename(
                                    f"RR_PM_LAG0_{scenario}_{st}_HOSP_{outc}"
                                ),
                            ),
                        )
                        # NEW lag0 CSV
                        lag0 = sub.copy()
                        lag0["lag"] = pd.to_numeric(lag0["lag"], errors="coerce").astype(int)
                        lag0 = lag0[lag0["lag"] == 0].copy()
                        if not lag0.empty:
                            save_csv_nature(
                                lag0,
                                os.path.join(
                                    OUT_TABLES,
                                    sanitize_filename(
                                        f"RR_PM_LAG0_{scenario}_{st}_HOSP_{outc}.csv"
                                    ),
                                ),
                            )

        if st in ts_mort_state and ts_mort_state[st] is not None and not ts_mort_state[st].empty:
            rr_pm_m = run_dlnm_pm_by_lag_for_state(
                ts_mort_state[st],
                "MORT",
                st,
                outcomes=CORE_OUTCOMES["MORT"],
                lag_max=LAG_MAX,
                df_time=TIME_DF,
                scenario=scenario,
            )
            if not rr_pm_m.empty:
                rr_pm_tables.append(rr_pm_m)
                save_csv_nature(
                    rr_pm_m,
                    os.path.join(
                        OUT_TABLES,
                        sanitize_filename(f"RR_PM_{scenario}_{st}_MORT.csv"),
                    ),
                )

                for outc in CORE_OUTCOMES["MORT"]:
                    sub = rr_pm_m[rr_pm_m["outcome"] == outc].copy()
                    if not sub.empty:
                        plot_rr_vs_pm(
                            sub,
                            os.path.join(
                                OUT_RR_PM,
                                sanitize_filename(
                                    f"RR_PM_{scenario}_{st}_MORT_{outc}"
                                ),
                            ),
                            lag_max=LAG_MAX,
                        )

                        # NEW lag0 plot with CI ribbon
                        plot_rr_vs_pm_lag0_with_ci(
                            sub,
                            os.path.join(
                                OUT_RR_PM_LAG0,
                                sanitize_filename(
                                    f"RR_PM_LAG0_{scenario}_{st}_MORT_{outc}"
                                ),
                            ),
                        )
                        # NEW lag0 CSV
                        lag0 = sub.copy()
                        lag0["lag"] = pd.to_numeric(lag0["lag"], errors="coerce").astype(int)
                        lag0 = lag0[lag0["lag"] == 0].copy()
                        if not lag0.empty:
                            save_csv_nature(
                                lag0,
                                os.path.join(
                                    OUT_TABLES,
                                    sanitize_filename(
                                        f"RR_PM_LAG0_{scenario}_{st}_MORT_{outc}.csv"
                                    ),
                                ),
                            )

    if rr_pm_tables:
        rr_pm_all_states = pd.concat(rr_pm_tables, ignore_index=True)
        save_csv_nature(
            rr_pm_all_states,
            os.path.join(OUT_TABLES, f"RR_PM_{scenario}_ALL_STATES.csv"),
        )

        log(f"\n[{scenario}] Running random-effects meta-analysis for RR~PM (OVERALL by pm & lag)...")
        rr_pm_overall = meta_overall_rr_pm(rr_pm_all_states)
        if rr_pm_overall is not None and not rr_pm_overall.empty:
            save_csv_nature(
                rr_pm_overall,
                os.path.join(OUT_TABLES, f"RR_PM_{scenario}_OVERALL.csv"),
            )

            # Existing multi-lag overall plot (kept)
            for dataset in ["HOSP", "MORT"]:
                for outc in CORE_OUTCOMES[dataset]:
                    sub = rr_pm_overall[
                        (rr_pm_overall["dataset"] == dataset) &
                        (rr_pm_overall["outcome"] == outc)
                    ].copy()
                    if not sub.empty:
                        plot_rr_vs_pm(
                            sub,
                            os.path.join(
                                OUT_RR_PM,
                                sanitize_filename(
                                    f"RR_PM_{scenario}_OVERALL_{dataset}_{outc}"
                                ),
                            ),
                            lag_max=LAG_MAX,
                        )

                        # NEW: overall lag0 concentration–response with CI ribbon + CSV
                        plot_rr_vs_pm_lag0_with_ci(
                            sub,
                            os.path.join(
                                OUT_RR_PM_LAG0,
                                sanitize_filename(
                                    f"RR_PM_LAG0_{scenario}_OVERALL_{dataset}_{outc}"
                                ),
                            ),
                        )
                        lag0 = sub.copy()
                        lag0["lag"] = pd.to_numeric(lag0["lag"], errors="coerce").astype(int)
                        lag0 = lag0[lag0["lag"] == 0].copy()
                        if not lag0.empty:
                            save_csv_nature(
                                lag0,
                                os.path.join(
                                    OUT_TABLES,
                                    sanitize_filename(
                                        f"RR_PM_LAG0_{scenario}_OVERALL_{dataset}_{outc}.csv"
                                    ),
                                ),
                            )
    else:
        log(f"[{scenario}] ⚠️ No RR~PM tables generated.")

    # ------------------------------------------------------------
    # SUMMARIES + FOREST (using cumulative RR 0..LAG_MAX)
    # ------------------------------------------------------------
    log(f"\n[{scenario}] Building summary tables (episodes / valid days / peak / cumulative RR 0..{LAG_MAX})...")
    df_state_summary = build_state_summary(
        rr_states,
        ts_hosp_state,
        ts_mort_state,
        states,
        CORE_OUTCOMES,
    )
    save_csv_nature(
        df_state_summary,
        os.path.join(OUT_TABLES, f"SUMMARY_{scenario}_STATES.csv"),
    )

    df_overall_summary = build_overall_summary_from_states(
        df_state_summary,
        rr_all,
        CORE_OUTCOMES,
    )
    df_summary_all = pd.concat([df_state_summary, df_overall_summary], ignore_index=True)
    save_csv_nature(
        df_summary_all,
        os.path.join(OUT_TABLES, f"SUMMARY_{scenario}_ALL.csv"),
    )

    # OPTIONAL: quick index of what was generated
    idx_outcomes = (
        df_summary_all[["dataset", "outcome", "outcome_group"]]
        .drop_duplicates()
        .sort_values(["dataset", "outcome_group", "outcome"])
    )
    save_csv_nature(
        idx_outcomes,
        os.path.join(OUT_TABLES, f"OUTCOMES_INDEX_{scenario}.csv"),
    )

    log(f"\n[{scenario}] Generating forest plots (cumulative RR 0–{LAG_MAX}) for TOTAL + SEX + AGE strata (states)...")
    make_forest_plots(df_summary_all, scenario, OUT_RR, OUT_TABLES)

    log(f"\n[{scenario}] Generating forest plots OVERALL vs Age groups and OVERALL vs Sex strata...")
    make_forest_plots_strata(df_summary_all, scenario, OUT_RR, OUT_TABLES)

    # GUIDE.TXT
    write_guide_txt(OUT_ROOT, scenario, OUT_RR, OUT_RR_PM, OUT_RR_PM_LAG0, OUT_TABLES)

    log(f"\n[{scenario}] ✅ DONE.")
    log(f"[{scenario}] Outputs in: {OUT_ROOT}")
    log(f"  RR(lag) curves and forests: {OUT_RR}")
    log(f"  RR(PM) multi-lag:            {OUT_RR_PM}")
    log(f"  RR(PM) LAG0 (with CI):       {OUT_RR_PM_LAG0}")
    log(f"  Tables (CSV):                {OUT_TABLES}")
    log("  GUIDE.TXT:                   describes folder contents.")
    log("\nAll done.")

if __name__ == "__main__":
    main()

