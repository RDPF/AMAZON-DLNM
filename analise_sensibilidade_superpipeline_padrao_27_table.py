#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# DLNM PM2.5 – AMAZON LEGAL (WILDFIRE EPISODES)
# GASPARINI-STYLE PIPELINE – LAG 0–7 (RR×LAG) + LAG 0 (RR×PM2.5)
#
# GERA TODAS AS SAÍDAS ESPECIFICADAS EM:
# 1) ARQUIVOS POR ESTADO (HOSP/<UF> e MORT/<UF>)
# 2) META-ANÁLISE (META/)
# 3) CURVAS POOLED LAG 0 (META/LAG0_PM25/)
# 4) FOREST PLOTS (FOREST/)
# 5) GUIDE.TXT
# 6) TABELA DE SENSIBILIDADE TIPO "SUPPLEMENTARY TABLE 1" (META/)
# 7) TABELA WIDE PUBLICATION-READY (TABLES/)
#
# ROOT: C:\NEWVERSIONGASPARRINI
# ============================================================

import os
import glob
import math
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# rpy2 / R
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import StrVector, FloatVector, ListVector

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

HOSP_BASE_DIR = (
    r"C:\dados3\DADOS - HOSP_ENV_LEGAL_AMAZON_DATA-20251201T163911Z-1-001"
    r"\DADOS - HOSP_ENV_LEGAL_AMAZON_DATA\Full"
)
MORT_BASE_DIR = r"C:\dados2\MORT"

# ROOT
GASPAR_ROOT = r"C:\NEWVERSIONGASPARRINI"
os.makedirs(GASPAR_ROOT, exist_ok=True)

HOSP_OUTPUT_ROOT = os.path.join(GASPAR_ROOT, "HOSP")
MORT_OUTPUT_ROOT = os.path.join(GASPAR_ROOT, "MORT")
META_OUTPUT_ROOT = os.path.join(GASPAR_ROOT, "META")
FOREST_OUTPUT_ROOT = os.path.join(GASPAR_ROOT, "FOREST")
LAG0_META_OUTPUT_ROOT = os.path.join(META_OUTPUT_ROOT, "LAG0_PM25")
TABLES_OUTPUT_ROOT = os.path.join(GASPAR_ROOT, "TABLES")  # pasta bem destacada para tabelas

os.makedirs(HOSP_OUTPUT_ROOT, exist_ok=True)
os.makedirs(MORT_OUTPUT_ROOT, exist_ok=True)
os.makedirs(META_OUTPUT_ROOT, exist_ok=True)
os.makedirs(FOREST_OUTPUT_ROOT, exist_ok=True)
os.makedirs(LAG0_META_OUTPUT_ROOT, exist_ok=True)
os.makedirs(TABLES_OUTPUT_ROOT, exist_ok=True)

# DLNM / exposure configuration
PM25_REFERENCE = 0.000144844  # µg/m³ (centering)
PM25_INCREMENT = 5.0          # contrast: ref + 5 µg/m³

# AGORA TUDO É LAG 0–7
LAG_MAX = 7                   # máximo lag usado no DLNM
LAG_MAX_SHORT = 7             # janela de interesse 0–7 (igual aqui)
DF_TIME_PER_YEAR = 7          # df/year for time spline

# Exposure knots (global quantiles of PM2.5)
PM_VAR_KNOT_PROBS = (0.10, 0.50, 0.90)

# Lag ticks for plots (apenas 0–7)
LAG_TICKS_007 = list(range(0, 8))    # 0,1,2,3,4,5,6,7

# Nature-like plotting style
COLOR_LINE_BLUE = "#0072B2"
COLOR_SHADE = "#A6BDD7"
SHADE_ALPHA = 0.55

# Outcomes (raw column names)
HOSP_OUTCOMES = [
    "HOSP_CIRC_TOTAL",
    "HOSP_CIRC_MASC",
    "HOSP_CIRC_FEM",
    "HOSP_CIRC_LT60",
    "HOSP_CIRC_>=60_ANOS",
    "HOSP_RESP_TOTAL",
    "HOSP_RESP_MASC",
    "HOSP_RESP_FEM",
    "HOSP_RESP_LT60",
    "HOSP_RESP_>=60_ANOS",
]

MORT_OUTCOMES = [
    "MORT_CIRC_TOTAL",
    "MORT_CIRC_MASC",
    "MORT_CIRC_FEM",
    "MORT_CIRC_LT60",
    "MORT_CIRC_>=60_ANOS",
    "MORT_RESP_TOTAL",
    "MORT_RESP_MASC",
    "MORT_RESP_FEM",
    "MORT_RESP_LT60",
    "MORT_RESP_>=60_ANOS",
]

# Containers for mvmeta exposure–response (cumulative)
meta_coefs_overall = {"HOSP": {y: [] for y in HOSP_OUTCOMES}, "MORT": {y: [] for y in MORT_OUTCOMES}}
meta_vcovs_overall = {"HOSP": {y: [] for y in HOSP_OUTCOMES}, "MORT": {y: [] for y in MORT_OUTCOMES}}
meta_labels_overall = {"HOSP": {y: [] for y in HOSP_OUTCOMES}, "MORT": {y: [] for y in MORT_OUTCOMES}}

# Containers for cumulative RR per lag per state (for meta and forest)
meta_cumlog_by_state = {"HOSP": {y: [] for y in HOSP_OUTCOMES}, "MORT": {y: [] for y in MORT_OUTCOMES}}
meta_cumse_by_state  = {"HOSP": {y: [] for y in HOSP_OUTCOMES}, "MORT": {y: [] for y in MORT_OUTCOMES}}
meta_lagseq_by_state = {"HOSP": {y: [] for y in HOSP_OUTCOMES}, "MORT": {y: [] for y in MORT_OUTCOMES}}
meta_state_names     = {"HOSP": {y: [] for y in HOSP_OUTCOMES}, "MORT": {y: [] for y in MORT_OUTCOMES}}

# Global PM ranges (for meta)
global_pm_min = {"HOSP": np.inf, "MORT": np.inf}
global_pm_max = {"HOSP": -np.inf, "MORT": -np.inf}

# Combined meta-summary (cumulative PM curves)
meta_summary_rows = []

# Class-level overall containers (for OVERALL_CLASSES forest)
overall_classes = {
    "HOSP": {"CIRC": {}, "RESP": {}},
    "MORT": {"CIRC": {}, "RESP": {}},
}

CLASS_LABEL_MAP = {
    "TOTAL": "Total",
    "LT60": "<60",
    ">=60_ANOS": "≥60",
    "MASC": "Male",
    "FEM": "Female",
}

AGE_LABELS = ["Total", "<60", "≥60"]
SEX_LABELS = ["Total", "Male", "Female"]

# ============================================================
# UTILITIES
# ============================================================

def log(msg: str):
    print(msg)

def list_excel_files(base_dir: str, pattern_suffix: str) -> list:
    pattern = os.path.join(base_dir, pattern_suffix)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{pattern_suffix}' in {base_dir}")
    log("Files found:")
    for f in files:
        log("  - " + os.path.basename(f))
    return files

def construct_lt60(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Build <60 years series as: total – >=60 years.
    Negative values are truncated to zero.
    """
    age_pairs = [
        (f"{prefix}_CIRC_TOTAL", f"{prefix}_CIRC_>=60_ANOS", f"{prefix}_CIRC_LT60"),
        (f"{prefix}_RESP_TOTAL", f"{prefix}_RESP_>=60_ANOS", f"{prefix}_RESP_LT60"),
    ]
    for total_col, ge60_col, lt60_col in age_pairs:
        if total_col in df.columns and ge60_col in df.columns:
            df[lt60_col] = df[total_col] - df[ge60_col]
            neg_mask = df[lt60_col] < 0
            if neg_mask.any():
                log(f"Warning: {int(neg_mask.sum())} negative values in {lt60_col}; set to zero.")
                df.loc[neg_mask, lt60_col] = 0
        else:
            log(f"Warning: missing {total_col} or {ge60_col}; cannot build {lt60_col}.")
    return df

def apply_end_of_year_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude 24–31 December and 1–6 January (HOSP).
    """
    mask_keep = ~(
        ((df["DATA"].dt.month == 12) & (df["DATA"].dt.day >= 24)) |
        ((df["DATA"].dt.month == 1) & (df["DATA"].dt.day <= 6))
    )
    return df.loc[mask_keep].reset_index(drop=True)

def prepare_state_dataframe(path: str, dataset_type: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "DATA" not in df.columns:
        raise ValueError(f"'DATA' column not found in {os.path.basename(path)}")
    df["DATA"] = pd.to_datetime(df["DATA"])
    df = df.sort_values("DATA").reset_index(drop=True)

    if dataset_type == "HOSP":
        df = construct_lt60(df, "HOSP")
        df = apply_end_of_year_filter(df)
    else:
        df = construct_lt60(df, "MORT")

    df["time"] = np.arange(1, len(df) + 1)
    df["dow"] = df["DATA"].dt.dayofweek  # 0–6

    pm_cols = [c for c in df.columns if c.startswith("WF_PM2.5_POND_")]
    if not pm_cols:
        raise ValueError(f"No WF_PM2.5_POND_ column in {os.path.basename(path)}")
    if len(pm_cols) > 1:
        log(f"Warning: multiple PM columns in {os.path.basename(path)}: {pm_cols}. Using {pm_cols[0]}")
    df = df.rename(columns={pm_cols[0]: "WF_PM2.5_POND_STATE"})
    return df

def compute_global_pm_knots(files: list, dataset_type: str, probs=(0.10, 0.50, 0.90)) -> np.ndarray:
    vals = []
    for pth in tqdm(files, desc=f"Computing global PM knots ({dataset_type})"):
        try:
            df = prepare_state_dataframe(pth, dataset_type=dataset_type)
            s = df["WF_PM2.5_POND_STATE"].to_numpy(dtype=float)
            s = s[np.isfinite(s)]
            if s.size:
                vals.append(s)
        except Exception as e:
            log(f"[knots] Skipping {os.path.basename(pth)} due to error: {e}")
            continue

    if not vals:
        raise RuntimeError(f"Cannot compute global knots for {dataset_type}: no valid PM values.")

    all_pm = np.concatenate(vals)
    all_pm = all_pm[np.isfinite(all_pm)]
    q = np.quantile(all_pm, probs).astype(float)

    eps = 1e-6
    for i in range(1, len(q)):
        if q[i] <= q[i - 1]:
            q[i] = q[i - 1] + eps
    return q

def safe_name(x: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in x)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ============================================================
# 0) LIST FILES + GLOBAL PM KNOTS
# ============================================================

hosp_files = list_excel_files(HOSP_BASE_DIR, "*_HOSPITALIZATION_FULL_DATA*.xlsx")
mort_files = list_excel_files(MORT_BASE_DIR, "*_MORTALITY_FULL_DATA.xlsx")

PM_KNOTS_HOSP = compute_global_pm_knots(hosp_files, "HOSP", probs=PM_VAR_KNOT_PROBS)
PM_KNOTS_MORT = compute_global_pm_knots(mort_files, "MORT", probs=PM_VAR_KNOT_PROBS)

log(f"\nGlobal PM knots (HOSP) at {PM_VAR_KNOT_PROBS}: {PM_KNOTS_HOSP}")
log(f"Global PM knots (MORT) at {PM_VAR_KNOT_PROBS}: {PM_KNOTS_MORT}\n")

# ============================================================
# R STAGE 1: DLNM BY STATE
# ============================================================

ro.globalenv["PM25_REFERENCE"] = PM25_REFERENCE
ro.globalenv["PM25_INCREMENT"] = PM25_INCREMENT
ro.globalenv["LAG_MAX"] = LAG_MAX
ro.globalenv["LAG_MAX_SHORT"] = LAG_MAX_SHORT
ro.globalenv["DF_TIME_PER_YEAR"] = DF_TIME_PER_YEAR

ro.globalenv["COLOR_LINE_BLUE"] = COLOR_LINE_BLUE
ro.globalenv["COLOR_SHADE"] = COLOR_SHADE
ro.globalenv["SHADE_ALPHA"] = SHADE_ALPHA

ro.globalenv["LAG_TICKS_007"] = FloatVector([float(x) for x in LAG_TICKS_007])

ro.globalenv["PM_KNOTS"] = FloatVector([1.0, 2.0, 3.0])  # placeholder, updated per dataset

r_stage1 = r"""
if (!requireNamespace("dlnm", quietly = TRUE)) stop("Package 'dlnm' is required.")
if (!requireNamespace("splines", quietly = TRUE)) stop("Package 'splines' is required.")
library(dlnm)
library(splines)
library(stats)
library(grDevices)

options(encoding = "UTF-8")

# Objetos globais usados para passar resultados de volta ao Python
last_summary      <- NULL
last_coef_overall <- NULL
last_vcov_overall <- NULL

last_cum_log <- NULL
last_cum_se  <- NULL
last_lag_seq <- NULL

last_meteo_table <- NULL

# ----------------------------------------------------------
# Funções auxiliares
# ----------------------------------------------------------

make_error_summary <- function(outcome, n_obs, msg_extra = "") {
  explanation <- paste0(
    "Stratum excluded from DLNM analysis: model not estimable. ",
    "Possible reasons: very low counts, severe collinearity or numerical failure. ",
    msg_extra
  )
  data.frame(
    stateName       = state_name,
    datasetType     = dataset_type,
    outcome         = outcome,
    nObs            = n_obs,
    used            = FALSE,
    pm25Ref         = PM25_REFERENCE,
    rrPm25Plus5     = NA_real_,
    rrPm25Plus5Low  = NA_real_,
    rrPm25Plus5High = NA_real_,
    modelUsed       = NA_character_,
    medianCiWidth   = NA_real_,
    maxCumulativeRR = NA_real_,
    nLagsProtective = NA_integer_,
    nLagsHarmful    = NA_integer_,
    diagnosticText  = explanation,
    stringsAsFactors = FALSE
  )
}

detect_precip_col <- function(nms) {
  candidates <- c(
    "PRECIP_MM", "PRECIPITATION_MM", "PRECIP_TOTAL_MM",
    "PRECIP_SUM_MM", "PRECIP_MEAN_MM", "RAIN_MM", "RAIN_SUM_MM"
  )
  for (c in candidates) {
    if (c %in% nms) return(c)
  }
  idx <- grep("PRECIP|RAIN", nms, ignore.case = TRUE)
  if (length(idx) > 0) return(nms[idx[1]])
  return(NA_character_)
}

# ----------------------------------------------------------
# Função principal: roda DLNM para um único desfecho
# ----------------------------------------------------------

run_dlnm_for_outcome <- function(outcome) {

  # reset das variáveis globais de saída
  last_summary      <<- NULL
  last_coef_overall <<- NULL
  last_vcov_overall <<- NULL
  last_cum_log      <<- NULL
  last_cum_se       <<- NULL
  last_lag_seq      <<- NULL
  last_meteo_table  <<- NULL

  var_pm   <- var_pm_name
  var_temp <- "TEMP_MEAN_C"

  # ------------------------------------------------------
  # Preparação de dados / meteo
  # ------------------------------------------------------
  dados$DATA <- as.Date(dados$DATA)

  # dow como fator (0 = Monday, ..., 6 = Sunday)
  dados$dow <- factor(
    dados$dow,
    levels = 0:6,
    labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
  )

  # Médias móveis de meteo, se não existirem
  if (!("TEMP_MA21" %in% names(dados))) {
    if (!var_temp %in% names(dados)) {
      last_summary <<- make_error_summary(
        outcome, nrow(dados),
        "Temperature column TEMP_MEAN_C not found."
      )
      return(invisible(NULL))
    }
    dados$TEMP_MA21 <- as.numeric(stats::filter(dados[[var_temp]],
                                                rep(1/21, 21), sides = 1))
  }

  if (!("RH_MA7" %in% names(dados))) {
    if (!("RH_MEAN_PCT" %in% names(dados))) {
      last_summary <<- make_error_summary(
        outcome, nrow(dados),
        "Relative humidity column RH_MEAN_PCT not found."
      )
      return(invisible(NULL))
    }
    dados$RH_MA7 <- as.numeric(stats::filter(dados[["RH_MEAN_PCT"]],
                                             rep(1/7, 7), sides = 1))
  }

  if (!("WIND_MA7" %in% names(dados))) {
    if (!("WIND_SPEED_MEAN_MPS" %in% names(dados))) {
      last_summary <<- make_error_summary(
        outcome, nrow(dados),
        "Wind column WIND_SPEED_MEAN_MPS not found."
      )
      return(invisible(NULL))
    }
    dados$WIND_MA7 <- as.numeric(stats::filter(dados[["WIND_SPEED_MEAN_MPS"]],
                                               rep(1/7, 7), sides = 1))
  }

  precip_col <- detect_precip_col(names(dados))
  if (!("PRECIP_MA7" %in% names(dados))) {
    if (!is.na(precip_col)) {
      dados$PRECIP_MA7 <- as.numeric(stats::filter(dados[[precip_col]],
                                                   rep(1/7, 7), sides = 1))
    } else {
      dados$PRECIP_MA7 <- NA_real_
    }
  }

  vars_key <- c(
    var_pm, outcome,
    "TEMP_MA21", "RH_MA7", "WIND_MA7", "PRECIP_MA7",
    "time", "dow"
  )

  cols_missing <- setdiff(vars_key, names(dados))
  if (length(cols_missing) > 0) {
    last_summary <<- make_error_summary(
      outcome, nrow(dados),
      paste("Missing variables:", paste(cols_missing, collapse = ", "))
    )
    return(invisible(NULL))
  }

  dados_model <- dados[complete.cases(dados[, vars_key]), vars_key]

  if (nrow(dados_model) < 40) {
    last_summary <<- make_error_summary(
      outcome, nrow(dados_model),
      "Sample size too small for stable DLNM."
    )
    return(invisible(NULL))
  }

  # ------------------------------------------------------
  # Definição de base cruzada (exposição)
  # ------------------------------------------------------
  lag_max  <- as.integer(LAG_MAX)
  knots_pm <- as.numeric(PM_KNOTS)

  cb_pm25 <- crossbasis(
    dados_model[[var_pm]],
    lag    = lag_max,
    argvar = list(fun = "ns", knots = knots_pm),
    arglag = list(fun = "ns", knots = logknots(lag_max, 3))
  )

  # df para tendência de longo prazo
  days_total <- as.numeric(diff(range(dados_model$time)))
  years <- days_total / 365
  df_time <- max(1, round(DF_TIME_PER_YEAR * years))

  form <- as.formula(
    paste0(
      "`", outcome, "` ~ cb_pm25 + dow + ns(time, df = ", df_time, ") + ",
      "ns(TEMP_MA21, df = 4) + ",
      "ns(RH_MA7, df = 4) + ",
      "ns(WIND_MA7, df = 3) + ",
      "ns(PRECIP_MA7, df = 3)"
    )
  )

  mod <- tryCatch(
    glm(form, family = quasipoisson(), data = dados_model),
    error = function(e) NULL
  )

  if (is.null(mod)) {
    last_summary <<- make_error_summary(
      outcome, nrow(dados_model),
      "Failure in glm fit."
    )
    return(invisible(NULL))
  }

  # ------------------------------------------------------
  # CURVA CUMULATIVA PM × RR (lag 0–LAG_MAX)
  # ------------------------------------------------------
  pm_min <- min(dados_model[[var_pm]], na.rm = TRUE)
  pm_max <- max(dados_model[[var_pm]], na.rm = TRUE)
  pm_seq <- seq(pm_min, pm_max, by = 1)

  pred_cum <- tryCatch(
    crosspred(
      cb_pm25,
      model = mod,
      cen   = PM25_REFERENCE,
      at    = pm_seq,
      cumul = TRUE
    ),
    error = function(e) NULL
  )

  if (is.null(pred_cum)) {
    last_summary <<- make_error_summary(
      outcome, nrow(dados_model),
      "Failure in crosspred (cumulative)."
    )
    return(invisible(NULL))
  }

  x_pm   <- pred_cum$predvar
  rr_pm  <- pred_cum$allRRfit
  low_pm <- pred_cum$allRRlow
  high_pm<- pred_cum$allRRhigh

  # Ponto de contraste: ref + incremento
  target_pm <- PM25_REFERENCE + PM25_INCREMENT
  idx_pm    <- which.min(abs(x_pm - target_pm))

  RR_PM25_5      <- rr_pm[idx_pm]
  RR_PM25_5_low  <- low_pm[idx_pm]
  RR_PM25_5_high <- high_pm[idx_pm]

  # Matriz cumulativa lag × PM
  cum_log_mat <- pred_cum$cumfit
  cum_se_mat  <- pred_cum$cumse

  lag_seq <- suppressWarnings(as.numeric(colnames(cum_log_mat)))
  if (any(is.na(lag_seq)) || length(lag_seq) == 0) {
    lag_seq <- 0:(ncol(cum_log_mat) - 1)
  }

  cum_log <- as.numeric(cum_log_mat[idx_pm, ])
  cum_se  <- as.numeric(cum_se_mat[idx_pm, ])

  last_lag_seq <<- lag_seq
  last_cum_log <<- cum_log
  last_cum_se  <<- cum_se

  cumRR      <- exp(cum_log)
  cumRR_low  <- exp(cum_log - 1.96 * cum_se)
  cumRR_high <- exp(cum_log + 1.96 * cum_se)

  ci_width        <- cumRR_high - cumRR_low
  median_ci_width <- median(ci_width, na.rm = TRUE)
  max_rr_lag      <- max(cumRR_high, na.rm = TRUE)

  prot_sig_flags <- cumRR_high < 1
  risk_sig_flags <- cumRR_low  > 1
  n_lag_prot_sig <- sum(prot_sig_flags, na.rm = TRUE)
  n_lag_risk_sig <- sum(risk_sig_flags, na.rm = TRUE)

  diag_text <- paste(
    sprintf("Median 95%% CI width across lags: %.3f", median_ci_width),
    sprintf("Maximum cumulative RR across lags: %.3f", max_rr_lag),
    sprintf("Significant protective lags (RR<1 & CI<1): %d", n_lag_prot_sig),
    sprintf("Significant harmful lags (RR>1 & CI>1): %d", n_lag_risk_sig),
    sep = " | "
  )

  safe_out <- gsub("[^A-Za-z0-9]+", "_", outcome)

  col_line  <- COLOR_LINE_BLUE
  col_shade <- adjustcolor(COLOR_SHADE, alpha.f = SHADE_ALPHA)

  # ----------------- CSV CURVA PM FULL ----------------------
  pm_curve_full_csv <- file.path(
    output_dir,
    paste0(safe_out, "_PM25_curve_cumulative_FULL_lag0_", LAG_MAX, ".csv")
  )
  write.csv(
    data.frame(
      pm              = x_pm,
      relativeRisk    = rr_pm,
      relativeRiskLow = low_pm,
      relativeRiskHigh= high_pm
    ),
    pm_curve_full_csv,
    row.names = FALSE
  )

  # ----------------- CSV CURVA PM CAP100 --------------------
  cap_mask <- is.finite(x_pm) & (x_pm <= 100)
  x_cap    <- x_pm[cap_mask]
  rr_cap   <- rr_pm[cap_mask]
  low_cap  <- low_pm[cap_mask]
  high_cap <- high_pm[cap_mask]

  pm_curve_cap_csv <- file.path(
    output_dir,
    paste0(safe_out, "_PM25_curve_cumulative_CAP100_lag0_", LAG_MAX, ".csv")
  )
  write.csv(
    data.frame(
      pm              = x_cap,
      relativeRisk    = rr_cap,
      relativeRiskLow = low_cap,
      relativeRiskHigh= high_cap
    ),
    pm_curve_cap_csv,
    row.names = FALSE
  )

  # ----------------- PLOTS PM CUMULATIVO FULL ---------------
  yrange_pm <- range(c(low_pm, high_pm, 1), na.rm = TRUE)

  pm_pdf <- file.path(
    output_dir,
    paste0(safe_out, "_PM25_overall_cumulative_FULL_lag0_", LAG_MAX, ".pdf")
  )
  pm_png <- sub("\\.pdf$", ".png", pm_pdf)
  pm_svg <- sub("\\.pdf$", ".svg", pm_pdf)

  pdf(pm_pdf, width = 7, height = 5)
  plot(x_pm, rr_pm, type = "n",
       xlab = "Wildfire-related PM2.5 (µg/m³)",
       ylab = "Relative Risk",
       main = "",
       ylim = yrange_pm)
  polygon(c(x_pm, rev(x_pm)), c(low_pm, rev(high_pm)),
          col = col_shade, border = NA)
  lines(x_pm, rr_pm, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  png(pm_png, width = 7, height = 5, units = "in", res = 300)
  plot(x_pm, rr_pm, type = "n",
       xlab = "Wildfire-related PM2.5 (µg/m³)",
       ylab = "Relative Risk",
       main = "",
       ylim = yrange_pm)
  polygon(c(x_pm, rev(x_pm)), c(low_pm, rev(high_pm)),
          col = col_shade, border = NA)
  lines(x_pm, rr_pm, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  svg(pm_svg, width = 7, height = 5)
  plot(x_pm, rr_pm, type = "n",
       xlab = "Wildfire-related PM2.5 (µg/m³)",
       ylab = "Relative Risk",
       main = "",
       ylim = yrange_pm)
  polygon(c(x_pm, rev(x_pm)), c(low_pm, rev(high_pm)),
          col = col_shade, border = NA)
  lines(x_pm, rr_pm, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  # ----------------- PLOTS PM CUMULATIVO CAP100 -------------
  yrange_pm2 <- range(c(low_cap, high_cap, 1), na.rm = TRUE)

  pm_cap_pdf <- file.path(
    output_dir,
    paste0(safe_out, "_PM25_overall_cumulative_CAP100_lag0_", LAG_MAX, ".pdf")
  )
  pm_cap_png <- sub("\\.pdf$", ".png", pm_cap_pdf)
  pm_cap_svg <- sub("\\.pdf$", ".svg", pm_cap_pdf)

  pdf(pm_cap_pdf, width = 7, height = 5)
  plot(x_cap, rr_cap, type = "n",
       xlab = "Wildfire-related PM2.5 (µg/m³)",
       ylab = "Relative Risk",
       main = "",
       ylim = yrange_pm2)
  polygon(c(x_cap, rev(x_cap)), c(low_cap, rev(high_cap)),
          col = col_shade, border = NA)
  lines(x_cap, rr_cap, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  png(pm_cap_png, width = 7, height = 5, units = "in", res = 300)
  plot(x_cap, rr_cap, type = "n",
       xlab = "Wildfire-related PM2.5 (µg/m³)",
       ylab = "Relative Risk",
       main = "",
       ylim = yrange_pm2)
  polygon(c(x_cap, rev(x_cap)), c(low_cap, rev(high_cap)),
          col = col_shade, border = NA)
  lines(x_cap, rr_cap, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  svg(pm_cap_svg, width = 7, height = 5)
  plot(x_cap, rr_cap, type = "n",
       xlab = "Wildfire-related PM2.5 (µg/m³)",
       ylab = "Relative Risk",
       main = "",
       ylim = yrange_pm2)
  polygon(c(x_cap, rev(x_cap)), c(low_cap, rev(high_cap)),
          col = col_shade, border = NA)
  lines(x_cap, rr_cap, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  # ------------------------------------------------------
  # CURVAS LAG-0 (RR × PM) – FULL e CAP100
  # ------------------------------------------------------
  pred_lag0 <- tryCatch(
    crosspred(
      cb_pm25,
      model = mod,
      cen   = PM25_REFERENCE,
      at    = pm_seq,
      cumul = FALSE
    ),
    error = function(e) NULL
  )

  if (!is.null(pred_lag0)) {

    mat_rr   <- pred_lag0$allRRfit
    mat_low  <- pred_lag0$allRRlow
    mat_high <- pred_lag0$allRRhigh

    # garante matriz (caso crosspred devolva vetor)
    if (is.null(dim(mat_rr))) {
      mat_rr   <- matrix(mat_rr,  ncol = 1)
      mat_low  <- matrix(mat_low, ncol = 1)
      mat_high <- matrix(mat_high,ncol = 1)
    }

    lag_names <- colnames(mat_rr)
    if (is.null(lag_names) || length(lag_names) == 0) {
      lag_names <- as.character(seq_len(ncol(mat_rr)) - 1L)
    }
    lag_num <- suppressWarnings(as.numeric(lag_names))
    if (length(lag_num) > 0 && any(is.finite(lag_num))) {
      idx0 <- which.min(abs(lag_num - 0))
    } else {
      idx0 <- 1L
    }
    if (idx0 < 1L || idx0 > ncol(mat_rr)) idx0 <- 1L

    rr0_full   <- as.numeric(mat_rr[,  idx0])
    low0_full  <- as.numeric(mat_low[, idx0])
    high0_full <- as.numeric(mat_high[,idx0])

    log0_full <- log(rr0_full)
    se0_full  <- (log(high0_full) - log(low0_full)) / (2 * 1.96)

    lag0_full_csv <- file.path(
      output_dir,
      paste0(safe_out, "_PM25_curve_lag0_FULL.csv")
    )
    write.csv(
      data.frame(
        pm              = x_pm,
        relativeRisk    = rr0_full,
        relativeRiskLow = low0_full,
        relativeRiskHigh= high0_full,
        logRelativeRisk = log0_full,
        standardError   = se0_full
      ),
      lag0_full_csv,
      row.names = FALSE
    )

    mask_cap2 = is.finite(x_pm) & (x_pm <= 100)
    x_lag0    = x_pm[mask_cap2]
    rr0       = rr0_full[mask_cap2]
    low0      = low0_full[mask_cap2]
    high0     = high0_full[mask_cap2]
    log0      = log0_full[mask_cap2]
    se0       = se0_full[mask_cap2]

    lag0_cap_csv <- file.path(
      output_dir,
      paste0(safe_out, "_PM25_curve_lag0_CAP100.csv")
    )
    write.csv(
      data.frame(
        pm              = x_lag0,
        relativeRisk    = rr0,
        relativeRiskLow = low0,
        relativeRiskHigh= high0,
        logRelativeRisk = log0,
        standardError   = se0
      ),
      lag0_cap_csv,
      row.names = FALSE
    )

    # plots FULL lag-0
    yrange_lag0_full <- range(c(low0_full, high0_full, 1), na.rm = TRUE)
    lag0_full_pdf <- file.path(
      output_dir,
      paste0(safe_out, "_PM25_lag0_FULL.pdf")
    )
    lag0_full_png <- sub("\\.pdf$", ".png", lag0_full_pdf)
    lag0_full_svg <- sub("\\.pdf$", ".svg", lag0_full_pdf)

    pdf(lag0_full_pdf, width = 7, height = 5)
    plot(x_pm, rr0_full, type = "n",
         xlab = "Wildfire-related PM2.5 (µg/m³)",
         ylab = "Relative Risk (lag 0)",
         main = "",
         ylim = yrange_lag0_full)
    polygon(c(x_pm, rev(x_pm)), c(low0_full, rev(high0_full)),
            col = col_shade, border = NA)
    lines(x_pm, rr0_full, lwd = 2, col = col_line)
    abline(h = 1, lty = 2)
    dev.off()

    png(lag0_full_png, width = 7, height = 5, units = "in", res = 300)
    plot(x_pm, rr0_full, type = "n",
         xlab = "Wildfire-related PM2.5 (µg/m³)",
         ylab = "Relative Risk (lag 0)",
         main = "",
         ylim = yrange_lag0_full)
    polygon(c(x_pm, rev(x_pm)), c(low0_full, rev(high0_full)),
            col = col_shade, border = NA)
    lines(x_pm, rr0_full, lwd = 2, col = col_line)
    abline(h = 1, lty = 2)
    dev.off()

    svg(lag0_full_svg, width = 7, height = 5)
    plot(x_pm, rr0_full, type = "n",
         xlab = "Wildfire-related PM2.5 (µg/m³)",
         ylab = "Relative Risk (lag 0)",
         main = "",
         ylim = yrange_lag0_full)
    polygon(c(x_pm, rev(x_pm)), c(low0_full, rev(high0_full)),
            col = col_shade, border = NA)
    lines(x_pm, rr0_full, lwd = 2, col = col_line)
    abline(h = 1, lty = 2)
    dev.off()

    # plots CAP100 lag-0
    yrange_lag0_cap <- range(c(low0, high0, 1), na.rm = TRUE)
    lag0_cap_pdf <- file.path(
      output_dir,
      paste0(safe_out, "_PM25_lag0_CAP100.pdf")
    )
    lag0_cap_png <- sub("\\.pdf$", ".png", lag0_cap_pdf)
    lag0_cap_svg <- sub("\\.pdf$", ".svg", lag0_cap_pdf)

    pdf(lag0_cap_pdf, width = 7, height = 5)
    plot(x_lag0, rr0, type = "n",
         xlab = "Wildfire-related PM2.5 (µg/m³)",
         ylab = "Relative Risk (lag 0)",
         main = "",
         ylim = yrange_lag0_cap)
    polygon(c(x_lag0, rev(x_lag0)), c(low0, rev(high0)),
            col = col_shade, border = NA)
    lines(x_lag0, rr0, lwd = 2, col = col_line)
    abline(h = 1, lty = 2)
    dev.off()

    png(lag0_cap_png, width = 7, height = 5, units = "in", res = 300)
    plot(x_lag0, rr0, type = "n",
         xlab = "Wildfire-related PM2.5 (µg/m³)",
         ylab = "Relative Risk (lag 0)",
         main = "",
         ylim = yrange_lag0_cap)
    polygon(c(x_lag0, rev(x_lag0)), c(low0, rev(high0)),
            col = col_shade, border = NA)
    lines(x_lag0, rr0, lwd = 2, col = col_line)
    abline(h = 1, lty = 2)
    dev.off()

    svg(lag0_cap_svg, width = 7, height = 5)
    plot(x_lag0, rr0, type = "n",
         xlab = "Wildfire-related PM2.5 (µg/m³)",
         ylab = "Relative Risk (lag 0)",
         main = "",
         ylim = yrange_lag0_cap)
    polygon(c(x_lag0, rev(x_lag0)), c(low0, rev(high0)),
            col = col_shade, border = NA)
    lines(x_lag0, rr0, lwd = 2, col = col_line)
    abline(h = 1, lty = 2)
    dev.off()
  }

  # ------------------------------------------------------
  # CURVA CUMLAG (RR × LAG, 0–LAG_MAX_SHORT)
  # ------------------------------------------------------
  short_mask <- is.finite(lag_seq) & (lag_seq <= LAG_MAX_SHORT)
  lag_s <- lag_seq[short_mask]
  rr_s  <- cumRR[short_mask]
  lo_s  <- cumRR_low[short_mask]
  hi_s  <- cumRR_high[short_mask]
  log_s <- cum_log[short_mask]
  se_s  <- cum_se[short_mask]

  cumlag_csv <- file.path(
    output_dir,
    paste0(safe_out, "_CUMLAG_curve_0_", LAG_MAX_SHORT, "_PMrefPlus5.csv")
  )
  write.csv(
    data.frame(
      lag             = lag_s,
      relativeRisk    = rr_s,
      relativeRiskLow = lo_s,
      relativeRiskHigh= hi_s,
      logRelativeRisk = log_s,
      standardError   = se_s
    ),
    cumlag_csv,
    row.names = FALSE
  )

  yrange_cum <- range(c(lo_s, hi_s, 1), na.rm = TRUE)
  lag_pdf <- file.path(
    output_dir,
    paste0(safe_out, "_PM25_cumlag_0_", LAG_MAX_SHORT, "_GASPAR.pdf")
  )
  lag_png <- sub("\\.pdf$", ".png", lag_pdf)
  lag_svg <- sub("\\.pdf$", ".svg", lag_pdf)

  pdf(lag_pdf, width = 7, height = 5)
  plot(lag_s, rr_s, type = "n",
       xlab = "Lag (days)",
       ylab = "Relative Risk",
       main = "",
       ylim = yrange_cum,
       xaxt = "n")
  axis(1, at = LAG_TICKS_007, labels = LAG_TICKS_007)
  polygon(c(lag_s, rev(lag_s)), c(lo_s, rev(hi_s)),
          col = col_shade, border = NA)
  lines(lag_s, rr_s, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  png(lag_png, width = 7, height = 5, units = "in", res = 300)
  plot(lag_s, rr_s, type = "n",
       xlab = "Lag (days)",
       ylab = "Relative Risk",
       main = "",
       ylim = yrange_cum,
       xaxt = "n")
  axis(1, at = LAG_TICKS_007, labels = LAG_TICKS_007)
  polygon(c(lag_s, rev(lag_s)), c(lo_s, rev(hi_s)),
          col = col_shade, border = NA)
  lines(lag_s, rr_s, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  svg(lag_svg, width = 7, height = 5)
  plot(lag_s, rr_s, type = "n",
       xlab = "Lag (days)",
       ylab = "Relative Risk",
       main = "",
       ylim = yrange_cum,
       xaxt = "n")
  axis(1, at = LAG_TICKS_007, labels = LAG_TICKS_007)
  polygon(c(lag_s, rev(lag_s)), c(lo_s, rev(hi_s)),
          col = col_shade, border = NA)
  lines(lag_s, rr_s, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  # ------------------------------------------------------
  # TABELA DE COVARIÁVEIS METEOROLÓGICAS
  # ------------------------------------------------------
  sm <- summary(mod)
  cf_tab <- sm$coefficients
  rn <- rownames(cf_tab)
  keep <- grepl("TEMP_MA21|RH_MA7|WIND_MA7|PRECIP_MA7", rn)
  met <- data.frame(
    term         = rn[keep],
    estimate     = cf_tab[keep, 1],
    standardError= cf_tab[keep, 2],
    tValue       = cf_tab[keep, 3],
    pValue       = cf_tab[keep, 4],
    stringsAsFactors = FALSE
  )
  last_meteo_table <<- met

  # ------------------------------------------------------
  # CROSSREDUCE overall (para mvmeta)
  # ------------------------------------------------------
  cr_overall <- tryCatch(
    crossreduce(cb_pm25, mod, type = "overall", cen = PM25_REFERENCE),
    error = function(e) NULL
  )

  if (is.null(cr_overall)) {
    last_coef_overall <<- NULL
    last_vcov_overall <<- NULL
  } else {
    last_coef_overall <<- cr_overall$coef
    last_vcov_overall <<- cr_overall$vcov
  }

  last_summary <<- data.frame(
    stateName       = state_name,
    datasetType     = dataset_type,
    outcome         = outcome,
    nObs            = nrow(dados_model),
    used            = TRUE,
    pm25Ref         = PM25_REFERENCE,
    rrPm25Plus5     = RR_PM25_5,
    rrPm25Plus5Low  = RR_PM25_5_low,
    rrPm25Plus5High = RR_PM25_5_high,
    modelUsed       = "ns_exposure_global_knots__ns_logknots_lag__meteo_TEMP_RH_WIND_PRECIP",
    medianCiWidth   = median_ci_width,
    maxCumulativeRR = max_rr_lag,
    nLagsProtective = n_lag_prot_sig,
    nLagsHarmful    = n_lag_risk_sig,
    diagnosticText  = diag_text,
    stringsAsFactors = FALSE
  )

  invisible(NULL)
}
"""
ro.r(r_stage1)

# ============================================================
# STAGE 1 (PYTHON): LOOP STATES – HOSP / MORT
# ============================================================

def process_dataset(files, dtype, outcomes, pm_knots, out_root):
    ro.globalenv["PM_KNOTS"] = FloatVector(list(map(float, pm_knots)))

    all_state_summaries = []

    for data_path in files:
        fname = os.path.basename(data_path)
        if dtype == "HOSP":
            state_name = fname.split("_HOSPITALIZATION")[0]
        else:
            state_name = fname.replace("_MORTALITY_FULL_DATA.xlsx", "")

        log("\n===================================================")
        log(f"Processing {dtype} – state: {state_name}")
        log(f"File: {fname}")
        log("===================================================\n")

        state_out_dir = os.path.join(out_root, state_name)
        ensure_dir(state_out_dir)

        try:
            df = prepare_state_dataframe(data_path, dataset_type=dtype)
        except Exception as e:
            log(f"[STATE] Error preparing {dtype} {state_name}: {e}")
            continue

        pm_min_state = df["WF_PM2.5_POND_STATE"].min(skipna=True)
        pm_max_state = df["WF_PM2.5_POND_STATE"].max(skipna=True)
        global_pm_min[dtype] = min(global_pm_min[dtype], pm_min_state)
        global_pm_max[dtype] = max(global_pm_max[dtype], pm_max_state)

        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["dados"] = df.copy()

        ro.globalenv["output_dir"]  = state_out_dir.replace("\\", "/")
        ro.globalenv["var_pm_name"] = "WF_PM2.5_POND_STATE"
        ro.globalenv["state_name"]  = state_name
        ro.globalenv["dataset_type"] = dtype

        summaries_state = []
        meteo_rows_all = []

        for outcome in tqdm(outcomes, desc=f"DLNM {dtype} – {state_name}"):
            ro.globalenv["current_outcome"] = outcome
            try:
                ro.r("run_dlnm_for_outcome(current_outcome)")
            except Exception as e:
                log(f"[DLNM] Error in R for {dtype} {state_name} {outcome}: {e}")
                continue

            with localconverter(ro.default_converter + pandas2ri.converter):
                s_r = ro.globalenv["last_summary"]
                s_df = pd.DataFrame(s_r)
            summaries_state.append(s_df)

            met_r = ro.globalenv["last_meteo_table"]
            if met_r is not None:
                with localconverter(ro.default_converter + pandas2ri.converter):
                    met_df = pd.DataFrame(met_r)
                if not met_df.empty:
                    met_df.insert(0, "state", state_name)
                    met_df.insert(1, "datasetType", dtype)
                    met_df.insert(2, "outcome", outcome)
                    meteo_rows_all.append(met_df)

            coef_r = ro.globalenv["last_coef_overall"]
            vcov_r = ro.globalenv["last_vcov_overall"]
            if coef_r is not None and vcov_r is not None:
                try:
                    coef_np = np.array(coef_r, dtype=float)
                    vcov_np = np.array(ro.r["as.matrix"](vcov_r), dtype=float)
                except Exception as e:
                    log(f"[DLNM] Error converting coef/vcov for {dtype} {state_name} {outcome}: {e}")
                else:
                    meta_coefs_overall[dtype][outcome].append(coef_np)
                    meta_vcovs_overall[dtype][outcome].append(vcov_np)
                    meta_labels_overall[dtype][outcome].append(state_name)

            lag_seq_r = ro.globalenv["last_lag_seq"]
            cum_log_r = ro.globalenv["last_cum_log"]
            cum_se_r  = ro.globalenv["last_cum_se"]
            if lag_seq_r is not None and cum_log_r is not None and cum_se_r is not None:
                lag_seq_np = np.array(lag_seq_r, dtype=float)
                cum_log_np = np.array(cum_log_r, dtype=float)
                cum_se_np  = np.array(cum_se_r, dtype=float)
                if lag_seq_np.size == cum_log_np.size == cum_se_np.size and lag_seq_np.size >= 2:
                    meta_lagseq_by_state[dtype][outcome].append(lag_seq_np)
                    meta_cumlog_by_state[dtype][outcome].append(cum_log_np)
                    meta_cumse_by_state[dtype][outcome].append(cum_se_np)
                    meta_state_names[dtype][outcome].append(state_name)

        if summaries_state:
            summary_df_state = pd.concat(summaries_state, ignore_index=True)
            all_state_summaries.append(summary_df_state)

            resumo_csv_path = os.path.join(
                state_out_dir,
                f"Summary_GASPAR_{dtype}_PM25_lag0_{LAG_MAX}_{state_name}.csv"
            )
            summary_df_state.to_csv(resumo_csv_path, index=False, encoding="utf-8-sig")
            log(f"{dtype} summary saved: {resumo_csv_path}")

        if meteo_rows_all:
            meteo_all = pd.concat(meteo_rows_all, ignore_index=True)
            meteo_csv = os.path.join(
                state_out_dir,
                f"Meteo_terms_{dtype}_{state_name}.csv"
            )
            meteo_all.to_csv(meteo_csv, index=False, encoding="utf-8-sig")
            log(f"{dtype} meteo terms saved: {meteo_csv}")

    if all_state_summaries:
        all_df = pd.concat(all_state_summaries, ignore_index=True)
        resumo_geral = os.path.join(
            GASPAR_ROOT,
            f"Summary_multi_state_GASPAR_{dtype}_PM25_lag0_{LAG_MAX}.csv"
        )
        all_df.to_csv(resumo_geral, index=False, encoding="utf-8-sig")
        log(f"\nMulti-state {dtype} summary saved: {resumo_geral}")
    else:
        log(f"\n[WARN] No valid state summaries for {dtype}.")
    return

process_dataset(hosp_files, "HOSP", HOSP_OUTCOMES, PM_KNOTS_HOSP, HOSP_OUTPUT_ROOT)
process_dataset(mort_files, "MORT", MORT_OUTCOMES, PM_KNOTS_MORT, MORT_OUTPUT_ROOT)

# ============================================================
# R STAGE 2: MVMeta pooled PM curve + pooled CUMLAG (overall)
# ============================================================

r_stage2 = r"""
if (!requireNamespace("mvmeta", quietly = TRUE)) stop("Package 'mvmeta' is required.")
if (!requireNamespace("dlnm", quietly = TRUE)) stop("Package 'dlnm' is required.")
if (!requireNamespace("splines", quietly = TRUE)) stop("Package 'splines' is required.")
library(mvmeta)
library(dlnm)
library(splines)
library(grDevices)

last_meta_summary <- NULL

run_mvmeta_for_outcome <- function(Y_mat, S_list, labels,
                                   outcome_name, dataset_type,
                                   pm_seq_global, meta_dir, tag) {

  if (is.null(Y_mat) || is.null(S_list) || nrow(Y_mat) == 0) {
    last_meta_summary <<- NULL
    return(invisible(NULL))
  }

  rownames(Y_mat) <- labels
  mv_fit <- tryCatch(
    mvmeta(Y_mat ~ 1, S_list, method = "reml"),
    error = function(e) NULL
  )
  if (is.null(mv_fit)) {
    last_meta_summary <<- NULL
    return(invisible(NULL))
  }

  coef_mv <- coef(mv_fit)
  vcov_mv <- vcov(mv_fit)

  knots_var <- as.numeric(PM_KNOTS)
  pm_seq <- as.numeric(pm_seq_global)
  bvar <- onebasis(pm_seq, fun = "ns", knots = knots_var)

  cp_meta <- tryCatch(
    crosspred(
      bvar,
      coef       = coef_mv,
      vcov       = vcov_mv,
      model.link = "log",
      cen        = PM25_REFERENCE
    ),
    error = function(e) NULL
  )
  if (is.null(cp_meta)) {
    last_meta_summary <<- NULL
    return(invisible(NULL))
  }

  safe_out   <- gsub("[^A-Za-z0-9_]+", "_", outcome_name)
  safe_dtype <- gsub("[^A-Za-z0-9_]+", "_", dataset_type)

  col_line  <- COLOR_LINE_BLUE
  col_shade <- adjustcolor(COLOR_SHADE, alpha.f = SHADE_ALPHA)

  pdf_file <- file.path(meta_dir, paste0("META_", safe_dtype, "_", tag, "_PM25_", safe_out, ".pdf"))
  png_file <- sub("\\.pdf$", ".png", pdf_file)
  svg_file <- sub("\\.pdf$", ".svg", pdf_file)

  ylim_all <- range(c(cp_meta$allRRlow, cp_meta$allRRhigh, 1), na.rm = TRUE)

  pdf(pdf_file, width = 7, height = 5)
  plot(cp_meta$predvar, cp_meta$allRRfit, type = "n",
       xlab = "Wildfire-related PM2.5 (µg/m³)",
       ylab = "Relative Risk",
       main = "",
       ylim = ylim_all)
  polygon(c(cp_meta$predvar, rev(cp_meta$predvar)),
          c(cp_meta$allRRlow,  rev(cp_meta$allRRhigh)),
          col = col_shade, border = NA)
  lines(cp_meta$predvar, cp_meta$allRRfit, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  png(png_file, width = 7, height = 5, units = "in", res = 300)
  plot(cp_meta$predvar, cp_meta$allRRfit, type = "n",
       xlab = "Wildfire-related PM2.5 (µg/m³)",
       ylab = "Relative Risk",
       main = "",
       ylim = ylim_all)
  polygon(c(cp_meta$predvar, rev(cp_meta$predvar)),
          c(cp_meta$allRRlow,  rev(cp_meta$allRRhigh)),
          col = col_shade, border = NA)
  lines(cp_meta$predvar, cp_meta$allRRfit, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  svg(svg_file, width = 7, height = 5)
  plot(cp_meta$predvar, cp_meta$allRRfit, type = "n",
       xlab = "Wildfire-related PM2.5 (µg/m³)",
       ylab = "Relative Risk",
       main = "",
       ylim = ylim_all)
  polygon(c(cp_meta$predvar, rev(cp_meta$predvar)),
          c(cp_meta$allRRlow,  rev(cp_meta$allRRhigh)),
          col = col_shade, border = NA)
  lines(cp_meta$predvar, cp_meta$allRRfit, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  curve_file <- file.path(
    meta_dir,
    paste0("META_", safe_dtype, "_", tag, "_PM25_", safe_out, ".csv")
  )
  write.csv(
    data.frame(
      pm        = cp_meta$predvar,
      relativeRisk     = cp_meta$allRRfit,
      relativeRiskLow  = cp_meta$allRRlow,
      relativeRiskHigh = cp_meta$allRRhigh
    ),
    curve_file, row.names = FALSE
  )

  target_pm <- PM25_REFERENCE + PM25_INCREMENT
  idx_5 <- which.min(abs(cp_meta$predvar - target_pm))

  rr_5      <- cp_meta$allRRfit[idx_5]
  rr_5_low  <- cp_meta$allRRlow[idx_5]
  rr_5_high <- cp_meta$allRRhigh[idx_5]

  last_meta_summary <<- data.frame(
    datasetType     = dataset_type,
    outcome         = outcome_name,
    tag             = tag,
    pm25Ref         = PM25_REFERENCE,
    pm25RefPlus5    = target_pm,
    rrPm25Plus5     = rr_5,
    rrPm25Plus5Low  = rr_5_low,
    rrPm25Plus5High = rr_5_high,
    stringsAsFactors = FALSE
  )

  invisible(NULL)
}

save_plot_overall_cumlag <- function(lag_seq, rr, rr_low, rr_high,
                                     dataset_type, outcome_name,
                                     meta_dir, tag) {

  safe_out   <- gsub("[^A-Za-z0-9_]+", "_", outcome_name)
  safe_dtype <- gsub("[^A-Za-z0-9_]+", "_", dataset_type)

  col_line  <- COLOR_LINE_BLUE
  col_shade <- adjustcolor(COLOR_SHADE, alpha.f = SHADE_ALPHA)

  pdf_file <- file.path(meta_dir, paste0("META_", safe_dtype, "_", tag, "_CUMLAG_", safe_out, ".pdf"))
  png_file <- sub("\\.pdf$", ".png", pdf_file)
  svg_file <- sub("\\.pdf$", ".svg", pdf_file)
  csv_file <- file.path(meta_dir, paste0("META_", safe_dtype, "_", tag, "_CUMLAG_", safe_out, ".csv"))

  out_df <- data.frame(
    lag             = lag_seq,
    relativeRisk    = rr,
    relativeRiskLow = rr_low,
    relativeRiskHigh= rr_high
  )
  write.csv(out_df, csv_file, row.names = FALSE)

  ylim_all <- range(c(rr_low, rr_high, 1), na.rm = TRUE)

  pdf(pdf_file, width = 7, height = 5)
  plot(lag_seq, rr, type = "n",
       xlab = "Lag (days)",
       ylab = "Relative Risk",
       main = "",
       ylim = ylim_all,
       xaxt = "n")
  axis(1, at = LAG_TICKS_007, labels = LAG_TICKS_007)
  polygon(c(lag_seq, rev(lag_seq)), c(rr_low, rev(rr_high)),
          col = col_shade, border = NA)
  lines(lag_seq, rr, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  png(png_file, width = 7, height = 5, units = "in", res = 300)
  plot(lag_seq, rr, type = "n",
       xlab = "Lag (days)",
       ylab = "Relative Risk",
       main = "",
       ylim = ylim_all,
       xaxt = "n")
  axis(1, at = LAG_TICKS_007, labels = LAG_TICKS_007)
  polygon(c(lag_seq, rev(lag_seq)), c(rr_low, rev(rr_high)),
          col = col_shade, border = NA)
  lines(lag_seq, rr, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  svg(svg_file, width = 7, height = 5)
  plot(lag_seq, rr, type = "n",
       xlab = "Lag (days)",
       ylab = "Relative Risk",
       main = "",
       ylim = ylim_all,
       xaxt = "n")
  axis(1, at = LAG_TICKS_007, labels = LAG_TICKS_007)
  polygon(c(lag_seq, rev(lag_seq)), c(rr_low, rev(rr_high)),
          col = col_shade, border = NA)
  lines(lag_seq, rr, lwd = 2, col = col_line)
  abline(h = 1, lty = 2)
  dev.off()

  invisible(NULL)
}
"""
ro.r(r_stage2)

# ============================================================
# R: FUNÇÃO LEVE PARA ANÁLISE DE SENSIBILIDADE
# ============================================================

r_sens = r"""
# ============================================================
# R: Função leve para análise de sensibilidade
# ============================================================

run_dlnm_sensitivity <- function(
  dados,
  outcome,
  lag_max        = LAG_MAX,
  lag_window_max = LAG_MAX_SHORT,
  arglag_mode    = c("logknots", "df"),
  df_lag         = 3,
  df_temp        = 4,
  df_rh          = 4,
  df_wind        = 3,
  df_precip      = 3,
  df_time_per_year = DF_TIME_PER_YEAR
) {
  arglag_mode <- match.arg(arglag_mode)

  var_pm <- var_pm_name

  dados$DATA <- as.Date(dados$DATA)

  if (!("time" %in% names(dados))) {
    dados$time <- seq_len(nrow(dados))
  }
  if (!("dow" %in% names(dados))) {
    dados$dow <- as.integer(format(dados$DATA, "%w"))
  }
  dados$dow <- factor(
    dados$dow,
    levels = 0:6,
    labels = c("Sun","Mon","Tue","Wed","Thu","Fri","Sat")
  )

  # --------- Médias móveis meteo ---------
  if (!("TEMP_MA21" %in% names(dados))) {
    if (!("TEMP_MEAN_C" %in% names(dados))) return(c(logRR = NA_real_, se = NA_real_))
    dados$TEMP_MA21 <- as.numeric(stats::filter(dados[["TEMP_MEAN_C"]],
                                                rep(1/21, 21), sides = 1))
  }

  if (!("RH_MA7" %in% names(dados))) {
    if (!("RH_MEAN_PCT" %in% names(dados))) return(c(logRR = NA_real_, se = NA_real_))
    dados$RH_MA7 <- as.numeric(stats::filter(dados[["RH_MEAN_PCT"]],
                                             rep(1/7, 7), sides = 1))
  }

  if (!("WIND_MA7" %in% names(dados))) {
    if (!("WIND_SPEED_MEAN_MPS" %in% names(dados))) return(c(logRR = NA_real_, se = NA_real_))
    dados$WIND_MA7 <- as.numeric(stats::filter(dados[["WIND_SPEED_MEAN_MPS"]],
                                               rep(1/7, 7), sides = 1))
  }

  if (!("PRECIP_MA7" %in% names(dados))) {
    precip_col <- detect_precip_col(names(dados))
    if (!is.na(precip_col)) {
      dados$PRECIP_MA7 <- as.numeric(stats::filter(dados[[precip_col]],
                                                   rep(1/7, 7), sides = 1))
    } else {
      dados$PRECIP_MA7 <- NA_real_
    }
  }

  vars_key <- c(var_pm, outcome, "TEMP_MA21", "RH_MA7",
                "WIND_MA7", "PRECIP_MA7", "time", "dow")
  cols_missing <- setdiff(vars_key, names(dados))
  if (length(cols_missing) > 0L) {
    return(c(logRR = NA_real_, se = NA_real_))
  }

  dados_model <- dados[complete.cases(dados[, vars_key]), vars_key]
  if (nrow(dados_model) < 40L) {
    return(c(logRR = NA_real_, se = NA_real_))
  }

  lag_max <- as.integer(lag_max)
  knots_var <- as.numeric(PM_KNOTS)

  if (arglag_mode == "logknots") {
    arglag_list <- list(fun = "ns", knots = logknots(lag_max, 3))
  } else {
    arglag_list <- list(fun = "ns", df = df_lag)
  }

  cb_pm25 <- dlnm::crossbasis(
    dados_model[[var_pm]],
    lag    = lag_max,
    argvar = list(fun = "ns", knots = knots_var),
    arglag = arglag_list
  )

  days_total <- as.numeric(diff(range(dados_model$time)))
  years <- days_total / 365
  df_time <- max(1, round(df_time_per_year * years))

  form <- as.formula(
    paste0(
      "`", outcome, "` ~ cb_pm25 + dow + ns(time, df = ", df_time, ") + ",
      "ns(TEMP_MA21, df = ", df_temp, ") + ",
      "ns(RH_MA7, df = ", df_rh, ") + ",
      "ns(WIND_MA7, df = ", df_wind, ") + ",
      "ns(PRECIP_MA7, df = ", df_precip, ")"
    )
  )

  mod <- tryCatch(
    glm(form, family = quasipoisson(), data = dados_model),
    error = function(e) NULL
  )
  if (is.null(mod)) {
    return(c(logRR = NA_real_, se = NA_real_))
  }

  pm_min <- min(dados_model[[var_pm]], na.rm = TRUE)
  pm_max <- max(dados_model[[var_pm]], na.rm = TRUE)
  pm_seq <- seq(pm_min, pm_max, by = 1)

  pred_pm <- tryCatch(
    dlnm::crosspred(
      cb_pm25,
      model = mod,
      cen   = PM25_REFERENCE,
      at    = pm_seq,
      cumul = TRUE
    ),
    error = function(e) NULL
  )
  if (is.null(pred_pm)) {
    return(c(logRR = NA_real_, se = NA_real_))
  }

  x_pm  <- pred_pm$predvar
  cf    <- pred_pm$cumfit
  cfse  <- pred_pm$cumse
  lag_seq <- suppressWarnings(as.numeric(colnames(cf)))
  if (any(is.na(lag_seq)) || length(lag_seq) == 0L) {
    lag_seq <- 0:(ncol(cf) - 1L)
  }

  target_pm <- PM25_REFERENCE + PM25_INCREMENT
  idx_pm  <- which.min(abs(x_pm - target_pm))
  idx_lag <- which.min(abs(lag_seq - lag_window_max))

  cum_log <- as.numeric(cf[idx_pm,  idx_lag])
  cum_se  <- as.numeric(cfse[idx_pm, idx_lag])

  if (!is.finite(cum_log) || !is.finite(cum_se) || cum_se <= 0) {
    return(c(logRR = NA_real_, se = NA_real_))
  }

  c(logRR = cum_log, se = cum_se)
}
"""
ro.r(r_sens)

# ============================================================
# AUX: FIXED-EFFECT POOLING FOR CUMLAG (ACROSS STATES)
# ============================================================

def pooled_cumlag_fixed_effect(lag_list, log_list, se_list):
    lag_ref = lag_list[0].astype(int)
    k = lag_ref.size

    logs = []
    ses = []
    for lg, ll, ss in zip(lag_list, log_list, se_list):
        lg_i = lg.astype(int)
        ll = np.asarray(ll, dtype=float)
        ss = np.asarray(ss, dtype=float)
        if lg_i.size != k or not np.all(lg_i == lag_ref):
            mapping = {int(L): (float(a), float(b)) for L, a, b in zip(lg_i, ll, ss)}
            aligned_log = np.full(k, np.nan, dtype=float)
            aligned_se  = np.full(k, np.nan, dtype=float)
            for j, L in enumerate(lag_ref):
                if int(L) in mapping:
                    aligned_log[j], aligned_se[j] = mapping[int(L)]
            ll, ss = aligned_log, aligned_se
        logs.append(ll)
        ses.append(ss)

    logs = np.vstack(logs)
    ses = np.vstack(ses)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / (ses ** 2)
    w[~np.isfinite(w)] = 0.0
    logs[~np.isfinite(logs)] = np.nan

    pooled_log = np.full(k, np.nan, dtype=float)
    pooled_se  = np.full(k, np.nan, dtype=float)
    for j in range(k):
        wj = w[:, j]
        lj = logs[:, j]
        ok = (wj > 0) & np.isfinite(lj)
        if ok.sum() < 2:
            continue
        sw = wj[ok].sum()
        pooled_log[j] = (wj[ok] * lj[ok]).sum() / sw
        pooled_se[j]  = np.sqrt(1.0 / sw)

    pooled_rr = np.exp(pooled_log)
    pooled_lo = np.exp(pooled_log - 1.96 * pooled_se)
    pooled_hi = np.exp(pooled_log + 1.96 * pooled_se)
    return lag_ref, pooled_log, pooled_se, pooled_rr, pooled_lo, pooled_hi

# ============================================================
# AUX: NORMAL CDF + P-VALUES + WEIGHTS FOR FORESTS
# ============================================================

def norm_cdf(z: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def two_sided_pvalue(log_rr: float, se: float) -> float:
    """Two-sided p-value for logRR with standard error se."""
    if not np.isfinite(log_rr) or not np.isfinite(se) or se <= 0:
        return np.nan
    z = log_rr / se
    return 2.0 * (1.0 - norm_cdf(abs(z)))

def add_weights_and_pvalues(
    df: pd.DataFrame,
    ref_label_col: str,
    ref_label_value: str
) -> pd.DataFrame:
    """
    Add weight, weightNormalized, pValue and pValueDifference to forest CSVs.
    """
    df = df.copy()
    logrr = df["logRelativeRisk"].to_numpy(dtype=float)
    se = df["standardError"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        weight = 1.0 / (se ** 2)
    weight[~np.isfinite(weight) | (se <= 0)] = np.nan
    total_w = np.nansum(weight)
    if total_w > 0:
        weight_norm = weight / total_w
    else:
        weight_norm = np.full_like(weight, np.nan)

    df["weight"] = weight
    df["weightNormalized"] = weight_norm
    df["pValue"] = [two_sided_pvalue(l, s) for l, s in zip(logrr, se)]

    p_diff = np.full_like(logrr, np.nan, dtype=float)
    if ref_label_value in df[ref_label_col].values:
        idx_ref = df.index[df[ref_label_col] == ref_label_value][0]
        log_ref = df.loc[idx_ref, "logRelativeRisk"]
        se_ref = df.loc[idx_ref, "standardError"]
        if np.isfinite(log_ref) and np.isfinite(se_ref) and se_ref > 0:
            for i in range(len(df)):
                if i == idx_ref:
                    continue
                li = logrr[i]
                sei = se[i]
                if not np.isfinite(li) or not np.isfinite(sei) or sei <= 0:
                    continue
                se_diff = math.sqrt(sei ** 2 + se_ref ** 2)
                p_diff[i] = two_sided_pvalue(li - log_ref, se_diff)

    df["pValueDifference"] = p_diff
    return df

# ============================================================
# STAGE 2 (PYTHON): META (PM CURVE CUM + CUMLAG OVERALL 0–7)
# ============================================================

for dtype, outcomes_list, pm_knots in [
    ("HOSP", HOSP_OUTCOMES, PM_KNOTS_HOSP),
    ("MORT", MORT_OUTCOMES, PM_KNOTS_MORT),
]:
    pm_min = global_pm_min[dtype]
    pm_max = global_pm_max[dtype]
    if not np.isfinite(pm_min) or not np.isfinite(pm_max):
        log(f"No PM range for {dtype}, skipping meta.")
        continue

    meta_dir_dtype = os.path.join(META_OUTPUT_ROOT, dtype)
    ensure_dir(meta_dir_dtype)

    ro.globalenv["PM_KNOTS"] = FloatVector(list(map(float, pm_knots)))
    ro.globalenv["meta_dir"] = meta_dir_dtype.replace("\\", "/")

    pm_seq_full = np.linspace(pm_min, pm_max, 120)
    pm_cap_max = min(100.0, float(pm_max))
    pm_cap_min = max(0.0, float(pm_min))
    if pm_cap_max <= pm_cap_min:
        pm_seq_cap = np.array([pm_cap_min, pm_cap_max], dtype=float)
    else:
        pm_seq_cap = np.linspace(pm_cap_min, pm_cap_max, 101)

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["pm_seq_full"] = pd.Series(pm_seq_full)
        ro.globalenv["pm_seq_cap"]  = pd.Series(pm_seq_cap)

    for outcome_name in outcomes_list:
        coefs_list = meta_coefs_overall[dtype][outcome_name]
        vcovs_list = meta_vcovs_overall[dtype][outcome_name]
        labels     = meta_labels_overall[dtype][outcome_name]

        # --------- mvmeta pooled cumulative PM curves (FULL and CAP100) ---------
        if not coefs_list:
            log(f"No mvmeta inputs for {dtype} – {outcome_name} (PM curve).")
        else:
            p = coefs_list[0].shape[0]
            if any(c.shape[0] != p for c in coefs_list):
                log(f"[META] Inconsistent coef length across states for {dtype} – {outcome_name}; skipping pooled PM curve for this outcome.")
            else:
                Y_np = np.vstack(coefs_list)
                with localconverter(ro.default_converter + pandas2ri.converter):
                    ro.globalenv["Y_mat"] = pd.DataFrame(Y_np)
                ro.globalenv["labels_vec"] = StrVector(labels)

                S_elems = []
                for vc in vcovs_list:
                    S_elems.append(
                        ro.r["matrix"](FloatVector(vc.flatten(order="F")), nrow=p, ncol=p)
                    )
                ro.globalenv["S_list"] = ListVector(
                    {str(i + 1): S_elems[i] for i in range(len(S_elems))}
                )

                ro.globalenv["this_outcome_name"] = outcome_name
                ro.globalenv["this_dataset_type"] = dtype

                # FULL
                ro.globalenv["tag"] = "overall_FULL"
                try:
                    ro.r(
                        "run_mvmeta_for_outcome(as.matrix(Y_mat), S_list, labels_vec, "
                        "this_outcome_name, this_dataset_type, pm_seq_full, meta_dir, tag)"
                    )
                except Exception as e:
                    log(f"[META] Error in mvmeta (FULL) for {dtype} – {outcome_name}: {e}")
                    ro.globalenv["last_meta_summary"] = None
                else:
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        meta_s_r = ro.globalenv["last_meta_summary"]
                        if meta_s_r is not None:
                            meta_s_df = pd.DataFrame(meta_s_r)
                            if not meta_s_df.empty:
                                meta_summary_rows.append(meta_s_df.iloc[0].to_dict())

                # CAP100
                ro.globalenv["tag"] = "overall_CAP100"
                try:
                    ro.r(
                        "run_mvmeta_for_outcome(as.matrix(Y_mat), S_list, labels_vec, "
                        "this_outcome_name, this_dataset_type, pm_seq_cap, meta_dir, tag)"
                    )
                except Exception as e:
                    log(f"[META] Error in mvmeta (CAP100) for {dtype} – {outcome_name}: {e}")
                    ro.globalenv["last_meta_summary"] = None
                else:
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        meta_s_r = ro.globalenv["last_meta_summary"]
                        if meta_s_r is not None:
                            meta_s_df = pd.DataFrame(meta_s_r)
                            if not meta_s_df.empty:
                                meta_summary_rows.append(meta_s_df.iloc[0].to_dict())

        # ------------------ pooled cumulative RR vs lag (overall, 0–7) -------------
        lag_list = meta_lagseq_by_state[dtype][outcome_name]
        log_list = meta_cumlog_by_state[dtype][outcome_name]
        se_list  = meta_cumse_by_state[dtype][outcome_name]
        if not lag_list or not log_list or not se_list:
            log(f"No CUMLAG vectors for {dtype} – {outcome_name}, skipping pooled CUMLAG.")
            continue

        lag_ref, pooled_log, pooled_se, pooled_rr, pooled_lo, pooled_hi = pooled_cumlag_fixed_effect(
            lag_list, log_list, se_list
        )

        mask7 = (lag_ref <= LAG_MAX_SHORT)
        lag7 = lag_ref[mask7]
        rr7  = pooled_rr[mask7]
        lo7  = pooled_lo[mask7]
        hi7  = pooled_hi[mask7]

        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["lag_seq_overall_07"] = pd.Series(lag7.astype(float))
            ro.globalenv["rr_overall_07"]      = pd.Series(rr7.astype(float))
            ro.globalenv["rr_low_overall_07"]  = pd.Series(lo7.astype(float))
            ro.globalenv["rr_high_overall_07"] = pd.Series(hi7.astype(float))

        ro.globalenv["this_dataset_type"] = dtype
        ro.globalenv["this_outcome_name"] = outcome_name
        ro.globalenv["tag"] = f"overall_0_{LAG_MAX_SHORT}"

        try:
            ro.r(
                "save_plot_overall_cumlag(lag_seq_overall_07, rr_overall_07, "
                "rr_low_overall_07, rr_high_overall_07, "
                "this_dataset_type, this_outcome_name, meta_dir, tag)"
            )
        except Exception as e:
            log(f"[META] Error saving pooled CUMLAG plot for {dtype} – {outcome_name}: {e}")

# Save combined meta summary (cumulative PM curves)
if meta_summary_rows:
    df_meta_all = pd.DataFrame(meta_summary_rows)
    fpath_all = os.path.join(
        META_OUTPUT_ROOT,
        "META_summary_HOSP_MORT_PM25_plus5_FULL_and_CAP100.csv"
    )
    df_meta_all.to_csv(fpath_all, index=False, encoding="utf-8-sig")
    log(f"\nCombined meta-analysis summary saved: {fpath_all}")
else:
    log("\nNo meta-analysis summaries produced.")

# ============================================================
# FOREST PLOTS: STATE + OVERALL (BY OUTCOME) – APENAS 0–7
# ============================================================

def extract_cum_at_lag(lag_seq, cum_log, cum_se, target_lag):
    lag_i = lag_seq.astype(int)
    if target_lag in set(lag_i.tolist()):
        j = int(np.where(lag_i == target_lag)[0][0])
        return float(cum_log[j]), float(cum_se[j])
    return np.nan, np.nan

def make_forest(df, out_prefix, title):
    df2 = df.copy().reset_index(drop=True)
    y = np.arange(len(df2))[::-1]

    fig = plt.figure(figsize=(7, max(4, 0.35 * len(df2) + 1.5)))
    ax = plt.gca()

    ax.errorbar(
        df2["relativeRisk"],
        y,
        xerr=[
            df2["relativeRisk"] - df2["lower95CI"],
            df2["upper95CI"] - df2["relativeRisk"],
        ],
        fmt="o",
        capsize=3,
    )

    ax.axvline(1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df2["state"])
    ax.set_xlabel("Relative Risk")
    ax.set_title(title)
    ax.set_ylim(-1, len(df2))

    xmin = np.nanmin(df2["lower95CI"].to_numpy())
    xmax = np.nanmax(df2["upper95CI"].to_numpy())
    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
        pad = 0.08 * (xmax - xmin)
        ax.set_xlim(max(0.1, xmin - pad), xmax + pad)

    fig.tight_layout()
    fig.savefig(out_prefix + ".pdf")
    fig.savefig(out_prefix + ".png", dpi=200)
    fig.savefig(out_prefix + ".svg")
    plt.close(fig)

# helper to parse group and class from outcome name
def parse_group_and_class(outcome: str):
    parts = outcome.split("_")
    group = None
    cls = None
    for token in parts:
        if token in ("CIRC", "RESP"):
            group = token
    for token in reversed(parts):
        if token in CLASS_LABEL_MAP:
            cls = token
            break
    return group, cls

for dtype, outcomes in [("HOSP", HOSP_OUTCOMES), ("MORT", MORT_OUTCOMES)]:
    dtype_dir = os.path.join(FOREST_OUTPUT_ROOT, dtype)
    ensure_dir(dtype_dir)

    for outcome in outcomes:
        lag_list = meta_lagseq_by_state[dtype][outcome]
        log_list = meta_cumlog_by_state[dtype][outcome]
        se_list  = meta_cumse_by_state[dtype][outcome]
        st_list  = meta_state_names[dtype][outcome]
        if not lag_list:
            continue

        rows_07 = []

        for state_name, lag_seq, cum_log, cum_se in zip(st_list, lag_list, log_list, se_list):
            L07, SE07 = extract_cum_at_lag(lag_seq, cum_log, cum_se, target_lag=LAG_MAX_SHORT)

            rr07 = np.exp(L07) if np.isfinite(L07) else np.nan
            lo07 = np.exp(L07 - 1.96 * SE07) if np.isfinite(L07) and np.isfinite(SE07) else np.nan
            hi07 = np.exp(L07 + 1.96 * SE07) if np.isfinite(L07) and np.isfinite(SE07) else np.nan

            rows_07.append(
                {
                    "state": state_name,
                    "logRelativeRisk": L07,
                    "standardError": SE07,
                    "relativeRisk": rr07,
                    "lower95CI": lo07,
                    "upper95CI": hi07,
                }
            )

        df07 = pd.DataFrame(rows_07).dropna(
            subset=["relativeRisk", "lower95CI", "upper95CI"], how="any"
        )

        # pooled overall (cumulative 0–7) para incluir como linha OVERALL
        lag_ref, pooled_log, pooled_se, pooled_rr, pooled_lo, pooled_hi = pooled_cumlag_fixed_effect(
            lag_list, log_list, se_list
        )
        L07p, SE07p = extract_cum_at_lag(lag_ref, pooled_log, pooled_se, LAG_MAX_SHORT)

        overall07 = {
            "state": "OVERALL",
            "logRelativeRisk": L07p,
            "standardError": SE07p,
            "relativeRisk": float(np.exp(L07p)) if np.isfinite(L07p) else np.nan,
            "lower95CI": float(np.exp(L07p - 1.96 * SE07p))
            if np.isfinite(L07p) and np.isfinite(SE07p)
            else np.nan,
            "upper95CI": float(np.exp(L07p + 1.96 * SE07p))
            if np.isfinite(L07p) and np.isfinite(SE07p)
            else np.nan,
        }

        df07 = pd.concat([df07, pd.DataFrame([overall07])], ignore_index=True)

        # Add weights and p-values (overall vs states) + normalized weight
        if not df07.empty:
            df07 = add_weights_and_pvalues(df07, ref_label_col="state", ref_label_value="OVERALL")

        out_base = os.path.join(dtype_dir, safe_name(outcome))
        df07.to_csv(
            out_base + f"_FOREST_cum0_{LAG_MAX_SHORT}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        title07 = (
            f"{dtype} – {outcome}\n"
            f"Cumulative effect (lag 0–{LAG_MAX_SHORT}) at PM ref+{PM25_INCREMENT:g}"
        )

        try:
            make_forest(
                df07,
                out_base + f"_FOREST_cum0_{LAG_MAX_SHORT}",
                title=title07,
            )
        except Exception as e:
            log(f"[FOREST] Error plotting forest for {dtype} – {outcome}: {e}")

        # ----------------- STORE OVERALL BY CLASS (for OVERALL_CLASSES, apenas 0–7) -------
        group, cls = parse_group_and_class(outcome)
        if group is not None and cls in CLASS_LABEL_MAP and np.isfinite(L07p) and np.isfinite(SE07p):
            label = CLASS_LABEL_MAP[cls]
            if label not in overall_classes[dtype][group]:
                overall_classes[dtype][group][label] = {}
            overall_classes[dtype][group][label][f"lag0_{LAG_MAX_SHORT}"] = overall07

# ============================================================
# OVERALL_CLASSES FORESTS (Total, <60, ≥60, Male, Female) – SÓ 0–7
# + AGE-ONLY E SEX-ONLY
# ============================================================

def make_class_forest(df, out_prefix, title):
    df2 = df.copy().reset_index(drop=True)
    y = np.arange(len(df2))[::-1]

    fig = plt.figure(figsize=(7, max(4, 0.4 * len(df2) + 1.5)))
    ax = plt.gca()

    ax.errorbar(
        df2["relativeRisk"],
        y,
        xerr=[
            df2["relativeRisk"] - df2["lower95CI"],
            df2["upper95CI"] - df2["relativeRisk"],
        ],
        fmt="o",
        capsize=3,
    )
    ax.axvline(1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df2["class"])
    ax.set_xlabel("Relative Risk")
    ax.set_title(title)
    ax.set_ylim(-1, len(df2))

    xmin = np.nanmin(df2["lower95CI"].to_numpy())
    xmax = np.nanmax(df2["upper95CI"].to_numpy())
    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
        pad = 0.08 * (xmax - xmin)
        ax.set_xlim(max(0.1, xmin - pad), xmax + pad)

    fig.tight_layout()
    fig.savefig(out_prefix + ".pdf")
    fig.savefig(out_prefix + ".png", dpi=200)
    fig.savefig(out_prefix + ".svg")
    plt.close(fig)

for dtype in ("HOSP", "MORT"):
    dtype_dir = os.path.join(FOREST_OUTPUT_ROOT, dtype)
    for group in ("CIRC", "RESP"):
        classes_dict = overall_classes[dtype][group]
        if not classes_dict:
            continue

        rows_07 = []
        for class_label, effects in classes_dict.items():
            e07 = effects.get(f"lag0_{LAG_MAX_SHORT}")
            if e07 is not None:
                rows_07.append(
                    {
                        "class": class_label,
                        "logRelativeRisk": e07["logRelativeRisk"],
                        "standardError": e07["standardError"],
                        "relativeRisk": e07["relativeRisk"],
                        "lower95CI": e07["lower95CI"],
                        "upper95CI": e07["upper95CI"],
                    }
                )

        if rows_07:
            df07_all = pd.DataFrame(rows_07)
            df07_all = add_weights_and_pvalues(df07_all, ref_label_col="class", ref_label_value="Total")

            out_base_all = os.path.join(
                dtype_dir,
                f"FOREST_CLASSES_{dtype}_{group}_ALL_cum0_{LAG_MAX_SHORT}"
            )
            df07_all.to_csv(out_base_all + ".csv", index=False, encoding="utf-8-sig")
            title = (
                f"{dtype} – {group}\n"
                f"Overall classes (Total, <60, ≥60, Male, Female), "
                f"cumulative effect (lag 0–{LAG_MAX_SHORT}) at PM ref+{PM25_INCREMENT:g}"
            )
            try:
                make_class_forest(df07_all, out_base_all, title)
            except Exception as e:
                log(f"[FOREST_CLASS] Error plotting ALL classes for {dtype} – {group}: {e}")

            # AGE-ONLY forest (Total, <60, ≥60)
            df07_age = df07_all[df07_all["class"].isin(AGE_LABELS)].copy()
            if len(df07_age) >= 2:
                out_base_age = os.path.join(
                    dtype_dir,
                    f"FOREST_CLASSES_{dtype}_{group}_AGES_cum0_{LAG_MAX_SHORT}"
                )
                df07_age.to_csv(out_base_age + ".csv", index=False, encoding="utf-8-sig")
                title_age = (
                    f"{dtype} – {group}\n"
                    f"Age strata (Total, <60, ≥60), cumulative effect (lag 0–{LAG_MAX_SHORT}) "
                    f"at PM ref+{PM25_INCREMENT:g}"
                )
                try:
                    make_class_forest(df07_age, out_base_age, title_age)
                except Exception as e:
                    log(f"[FOREST_CLASS] Error plotting AGE classes for {dtype} – {group}: {e}")

            # SEX-ONLY forest (Total, Male, Female)
            df07_sex = df07_all[df07_all["class"].isin(SEX_LABELS)].copy()
            if len(df07_sex) >= 2:
                out_base_sex = os.path.join(
                    dtype_dir,
                    f"FOREST_CLASSES_{dtype}_{group}_SEX_cum0_{LAG_MAX_SHORT}"
                )
                df07_sex.to_csv(out_base_sex + ".csv", index=False, encoding="utf-8-sig")
                title_sex = (
                    f"{dtype} – {group}\n"
                    f"Sex strata (Total, Male, Female), cumulative effect (lag 0–{LAG_MAX_SHORT}) "
                    f"at PM ref+{PM25_INCREMENT:g}"
                )
                try:
                    make_class_forest(df07_sex, out_base_sex, title_sex)
                except Exception as e:
                    log(f"[FOREST_CLASS] Error plotting SEX classes for {dtype} – {group}: {e}")

# ============================================================
# OVERALL LAG-0 PM CURVES (FIXED-EFFECT META ACROSS STATES)
# ============================================================

def pooled_lag0_pm_curve(dtype: str, outcomes_list):
    if dtype == "HOSP":
        base_root = HOSP_OUTPUT_ROOT
    else:
        base_root = MORT_OUTPUT_ROOT

    pm_grid = np.arange(0.0, 101.0, 1.0)

    for outcome in outcomes_list:
        state_list = meta_state_names[dtype][outcome]
        if not state_list:
            log(f"[LAG0_META] No states for {dtype} – {outcome}, skipping lag-0 pooled curve.")
            continue

        safe_out = safe_name(outcome)
        state_curves = []

        for state_name in state_list:
            state_dir = os.path.join(base_root, state_name)
            fpath = os.path.join(state_dir, f"{safe_out}_PM25_curve_lag0_CAP100.csv")
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
            if "logRelativeRisk" not in df.columns or "standardError" not in df.columns:
                rr = df["relativeRisk"].to_numpy(dtype=float)
                low = df["relativeRiskLow"].to_numpy(dtype=float)
                high = df["relativeRiskHigh"].to_numpy(dtype=float)
                log_rr = np.log(rr)
                se = (np.log(high) - np.log(low)) / (2.0 * 1.96)
                df["logRelativeRisk"] = log_rr
                df["standardError"] = se
            state_curves.append((state_name, df))

        if len(state_curves) < 2:
            log(f"[LAG0_META] Not enough state lag-0 curves for {dtype} – {outcome} to pool.")
            continue

        pooled_pm = []
        pooled_rr = []
        pooled_lo = []
        pooled_hi = []

        for pm_val in pm_grid:
            logs = []
            ses = []
            for _, df_state in state_curves:
                pm_state = df_state["pm"].to_numpy(dtype=float)
                idx = np.argmin(np.abs(pm_state - pm_val))
                if not np.isfinite(pm_state[idx]):
                    continue
                if abs(pm_state[idx] - pm_val) > 0.5:
                    continue  # do not extrapolate too far
                lrr = float(df_state["logRelativeRisk"].iloc[idx])
                se = float(df_state["standardError"].iloc[idx])
                if np.isfinite(lrr) and np.isfinite(se) and se > 0:
                    logs.append(lrr)
                    ses.append(se)
            if len(logs) < 2:
                pooled_pm.append(pm_val)
                pooled_rr.append(np.nan)
                pooled_lo.append(np.nan)
                pooled_hi.append(np.nan)
                continue
            logs = np.array(logs)
            ses = np.array(ses)
            w = 1.0 / (ses ** 2)
            sw = w.sum()
            mu = (w * logs).sum() / sw
            se_p = math.sqrt(1.0 / sw)
            rr_p = math.exp(mu)
            lo_p = math.exp(mu - 1.96 * se_p)
            hi_p = math.exp(mu + 1.96 * se_p)
            pooled_pm.append(pm_val)
            pooled_rr.append(rr_p)
            pooled_lo.append(lo_p)
            pooled_hi.append(hi_p)

        df_pool = pd.DataFrame(
            {
                "pm": pooled_pm,
                "relativeRisk": pooled_rr,
                "relativeRiskLow": pooled_lo,
                "relativeRiskHigh": pooled_hi,
            }
        )
        df_pool = df_pool[np.isfinite(df_pool["relativeRisk"])]

        if df_pool.empty:
            log(f"[LAG0_META] All pooled lag-0 RR are NaN for {dtype} – {outcome}, skipping saving.")
            continue

        out_dir = os.path.join(LAG0_META_OUTPUT_ROOT, dtype)
        ensure_dir(out_dir)
        out_base = os.path.join(
            out_dir,
            f"META_{dtype}_overall_lag0_PM25_{safe_out}"
        )
        df_pool.to_csv(out_base + ".csv", index=False, encoding="utf-8-sig")

        # Plot (Nature-like)
        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()
        pm_vals = df_pool["pm"].to_numpy()
        rr_vals = df_pool["relativeRisk"].to_numpy()
        lo_vals = df_pool["relativeRiskLow"].to_numpy()
        hi_vals = df_pool["relativeRiskHigh"].to_numpy()

        ax.fill_between(pm_vals, lo_vals, hi_vals, alpha=0.35)
        ax.plot(pm_vals, rr_vals, linewidth=2)
        ax.axhline(1.0, linestyle="--")
        ax.set_xlabel("Wildfire-related PM2.5 (µg/m³)")
        ax.set_ylabel("Relative Risk (lag 0)")
        ax.set_title(
            f"{dtype} – {outcome}\n"
            f"Fixed-effect pooled concentration–response at lag 0"
        )
        ymin = np.nanmin(lo_vals)
        ymax = np.nanmax(hi_vals)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = 0.08 * (ymax - ymin)
            ax.set_ylim(max(0.1, ymin - pad), ymax + pad)
        fig.tight_layout()
        fig.savefig(out_base + ".pdf")
        fig.savefig(out_base + ".png", dpi=200)
        fig.savefig(out_base + ".svg")
        plt.close(fig)

pooled_lag0_pm_curve("HOSP", HOSP_OUTCOMES)
pooled_lag0_pm_curve("MORT", MORT_OUTCOMES)

# ============================================================
# GUIDE.TXT – DESCRIPTION OF OUTPUT FOLDERS (ATUALIZADO PARA 0–7)
# ============================================================

guide_path = os.path.join(GASPAR_ROOT, "GUIDE.TXT")
with open(guide_path, "w", encoding="utf-8") as f:
    f.write("DLNM PM2.5 – Amazon Legal (wildfire episodes)\r\n")
    f.write("Gasparrini-style pipeline – hospitalization (HOSP) and mortality (MORT)\r\n")
    f.write("\r\n")
    f.write("Root folder:\r\n")
    f.write(f"  {GASPAR_ROOT}\r\n")
    f.write("\r\n")
    f.write("Folder structure:\r\n")
    f.write("  HOSP/  : State-level DLNM outputs for hospitalization outcomes.\r\n")
    f.write("           One subfolder per state with:\r\n")
    f.write("             - Cumulative PM2.5 concentration–response curves (FULL and CAP100; lag 0–7).\r\n")
    f.write("             - Single-lag (lag 0) PM2.5 curves (FULL and CAP100).\r\n")
    f.write("             - Cumulative RR vs lag curves (0–7) at PM ref+5.\r\n")
    f.write("             - Summary_GASPAR_* files (epidemiological summaries).\r\n")
    f.write("             - Meteorological regression coefficients (Meteo_terms_*).\r\n")
    f.write("\r\n")
    f.write("  MORT/  : Same structure as HOSP/, but for mortality outcomes.\r\n")
    f.write("\r\n")
    f.write("  META/  : Meta-analysis results pooled across states.\r\n")
    f.write("           - META/HOSP/ and META/MORT/ contain:\r\n")
    f.write("               * Pooled cumulative PM2.5 curves (FULL and CAP100) per outcome.\r\n")
    f.write("               * Pooled cumulative RR vs lag (overall, 0–7 days).\r\n")
    f.write("           - META/LAG0_PM25/ contains fixed-effect pooled lag-0\r\n")
    f.write("             concentration–response curves (0–100 µg/m³) across states.\r\n")
    f.write("           - META_summary_HOSP_MORT_PM25_plus5_FULL_and_CAP100.csv\r\n")
    f.write("             summarizes pooled effects at PM_ref+5 for FULL and CAP100.\r\n")
    f.write("\r\n")
    f.write("  TABLES/: Publication-ready tables for manuscripts.\r\n")
    f.write("           - Supplementary_Table1_sensitivity_wide.csv: wide-format\r\n")
    f.write("             sensitivity analysis table (rows = model specifications,\r\n")
    f.write("             columns = main outcomes with RR (95%CI) and p-values).\r\n")
    f.write("\r\n")
    f.write("  FOREST/: Forest plots and CSVs.\r\n")
    f.write("           - FOREST/HOSP/ and FOREST/MORT/ contain, for each outcome:\r\n")
    f.write("               * <outcome>_FOREST_cum0_7.*\r\n")
    f.write("                 State-level forest plots (one line per state + OVERALL)\r\n")
    f.write("                 with weights, normalized weights, p-values and p-values\r\n")
    f.write("                 for difference vs OVERALL.\r\n")
    f.write("               * FOREST_CLASSES_<HOSP|MORT>_<CIRC|RESP>_ALL_cum0_7.*\r\n")
    f.write("                 Overall pooled effects by class (Total, <60, ≥60, Male, Female).\r\n")
    f.write("               * FOREST_CLASSES_<HOSP|MORT>_<CIRC|RESP>_AGES_cum0_7.*\r\n")
    f.write("                 Age strata (Total, <60, ≥60).\r\n")
    f.write("               * FOREST_CLASSES_<HOSP|MORT>_<CIRC|RESP>_SEX_cum0_7.*\r\n")
    f.write("                 Sex strata (Total, Male, Female).\r\n")
    f.write("\r\n")
    f.write("All forest CSVs include:\r\n")
    f.write("  - logRelativeRisk, standardError, relativeRisk, lower95CI, upper95CI\r\n")
    f.write("  - weight (1/SE^2), weightNormalized (weight / sum(weight))\r\n")
    f.write("  - pValue (two-sided, logRR vs 0)\r\n")
    f.write("  - pValueDifference (comparison vs OVERALL or vs Total, depending on the forest).\r\n")

log("\n✅ DONE main outputs:")
log(f"  Root: {GASPAR_ROOT}")
log(f"  State plots+CSVs: {HOSP_OUTPUT_ROOT} and {MORT_OUTPUT_ROOT}")
log(f"  Meta pooled cumulative plots+CSVs: {META_OUTPUT_ROOT}")
log(f"  Forest plots (states + OVERALL + age/sex classes): {FOREST_OUTPUT_ROOT}")
log(f"  Pooled lag-0 PM curves: {LAG0_META_OUTPUT_ROOT}")
log(f"  Tables (publication-ready): {TABLES_OUTPUT_ROOT}")
log(f"  Guide file: {guide_path}")

# ============================================================
# STAGE 3 – SENSITIVITY GRID (múltiplos modelos)
#    - Vários modelos (df_time, df_lag, janela de lag, etc.)
#    - Meta-análise de efeito fixo entre estados
#    - Tabela "long" + tabela "wide" com uma linha por modelo
#      (RR, IC95% e p-values) para comparar robustez
# ============================================================

# Desfechos principais para a tabela de sensibilidade (EXPANDIDO)
SENS_OUTCOMES_HOSP = [
    "HOSP_RESP_TOTAL",
    "HOSP_RESP_LT60",
    "HOSP_RESP_>=60_ANOS",
    "HOSP_RESP_MASC",
    "HOSP_RESP_FEM",
    "HOSP_CIRC_TOTAL",
    "HOSP_CIRC_LT60",
    "HOSP_CIRC_>=60_ANOS",
    "HOSP_CIRC_MASC",
    "HOSP_CIRC_FEM",
]

SENS_OUTCOMES_MORT = [
    "MORT_RESP_TOTAL",
    "MORT_RESP_LT60",
    "MORT_RESP_>=60_ANOS",
    "MORT_RESP_MASC",
    "MORT_RESP_FEM",
    "MORT_CIRC_TOTAL",
    "MORT_CIRC_LT60",
    "MORT_CIRC_>=60_ANOS",
    "MORT_CIRC_MASC",
    "MORT_CIRC_FEM",
]

# ------------------------------------------------------------
# GRID DE MODELOS DE SENSIBILIDADE
#   - Você pode adicionar/remover linhas aqui conforme quiser
#   - lag_window_max: janela de lag usada para o resumo (0–lag_window_max)
#   - df_time_per_year: df/ano para spline de tendência temporal
#   - df_lag: df para spline de lag (quando arglag_mode = "df")
# ------------------------------------------------------------
SENS_MODEL_SPECS = [
    # Modelo "primário": igual ao do pipeline principal
    {
        "modelLabel": "Primary_dfTime7_lagWindow7_dfLag3_logknots",
        "lag_window_max": float(LAG_MAX_SHORT),  # 7
        "arglag_mode": "logknots",
        "df_lag": 3,
        "df_temp": 4,
        "df_rh": 4,
        "df_wind": 3,
        "df_precip": 3,
        "df_time_per_year": float(DF_TIME_PER_YEAR),  # 7
    },
    # Exemplo 2: menos df em tempo (mais suave)
    {
        "modelLabel": "dfTime5_lagWindow7_dfLag3_logknots",
        "lag_window_max": float(LAG_MAX_SHORT),  # 7
        "arglag_mode": "logknots",
        "df_lag": 3,
        "df_temp": 4,
        "df_rh": 4,
        "df_wind": 3,
        "df_precip": 3,
        "df_time_per_year": 5.0,
    },
    # Exemplo 3: janela de lag menor (0–5)
    {
        "modelLabel": "dfTime7_lagWindow5_dfLag3_logknots",
        "lag_window_max": 5.0,
        "arglag_mode": "logknots",
        "df_lag": 3,
        "df_temp": 4,
        "df_rh": 4,
        "df_wind": 3,
        "df_precip": 3,
        "df_time_per_year": float(DF_TIME_PER_YEAR),
    },
    # Exemplo 4: spline de lag mais flexível (df_lag = 4)
    {
        "modelLabel": "dfTime7_lagWindow7_dfLag4_logknots",
        "lag_window_max": float(LAG_MAX_SHORT),
        "arglag_mode": "df",
        "df_lag": 4,
        "df_temp": 4,
        "df_rh": 4,
        "df_wind": 3,
        "df_precip": 3,
        "df_time_per_year": float(DF_TIME_PER_YEAR),
    },
]


def fixed_effect_meta_from_logrr(log_rr_list, se_list):
    """Meta-análise de efeito fixo em logRR."""
    logs = np.asarray(log_rr_list, dtype=float)
    ses = np.asarray(se_list, dtype=float)
    mask = np.isfinite(logs) & np.isfinite(ses) & (ses > 0)
    logs = logs[mask]
    ses = ses[mask]
    if logs.size < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    w = 1.0 / (ses ** 2)
    sw = w.sum()
    mu = (w * logs).sum() / sw
    se_p = math.sqrt(1.0 / sw)
    rr = math.exp(mu)
    low = math.exp(mu - 1.96 * se_p)
    high = math.exp(mu + 1.96 * se_p)
    return mu, se_p, rr, low, high


def run_sensitivity_grid():
    """
    Roda a análise de sensibilidade para um GRID de modelos.
      - Para cada modelo em SENS_MODEL_SPECS
      - Para cada tipo de dado (HOSP/MORT)
      - Para cada desfecho da lista expandida
    Resultado:
      META/Sensitivity_GRID_pmRefPlus5.csv
      (uma linha por modelo × datasetType × outcome)
    """
    log("\n[SENS] Starting SENSITIVITY GRID (multiple model specs)...")

    rows = []

    for model in SENS_MODEL_SPECS:
        label = model["modelLabel"]
        lag_window_max = float(model["lag_window_max"])
        arglag_mode = model.get("arglag_mode", "logknots")
        df_lag = int(model.get("df_lag", 3))
        df_temp = int(model.get("df_temp", 4))
        df_rh = int(model.get("df_rh", 4))
        df_wind = int(model.get("df_wind", 3))
        df_precip = int(model.get("df_precip", 3))
        df_time_per_year = float(model.get("df_time_per_year", DF_TIME_PER_YEAR))

        log(
            f"[SENS] MODEL '{label}': lag_window_max={lag_window_max}, "
            f"arglag_mode={arglag_mode}, df_lag={df_lag}, "
            f"df_time/yr={df_time_per_year}, df_temp={df_temp}, "
            f"df_rh={df_rh}, df_wind={df_wind}, df_precip={df_precip}"
        )

        for dtype, files, sens_outcomes, pm_knots in [
            ("HOSP", hosp_files, SENS_OUTCOMES_HOSP, PM_KNOTS_HOSP),
            ("MORT", mort_files, SENS_OUTCOMES_MORT, PM_KNOTS_MORT),
        ]:
            if not files:
                continue

            ro.globalenv["PM_KNOTS"] = FloatVector(list(map(float, pm_knots)))
            ro.globalenv["var_pm_name"] = "WF_PM2.5_POND_STATE"

            for outcome in sens_outcomes:
                state_logrr = []
                state_se = []

                for data_path in files:
                    fname = os.path.basename(data_path)
                    if dtype == "HOSP":
                        state_name = fname.split("_HOSPITALIZATION")[0]
                    else:
                        state_name = fname.replace("_MORTALITY_FULL_DATA.xlsx", "")

                    try:
                        df_state = prepare_state_dataframe(data_path, dataset_type=dtype)
                    except Exception as e:
                        log(f"[SENS] Skipping {fname} for {dtype} {outcome} ({e})")
                        continue

                    with localconverter(ro.default_converter + pandas2ri.converter):
                        ro.globalenv["dados_sens"] = df_state.copy()

                    # parâmetros do modelo da grade
                    ro.globalenv["sens_outcome"] = outcome
                    ro.globalenv["sens_lag_window_max"] = float(lag_window_max)
                    ro.globalenv["sens_arglag_mode"] = arglag_mode
                    ro.globalenv["sens_df_lag"] = float(df_lag)
                    ro.globalenv["sens_df_temp"] = float(df_temp)
                    ro.globalenv["sens_df_rh"] = float(df_rh)
                    ro.globalenv["sens_df_wind"] = float(df_wind)
                    ro.globalenv["sens_df_precip"] = float(df_precip)
                    ro.globalenv["sens_df_time_per_year"] = float(df_time_per_year)

                    try:
                        res_vec = ro.r(
                            "run_dlnm_sensitivity(dados_sens, sens_outcome, "
                            "lag_max = LAG_MAX, "
                            "lag_window_max = sens_lag_window_max, "
                            "arglag_mode = sens_arglag_mode, "
                            "df_lag = sens_df_lag, "
                            "df_temp = sens_df_temp, "
                            "df_rh = sens_df_rh, "
                            "df_wind = sens_df_wind, "
                            "df_precip = sens_df_precip, "
                            "df_time_per_year = sens_df_time_per_year)"
                        )
                    except Exception as e:
                        log(f"[SENS] Error in R for {dtype} {state_name} {outcome} ({label}): {e}")
                        continue

                    res_np = np.array(res_vec, dtype=float)
                    if res_np.size != 2:
                        continue
                    log_rr, se = res_np[0], res_np[1]
                    if not (np.isfinite(log_rr) and np.isfinite(se) and se > 0):
                        continue

                    state_logrr.append(log_rr)
                    state_se.append(se)

                if not state_logrr:
                    log(f"[SENS] No valid states for {dtype} {outcome} ({label})")
                    rows.append(
                        {
                            "datasetType": dtype,
                            "outcome": outcome,
                            "modelLabel": label,
                            "lagWindowMax": lag_window_max,
                            "arglagMode": arglag_mode,
                            "df_lag": df_lag,
                            "df_timePerYear": df_time_per_year,
                            "df_temp": df_temp,
                            "df_rh": df_rh,
                            "df_wind": df_wind,
                            "df_precip": df_precip,
                            "RR": np.nan,
                            "RR_low": np.nan,
                            "RR_high": np.nan,
                            "pValue": np.nan,
                        }
                    )
                    continue

                mu, se_p, rr, low, high = fixed_effect_meta_from_logrr(
                    state_logrr, state_se
                )
                p_val = two_sided_pvalue(mu, se_p)

                rows.append(
                    {
                        "datasetType": dtype,
                        "outcome": outcome,
                        "modelLabel": label,
                        "lagWindowMax": lag_window_max,
                        "arglagMode": arglag_mode,
                            "df_lag": df_lag,
                            "df_timePerYear": df_time_per_year,
                            "df_temp": df_temp,
                            "df_rh": df_rh,
                            "df_wind": df_wind,
                            "df_precip": df_precip,
                            "RR": rr,
                            "RR_low": low,
                            "RR_high": high,
                            "pValue": p_val,
                        }
                    )

    if not rows:
        log("[SENS] No sensitivity results produced in GRID.")
        return

    df_sens = pd.DataFrame(rows)
    sens_path = os.path.join(META_OUTPUT_ROOT, "Sensitivity_GRID_pmRefPlus5.csv")
    df_sens.to_csv(sens_path, index=False, encoding="utf-8-sig")
    log(f"[SENS] SENSITIVITY GRID table saved: {sens_path}")


def build_sensitivity_wide():
    """
    Constrói tabela wide 'publication-ready' a partir de
    META/Sensitivity_GRID_pmRefPlus5.csv e salva em:
      TABLES/Supplementary_Table1_sensitivity_wide.csv (CSV)
      TABLES/Supplementary_Table1_sensitivity_wide.[pdf|png|svg]
    Cada linha = um modelo da grade (modelLabel),
    com colunas:
      - parâmetros do modelo (lagWindowMax, df_time, df_lag, etc.)
      - HOSP_/MORT_ + desfechos, no formato RR (95% CI) e p-value.
    """
    sens_src = os.path.join(META_OUTPUT_ROOT, "Sensitivity_GRID_pmRefPlus5.csv")
    if not os.path.exists(sens_src):
        log(f"[SENS_TABLE] Sensitivity GRID CSV not found: {sens_src}")
        return

    df = pd.read_csv(sens_src)
    if df.empty:
        log("[SENS_TABLE] Sensitivity GRID CSV is empty, nothing to build.")
        return

    # Ordena para estabilidade
    df = df.sort_values(["modelLabel", "datasetType", "outcome"])

    # Lista única de combinações datasetType/outcome (colunas de efeito)
    combos = (
        df[["datasetType", "outcome"]]
        .drop_duplicates()
        .sort_values(["datasetType", "outcome"])
        .to_records(index=False)
    )

    # Agrupa por modelo
    model_groups = df.groupby("modelLabel", sort=False)

    rows = []
    for model_label, subm in model_groups:
        # Pega parâmetros do modelo (primeira linha serve)
        first = subm.iloc[0]
        lag_window_max = first.get("lagWindowMax", np.nan)
        arglag_mode = first.get("arglagMode", "")
        df_lag = first.get("df_lag", np.nan)
        df_time_per_year = first.get("df_timePerYear", np.nan)
        df_temp = first.get("df_temp", np.nan)
        df_rh = first.get("df_rh", np.nan)
        df_wind = first.get("df_wind", np.nan)
        df_precip = first.get("df_precip", np.nan)

        row = {
            "Model": model_label,
            "LagWindowMax": int(lag_window_max) if np.isfinite(lag_window_max) else "",
            "ArglagMode": arglag_mode,
            "df_timePerYear": int(df_time_per_year) if np.isfinite(df_time_per_year) else "",
            "df_lag": int(df_lag) if np.isfinite(df_lag) else "",
            "df_temp": int(df_temp) if np.isfinite(df_temp) else "",
            "df_rh": int(df_rh) if np.isfinite(df_rh) else "",
            "df_wind": int(df_wind) if np.isfinite(df_wind) else "",
            "df_precip": int(df_precip) if np.isfinite(df_precip) else "",
        }

        for (dtype, out) in combos:
            sub = subm[(subm["datasetType"] == dtype) & (subm["outcome"] == out)]
            if sub.empty:
                rr_str = ""
                p_str = ""
            else:
                r = float(sub["RR"].iloc[0]) if pd.notna(sub["RR"].iloc[0]) else np.nan
                low = float(sub["RR_low"].iloc[0]) if pd.notna(sub["RR_low"].iloc[0]) else np.nan
                high = float(sub["RR_high"].iloc[0]) if pd.notna(sub["RR_high"].iloc[0]) else np.nan
                p = float(sub["pValue"].iloc[0]) if "pValue" in sub.columns and pd.notna(sub["pValue"].iloc[0]) else np.nan

                if np.isfinite(r) and np.isfinite(low) and np.isfinite(high):
                    rr_str = f"{r:.3f} ({low:.3f}–{high:.3f})"
                else:
                    rr_str = ""
                p_str = f"{p:.3g}" if np.isfinite(p) else ""

            var_id = f"{dtype}_{out}"
            rr_col = f"{var_id} RR (95% CI)"
            p_col = f"{var_id} p-value"
            row[rr_col] = rr_str
            row[p_col] = p_str

        rows.append(row)

    df_pub = pd.DataFrame(rows)

    out_csv = os.path.join(TABLES_OUTPUT_ROOT, "Supplementary_Table1_sensitivity_wide.csv")
    df_pub.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log(f"[SENS_TABLE] Publication-ready SENSITIVITY GRID table saved: {out_csv}")
    log("[SENS_TABLE] Preview:")
    log(df_pub.to_string(index=False))

    # Também salva a tabela como figura (PDF, PNG, SVG) para suplementar
    def export_sensitivity_table_figure(df_table: pd.DataFrame, base_path: str):
        """
        Exporta a tabela wide como figura (PDF, PNG, SVG) usando matplotlib.table.
        """
        n_rows, n_cols = df_table.shape
        # Tenta escalar o tamanho da figura conforme número de linhas/colunas
        fig_width = min(22, 2.0 + 0.6 * n_cols)
        fig_height = min(22, 1.0 + 0.4 * n_rows)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=df_table.values,
            colLabels=df_table.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.0, 1.3)

        fig.tight_layout()

        fig.savefig(base_path + ".pdf")
        fig.savefig(base_path + ".png", dpi=300)
        fig.savefig(base_path + ".svg")
        plt.close(fig)

    base_fig = os.path.join(TABLES_OUTPUT_ROOT, "Supplementary_Table1_sensitivity_wide")
    export_sensitivity_table_figure(df_pub, base_fig)
    log(
        f"[SENS_TABLE] Sensitivity wide table figure saved as: "
        f"{base_fig}.pdf / .png / .svg"
    )


# ------------------------------------------------------------
# Executa Stage 3 (GRID de modelos) e fecha o pipeline
# ------------------------------------------------------------

run_sensitivity_grid()
build_sensitivity_wide()

log("\n✅ DONE ALL STEPS (DLNM, meta-analysis, forests, sensitivity GRID).")
log(f"  Root: {GASPAR_ROOT}")
log(f"  State plots+CSVs: {HOSP_OUTPUT_ROOT} and {MORT_OUTPUT_ROOT}")
log(f"  Meta pooled cumulative plots+CSVs: {META_OUTPUT_ROOT}")
log(f"  Forest plots (states + OVERALL + age/sex classes): {FOREST_OUTPUT_ROOT}")
log(f"  Pooled lag-0 PM curves: {LAG0_META_OUTPUT_ROOT}")
log(f"  Tables (publication-ready): {TABLES_OUTPUT_ROOT}")
log(f"  Guide file: {guide_path}")

