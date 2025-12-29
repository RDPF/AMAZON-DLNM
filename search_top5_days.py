#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# SIMPLE PM2.5 EPISODE FINDER — TOP 5% DAYS + 15-DAY WINDOWS
#
# Ideia:
#   - Para cada estado, montar série de PM2.5 (HOSP + MORT)
#   - Usar apenas PM_CANON = "WF_PM2.5_POND_STATE"
#   - Considerar todos os dias com PM válida (finita)
#   - Definir K = ceil(5% de N_valid)
#   - Ordenar por PM decrescente e pegar os K dias de maior PM
#   - Para cada um dos K dias (peak_day):
#       * tentar construir uma janela de 15 dias consecutivos:
#             [peak_day, peak_day+14]
#         exigindo que TODOS esses dias existam e tenham PM finita
#       * se conseguir, registrar um episódio:
#             episode_start = peak_day
#             episode_end   = peak_day + 14
#             duration_days = 15
#             peak_day      = peak_day
#             peak_pm       = PM(peak_day)
#             area_pm       = soma de PM nos 15 dias
#             method        = "TOP5PM_STATE"
#
#   - Após juntar TODOS episódios de TODOS estados:
#       * calcular percentil 95 de area_pm
#       * marcar is_global_5pct = True se area_pm >= p95_global
#
# Saídas (em C:\busca5):
#   - episodes_simple.csv
#   - episodes_compare_summary.csv
#   - report_top5pm.pdf (1 página por estado)
# ============================================================

import os
os.environ["MPLBACKEND"] = "Agg"

import glob
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")
print("Matplotlib backend:", matplotlib.get_backend())

# ---------------- CONFIG ----------------
HOSP_BASE_DIR = (
    r"C:\dados3\DADOS - HOSP_ENV_LEGAL_AMAZON_DATA-20251201T163911Z-1-001"
    r"\DADOS - HOSP_ENV_LEGAL_AMAZON_DATA\Full"
)
MORT_BASE_DIR = r"C:\dados2\MORT"

OUT_DIR = r"C:\busca5"
os.makedirs(OUT_DIR, exist_ok=True)

DATE_COL = "DATA"
PM_PREFIX = "WF_PM2.5_POND_"
PM_CANON = "WF_PM2.5_POND_STATE"

# tamanho da janela em dias (peak_day + 0..14 => 15 dias)
WINDOW_DAYS = 15

# fração de dias com maior PM usados como candidatos de pico
TOP_FRAC = 0.05  # 5%

# estilo de gráficos
LINE_BLUE = "#0072B2"
SHADE_BLUEGRAY = "#A6BDD7"
SHADE_ALPHA = 0.30

# ---------------- HELPERS ----------------
def list_excel_files(base_dir: str, pattern: str):
    return sorted(glob.glob(os.path.join(base_dir, pattern)))

def state_name_from_hosp(fp: str) -> str:
    return os.path.basename(fp).split("_HOSPITALIZATION")[0].strip().upper()

def state_name_from_mort(fp: str) -> str:
    return os.path.basename(fp).replace("_MORTALITY_FULL_DATA.xlsx", "").strip().upper()

def read_pm(path: str, source: str) -> pd.DataFrame:
    """
    Lê o arquivo Excel, encontra a coluna de PM que começa com PM_PREFIX,
    renomeia para PM_CANON, e retorna DATA, PM_CANON, source.
    """
    df = pd.read_excel(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    pm_cols = [c for c in df.columns if str(c).startswith(PM_PREFIX)]
    if not pm_cols:
        raise ValueError(f"No PM column starting with {PM_PREFIX} in {os.path.basename(path)}")

    df = df.rename(columns={pm_cols[0]: PM_CANON})
    df[PM_CANON] = pd.to_numeric(df[PM_CANON], errors="coerce")

    out = df[[DATE_COL, PM_CANON]].copy()
    out["source"] = source
    return out

def state_stats(pm: np.ndarray):
    """
    Estatísticas básicas por estado: mean, sd, median, p95, p99, min, max.
    """
    pm = np.asarray(pm, dtype=float)
    pm = pm[np.isfinite(pm)]
    if pm.size < 10:
        return None
    mu = float(np.mean(pm))
    sd = float(np.std(pm, ddof=1)) if pm.size > 1 else float("nan")
    med = float(np.median(pm))
    p95 = float(np.quantile(pm, 0.95))
    p99 = float(np.quantile(pm, 0.99))
    pmin = float(np.min(pm))
    pmax = float(np.max(pm))
    return {
        "mu": mu,
        "sd": sd,
        "median": med,
        "p95": p95,
        "p99": p99,
        "min": pmin,
        "max": pmax,
    }

def find_top5_windows(df_pm: pd.DataFrame, window_days: int = WINDOW_DAYS, top_frac: float = TOP_FRAC):
    """
    Para um df_pm de um estado (DATA, PM_CANON, source):
      - filtra PM finita
      - escolhe K = ceil(top_frac * N_valid) dias de maior PM
      - para cada dia candidato, tenta construir janela [t, t+window_days-1]
        (com todos os dias presentes e PM finita)
      - retorna lista de episódios com:
          episode_start, episode_end, duration_days,
          peak_day, peak_pm, area_pm
      - sobreposição de janelas é permitida
    """
    d = df_pm.dropna(subset=[PM_CANON]).copy()
    d = d.sort_values(DATE_COL).reset_index(drop=True)
    if d.empty:
        return [], 0, 0  # episodes, N_valid, N_candidates

    # index pela data (normalizada para dia)
    d[DATE_COL] = d[DATE_COL].dt.normalize()
    d = d.drop_duplicates(subset=[DATE_COL], keep="first").reset_index(drop=True)
    d = d.sort_values(DATE_COL).reset_index(drop=True)

    N_valid = d.shape[0]
    if N_valid < window_days + 5:
        # muito poucos dados para janelas de 15 dias
        return [], N_valid, 0

    # ordenar por PM decrescente
    d_sorted = d.sort_values(PM_CANON, ascending=False).reset_index(drop=True)

    K = max(1, int(math.ceil(top_frac * N_valid)))
    K = min(K, N_valid)
    top_candidates = d_sorted.iloc[:K].copy()

    # reindexar por data para facilitar a checagem de janelas
    d_idx = d.set_index(DATE_COL)

    episodes = []
    n_candidates = top_candidates.shape[0]

    for _, row in top_candidates.iterrows():
        peak_day = row[DATE_COL]
        peak_pm = float(row[PM_CANON])

        # construir datas da janela
        window_dates = pd.date_range(start=peak_day, periods=window_days, freq="D")

        # verificar se TODOS os dias estão presentes em d_idx
        try:
            sub = d_idx.loc[window_dates]
        except KeyError:
            # algum dia da janela não existe na série => descarta esse candidato
            continue

        if sub.shape[0] != window_days:
            # por segurança, se faltar linha
            continue

        # exigir PM finita nos 15 dias
        if not np.isfinite(sub[PM_CANON].to_numpy(dtype=float)).all():
            continue

        area_pm = float(sub[PM_CANON].sum())
        episode_start = window_dates[0]
        episode_end = window_dates[-1]
        duration_days = window_days

        episodes.append({
            "episode_start": episode_start.strftime("%Y-%m-%d"),
            "episode_end": episode_end.strftime("%Y-%m-%d"),
            "duration_days": int(duration_days),
            "peak_day": peak_day.strftime("%Y-%m-%d"),
            "peak_pm": peak_pm,
            "area_pm": area_pm,
        })

    return episodes, N_valid, n_candidates

# ---------------- LOAD FILES ----------------
hosp_files = list_excel_files(HOSP_BASE_DIR, "*_HOSPITALIZATION_FULL_DATA*.xlsx")
mort_files = list_excel_files(MORT_BASE_DIR, "*_MORTALITY_FULL_DATA.xlsx")

hosp_map = {state_name_from_hosp(fp): fp for fp in hosp_files}
mort_map = {state_name_from_mort(fp): fp for fp in mort_files}
states = sorted(set(hosp_map.keys()) | set(mort_map.keys()))

print("States:", states)

# ---------------- MAIN ----------------
episodes_rows = []
compare_rows = []

pdf_path = os.path.join(OUT_DIR, "report_top5pm.pdf")
with PdfPages(pdf_path) as pdf:
    # cover page
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.axis("off")
    ax.text(0.01, 0.95, "PM2.5 wildfire episode screening — TOP 5% PEAK DAYS",
            fontsize=18, fontweight="bold", va="top")
    ax.text(0.01, 0.88, "Episode definition (per state):", fontsize=12)
    ax.text(0.02, 0.84,
            "- Let N_valid = number of days with valid PM2.5.",
            fontsize=11)
    ax.text(0.02, 0.80,
            "- Define K = ceil(0.05 × N_valid): top 5% days with highest PM2.5.",
            fontsize=11)
    ax.text(0.02, 0.76,
            "- For each of these K peak days, attempt to build a 15-day window:",
            fontsize=11)
    ax.text(0.03, 0.72,
            "  [peak_day, peak_day + 14], requiring all 15 days to exist with finite PM2.5.",
            fontsize=11)
    ax.text(0.02, 0.68,
            "- Each successful window defines one episode:",
            fontsize=11)
    ax.text(0.03, 0.64,
            "  episode_start = peak_day, episode_end = peak_day+14, duration=15,",
            fontsize=11)
    ax.text(0.03, 0.60,
            "  peak_pm = PM(peak_day), area_pm = sum(PM over the 15 days).",
            fontsize=11)
    ax.text(0.01, 0.54,
            "Global 5% flag: episodes whose area_pm is ≥ 95th percentile across all states (is_global_5pct = TRUE).",
            fontsize=12)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # --------- loop por estado ---------
    for state in states:
        df_list = []
        # prefer MORT se ambos existirem (mantém compatibilidade com o script anterior)
        if state in mort_map:
            try:
                df_list.append(read_pm(mort_map[state], "MORT"))
            except Exception as e:
                print(f"[{state}] MORT read failed:", e)
        if state in hosp_map:
            try:
                df_list.append(read_pm(hosp_map[state], "HOSP"))
            except Exception as e:
                print(f"[{state}] HOSP read failed:", e)

        if not df_list:
            continue

        # juntar HOSP+MORT, preferindo MORT em datas duplicadas
        df_pm = pd.concat(df_list, ignore_index=True).copy()
        df_pm["rank"] = df_pm["source"].map({"MORT": 0, "HOSP": 1}).fillna(9).astype(int)
        df_pm = (
            df_pm.sort_values([DATE_COL, "rank"])
                 .drop_duplicates(subset=[DATE_COL], keep="first")
        )
        df_pm = df_pm[[DATE_COL, PM_CANON, "source"]].sort_values(DATE_COL).reset_index(drop=True)

        st_stats = state_stats(df_pm[PM_CANON].to_numpy(dtype=float))
        if st_stats is None:
            print(f"[{state}] Not enough PM data, skipping.")
            continue

        episodes_state, n_valid, n_candidates = find_top5_windows(
            df_pm,
            window_days=WINDOW_DAYS,
            top_frac=TOP_FRAC
        )

        # registrar episódios
        for local_id, r in enumerate(episodes_state, start=1):
            row = {
                "state": state,
                "method": "TOP5PM_STATE",
                "episode_id_state": local_id,
                "episode_start": r["episode_start"],
                "episode_end": r["episode_end"],
                "duration_days": r["duration_days"],
                "peak_day": r["peak_day"],
                "peak_pm": r["peak_pm"],
                "area_pm": r["area_pm"],
                # estatísticas de PM para referência
                "pm_mean": st_stats["mu"],
                "pm_sd": st_stats["sd"],
                "pm_median": st_stats["median"],
                "pm_p95": st_stats["p95"],
                "pm_p99": st_stats["p99"],
                "pm_min": st_stats["min"],
                "pm_max": st_stats["max"],
                "n_valid_days": n_valid,
                "n_top5_candidates": n_candidates,
            }
            episodes_rows.append(row)

        # resumo por estado
        compare_rows.append({
            "state": state,
            "n_valid_days": n_valid,
            "n_top5_candidates": n_candidates,
            "n_episodes_15d": len(episodes_state),
            "pm_mean": st_stats["mu"],
            "pm_sd": st_stats["sd"],
            "pm_median": st_stats["median"],
            "pm_p95": st_stats["p95"],
            "pm_p99": st_stats["p99"],
            "pm_min": st_stats["min"],
            "pm_max": st_stats["max"],
        })

        # ---- página PDF por estado ----
        d = df_pm.dropna(subset=[PM_CANON]).copy()
        if d.empty:
            continue

        d = d.sort_values(DATE_COL)
        fig, ax = plt.subplots(figsize=(12.5, 3.5))

        ax.plot(d[DATE_COL], d[PM_CANON], color=LINE_BLUE, lw=1.0, label="PM2.5")

        # sombrear episódios (janelas de 15 dias)
        for r in episodes_state:
            s = pd.to_datetime(r["episode_start"])
            e = pd.to_datetime(r["episode_end"])
            ax.axvspan(s, e, color=SHADE_BLUEGRAY, alpha=SHADE_ALPHA, lw=0)

        ax.set_title("")  # sem título
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_xlabel("Date")
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.text(
            0.01, 0.97,
            f"{state} | TOP 5% peak days, 15-day windows",
            transform=ax.transAxes, va="top", fontsize=11, fontweight="bold"
        )
        ax.text(
            0.01, 0.90,
            f"Valid days = {n_valid} | top5% candidates = {n_candidates} | episodes (15d) = {len(episodes_state)}",
            transform=ax.transAxes, va="top", fontsize=9
        )
        ax.text(
            0.01, 0.84,
            f"PM median = {st_stats['median']:.2f} | P95 = {st_stats['p95']:.2f} | P99 = {st_stats['p99']:.2f} | max = {st_stats['max']:.2f}",
            transform=ax.transAxes, va="top", fontsize=9
        )

        ax.legend(loc="upper right", fontsize=8)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

# ---- CONSTRUIR DATAFRAMES FINAIS ----
df_eps = pd.DataFrame(episodes_rows)
df_cmp = pd.DataFrame(compare_rows)

if not df_eps.empty:
    # Colunas auxiliares para compatibilidade com pipeline DLNM:
    df_eps["STATE"] = df_eps["state"].astype(str).str.upper()
    df_eps["PEAK_DATE"] = pd.to_datetime(df_eps["peak_day"], errors="coerce")

    # EPISODE_ID global (único no arquivo todo)
    df_eps = df_eps.sort_values(
        ["method", "STATE", "episode_start", "episode_end"]
    ).reset_index(drop=True)
    df_eps["EPISODE_ID"] = np.arange(1, df_eps.shape[0] + 1).astype(int)

    # Flag GLOBAL 5% em area_pm (top 5% mais intensos)
    df_eps["is_global_5pct"] = False
    if df_eps["area_pm"].notna().sum() > 0:
        thr_global = df_eps["area_pm"].quantile(0.95)
        df_eps.loc[
            df_eps["area_pm"] >= thr_global,
            "is_global_5pct"
        ] = True

# ---- SALVAR CSVs ----
episodes_csv = os.path.join(OUT_DIR, "episodes_simple.csv")
compare_csv = os.path.join(OUT_DIR, "episodes_compare_summary.csv")

df_eps.to_csv(episodes_csv, index=False, encoding="utf-8-sig")
df_cmp.to_csv(compare_csv, index=False, encoding="utf-8-sig")

print("\nDONE.")
print("Saved:", episodes_csv)
print("Saved:", compare_csv)
print("Saved:", pdf_path)

