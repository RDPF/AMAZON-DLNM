# AMAZON-DLNM

**AMAZON-DLNM** is an episode-based analytical framework designed to quantify
short-term health effects of extreme wildfire-related air pollution in the
Brazilian Amazon using **Distributed Lag Non-Linear Models (DLNM)**.

The project integrates state-level time-series analysis, meteorological
adjustment, and meta-analysis to produce robust estimates of lagged and
cumulative health risks associated with extreme exposure episodes.

---

## Project Scope

AMAZON-DLNM focuses on:

- Extreme exposure episodes (e.g., top 5% PM2.5 or other climate-related shocks)
- Short-term lag structures (typically 0–7 or 0–15 days)
- Health outcomes including respiratory and circulatory hospitalizations and
  mortality
- Reproducible epidemiological modeling using Python and R

The framework is methodologically transferable to other regions and
climate-related extreme events.

---

## Methodological Overview

The pipeline implements:

- Episode-based exposure definitions (peak-day centered)
- Distributed Lag Non-Linear Models (DLNM)
- Quasi-Poisson generalized linear models for overdispersed count data
- Control for long-term trends, seasonality, and day-of-week effects
- Adjustment for meteorological confounders
- State-level estimation followed by random-effects meta-analysis
- Generation of lag–response curves, cumulative risk estimates, and forest plots

The DLNM methodology follows the general framework proposed by  
**Gasparrini et al.**, with original episode-based extensions developed within
AMAZON-DLNM.

---

## Software and Dependencies

Core technologies:

- **Python** (data processing, orchestration, visualization)
- **R** (DLNM modeling and meta-analysis)
- R packages: `dlnm`, `mvmeta`, `splines`
- Python–R interface via `rpy2`

> ⚠️ This repository contains **only source code and documentation**.

---

## License and Data Use

### License (Code Only)

The source code of **AMAZON-DLNM** is released under the  
**BSD 3-Clause License**.

See the `LICENSE` file for full terms.

### Data Use

Datasets used in this project are **not distributed** and are **not covered**
by the BSD 3-Clause License.

All data remain the property of their original owners and are subject to
respective data-use agreements, ethical approvals, and legal restrictions.

Researchers interested in accessing the data must contact the original
data providers.

---

## Intellectual Property

AMAZON-DLNM is an **original analytical pipeline** developed by the authors.
The project name, code structure, and episode-based modeling strategy
constitute original intellectual contributions.

Use of the DLNM methodology does **not** imply transfer of intellectual
property from existing software packages.

---

## Citation

If you use AMAZON-DLNM in academic work, please cite:

1. Gasparrini A. (2011).  
   *Distributed lag linear and non-linear models in R: the package dlnm.*  
   Journal of Statistical Software.

2. The corresponding **AMAZON-DLNM** project publication (when available).

---

## Disclaimer

This software is provided **“as is”**, without warranty of any kind, express
or implied. The authors assume no responsibility for the use or misuse of
the code or for interpretations derived from its outputs.

---

## Contact

**Project:** AMAZON-DLNM  
**Institution:** [FURG e Unifesp]  
**Contact:** [Renato Dutra Pereira Filho / renatodutrapereira@gmail.com ]
             [Ronan Adler Tavella  / ronanadlertavella@gmail.com ]

---
