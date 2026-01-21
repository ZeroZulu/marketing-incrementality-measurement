# 📊 Marketing Incrementality & Lift Measurement

A comprehensive causal inference toolkit for measuring true marketing campaign effectiveness using three statistical methods: **Difference-in-Differences (DiD)**, **Propensity Score Matching (PSM)**, and **Synthetic Control**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

Marketing teams often struggle to measure the true incremental impact of their campaigns. Traditional attribution overstates effectiveness due to **selection bias** - users who see ads are already more likely to convert.

This project implements three gold-standard causal inference methods to estimate the **true causal lift** of marketing campaigns:

| Method | Use Case | Key Assumption |
|--------|----------|----------------|
| **DiD** | Regional/time-based experiments | Parallel trends |
| **PSM** | User-level observational data | Conditional independence |
| **Synthetic Control** | Single treated unit (state/region) | Convex combination of donors |

## 📁 Project Structure

```
marketing-incrementality-measurement/
├── run_analysis.py          # Main entry point - runs full pipeline
├── config/
│   └── config.yaml          # Configuration (data source, model params)
├── src/
│   ├── pipeline.py          # Orchestrates the analysis
│   ├── export.py            # Exports results to JSON
│   ├── data/
│   │   └── data_generator.py    # Synthetic data with known effects
│   └── models/
│       ├── did.py           # Difference-in-Differences
│       ├── psm.py           # Propensity Score Matching
│       └── synthetic_control.py  # Synthetic Control Method
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── output/                  # JSON results (auto-generated)
└── tests/                   # Unit tests
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/ZeroZulu/marketing-incrementality-measurement.git
cd marketing-incrementality-measurement

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Quick Start

### 1. Run the Analysis

```bash
python run_analysis.py
```

This will:
- Generate synthetic data with a **known 12% true effect**
- Run all three causal inference methods
- Export results to `output/*.json`

### 2. Launch the Dashboard

```bash
pip install streamlit plotly
streamlit run dashboard/app.py
```

Open http://localhost:8501 to view the interactive dashboard.

## 🔬 Methodology Deep Dive

### Difference-in-Differences (DiD)

Compares treatment vs control groups over time:

```
DiD Effect = (Y_treatment_post - Y_treatment_pre) - (Y_control_post - Y_control_pre)
```

**Key validation**: Parallel trends assumption - groups must follow similar trajectories pre-treatment.

### Propensity Score Matching (PSM)

Matches treated users to similar control users based on observable characteristics:

1. Estimate propensity scores via logistic regression
2. Match treated → control using nearest-neighbor
3. Compare outcomes on matched samples

**Key output**: Average Treatment Effect on Treated (ATT)

### Synthetic Control

Constructs a "synthetic" version of the treated unit from weighted donors:

```
Synthetic California = 35% Texas + 28% Florida + 22% Illinois + ...
```

Weights are optimized to minimize pre-treatment prediction error.

## 📈 Sample Results

| Method | Lift Estimate | 95% CI | p-value |
|--------|--------------|--------|---------|
| DiD | 12.3% | [5.9%, 18.7%] | < 0.001 |
| PSM | 11.2% | [5.7%, 16.6%] | < 0.001 |
| Synthetic Control | 9.8% | [4.2%, 15.4%] | 0.024 |
| **Triangulated Average** | **11.1%** | — | — |

*True effect in synthetic data: 12.0%*

## 🎨 Dashboard Features

- **Overview**: KPIs, method comparison, key findings
- **DiD Analysis**: Parallel trends visualization
- **PSM Analysis**: Covariate balance love plot, bias detection
- **Synthetic Control**: Actual vs synthetic comparison, donor weights
- **Business Impact**: Interactive calculator (spend → iROAS)

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📚 Key Concepts

### Selection Bias

Users exposed to ads are often **already more likely to convert** (higher engagement, more active, etc.). Naive comparison overstates lift.

**Example from this analysis:**
- Naive estimate: 24.5% lift
- Causal estimate (PSM): 11.2% lift
- **Bias reduction: 54%**

### Triangulation

Using multiple methods provides robustness. If DiD, PSM, and SC all converge on ~11-12% lift, we have high confidence in the estimate.

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  source: synthetic  # or 'bigquery'
  
treatment:
  true_effect: 0.12  # 12% true lift (for validation)
  confounding_strength: 0.3

models:
  did:
    cluster_robust_se: true
  psm:
    matching_method: nearest
    caliper: 0.1
```

## 🚀 Deploy to Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub
4. Set main file: `dashboard/app.py`
5. Deploy!

## 📖 References

- Abadie, A., & Gardeazabal, J. (2003). *The Economic Costs of Conflict: A Case Study of the Basque Country*
- Rosenbaum, P. R., & Rubin, D. B. (1983). *The Central Role of the Propensity Score*
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*

## 📄 License

MIT License - feel free to use for your own projects!

## 👤 Author

**Shril** - [GitHub](https://github.com/ZeroZulu)

---

*Built for marketing data science portfolio • Causal inference • Incrementality measurement*
