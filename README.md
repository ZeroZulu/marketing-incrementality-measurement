# 📊 Marketing Incrementality Measurement

**Measuring true campaign lift when A/B tests aren't an option.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 The Problem

> *"Our campaign drove 50,000 conversions!"*  
> *But how many would have converted anyway?*

Traditional marketing attribution is broken. Users who see ads are already more likely to convert — leading to inflated metrics and wasted budget. This project applies causal inference to cut through the noise and measure what actually works.

---

## 🔬 The Solution

Three complementary approaches, triangulated for confidence:

| Method | What It Does |
|--------|--------------|
| **Difference-in-Differences** | Compares treatment vs control regions over time |
| **Propensity Score Matching** | Creates synthetic control groups from observational data |
| **Synthetic Control** | Builds counterfactual outcomes from weighted donor units |

When all three converge on the same estimate, you can trust the result.

---

## 📊 View the Analysis

### 🚀 Interactive Dashboard
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://marketing-incrementality-measurement-px9sfmsau9hqvnxo8hzhu8.streamlit.app/)

Explore the results interactively with filters for regions, time periods, and methods.

### 📓 Jupyter Notebook
[![nbviewer](https://img.shields.io/badge/View%20on-nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/ZeroZulu/marketing-incrementality-measurement/blob/main/final-project/notebooks/marketing_incrementality_analysis.ipynb)

Full analysis with code, visualizations, and methodology explanations.

---

## 📈 Key Finding

```
Naive Attribution:  24.5% lift  ← What marketing reported
Causal Estimate:    11.2% lift  ← What actually happened
Bias:               54% overstated
```

---

## 🚀 Quick Start

```bash
# Run analysis
python run_analysis.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
├── run_analysis.py      # Entry point
├── src/models/          # DiD, PSM, Synthetic Control implementations
├── dashboard/app.py     # Interactive Streamlit dashboard
└── output/              # JSON results for visualization
```

---

## 🎨 Dashboard

Interactive exploration of results — parallel trends validation, covariate balance plots, donor weights, and a business impact calculator.

Built with `Streamlit` + `Plotly`. Dark theme.

---

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `SciPy` `Scikit-learn` `Streamlit` `Plotly`

---

## 💡 Why This Matters

Every dollar spent on marketing that *doesn't* drive incremental value is a dollar wasted. Causal inference turns gut feelings into evidence — enabling smarter budget allocation and honest performance measurement.

---

*Built by [Shril](https://github.com/ZeroZulu)*

