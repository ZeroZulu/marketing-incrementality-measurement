# Marketing Incrementality Measurement

**Measuring true campaign lift when A/B tests aren't an option.**

Traditional marketing attribution is broken. Users who see ads are already more likely to convert — leading to inflated performance metrics and wasted budget. This project applies causal inference to cut through the noise and measure what actually works.

## The Problem

> *"Our campaign drove 50,000 conversions!"*
> *But how many would have converted anyway?*

Selection bias is everywhere. High-value users get targeted more. Engaged users see more ads. Without proper methodology, you're measuring correlation, not causation.

## The Solution

Three complementary approaches, triangulated for confidence:

| Method | What It Does |
|--------|--------------|
| **Difference-in-Differences** | Compares treatment vs control regions over time |
| **Propensity Score Matching** | Creates synthetic control groups from observational data |
| **Synthetic Control** | Builds counterfactual outcomes from weighted donor units |

When all three converge on the same estimate, you can trust the result.

## Key Finding

```
Naive Attribution:  24.5% lift  ← What marketing reported
Causal Estimate:    11.2% lift  ← What actually happened
Bias:               54% overstated
```

## Quick Start

```bash
# Run analysis
python run_analysis.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
├── run_analysis.py      # Entry point
├── src/models/          # DiD, PSM, Synthetic Control implementations
├── dashboard/app.py     # Interactive Streamlit dashboard
└── output/              # JSON results for visualization
```

## Dashboard

Interactive exploration of results — parallel trends validation, covariate balance plots, donor weights, and a business impact calculator.

Built with Streamlit + Plotly. Dark theme.

## Tech Stack

`Python` `Pandas` `NumPy` `SciPy` `Scikit-learn` `Streamlit` `Plotly`

## Why This Matters

Every dollar spent on marketing that *doesn't* drive incremental value is a dollar wasted. Causal inference turns gut feelings into evidence — enabling smarter budget allocation and honest performance measurement.

---

*Built by [Shril](https://github.com/ZeroZulu)*
