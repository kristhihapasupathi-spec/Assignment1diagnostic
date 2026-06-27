# UAE Mental Health & EQ Survey — Diagnostic Analytics Dashboard

Interactive Streamlit dashboard performing a full diagnostic-analytics pass over the synthetic UAE
youth mental-health/EQ survey dataset (n=1200): noise-vs-signal detection, factor-impact mapping,
hypothesis validation, feature engineering, dynamic drill-down segmentation, regression, classification
and data mining, decision trees with explicit Gini/information-gain calculations, a CHAID-style
chi-square interaction detector, Random Forest vs. Gradient Boosted Trees, and Apriori association
rule mining — all in interactive Plotly visuals.

## Sections

1. **Overview & Descriptive Analysis** — KPIs, demographics, scale-variable distributions, top drivers, data-quality flags.
2. **Noise vs. Signal Diagnostics** — ANOVA + control-chart-style bands to separate random noise from statistically significant, systemic group differences.
3. **Factor Impact Mapping** — Pearson/Spearman correlation matrix and an interactive "driver network" diagram of significant relationships between operational factors.
4. **Hypothesis Validation** — four specific assumptions about underlying problems, each tested statistically (Spearman, Chi-square, Pearson, ANOVA) and explicitly supported/rejected.
5. **Feature Engineering** — derived features (Risk_Index, AI_Trust_Index, Stress_Tier, interaction terms) with formulas, rationale, and distributions.
6. **Dynamic Segmentation (Drill-Down)** — K-Means segmentation that dynamically splits the population into top-level segments, then lets you drill into any one segment and re-cluster it into granular sub-segments, visualized as a sunburst.
7. **Regression Analysis** — Linear Regression vs. Random Forest Regressor on any of the 5 Likert-scale outcomes, with leakage-safe feature sets, R²/MAE, and feature importances.
8. **Classification & Data Mining** — Logistic Regression vs. Random Forest on key business outcomes, confusion matrices, feature importances, and mined highest-propensity sub-groups.
9. **Decision Tree (Gini / Information Gain)** — a trained, visualized decision tree with a full node-by-node table of Gini impurity, entropy, and information gain for every split.
10. **CHAID** — a simplified 2-level Chi-Square Automatic Interaction Detector that recursively picks the most statistically significant categorical split at each node, shown as an interactive treemap.
11. **Random Forest vs. Gradient Boosted Trees** — side-by-side accuracy/F1, ROC curves, and importance comparison (impurity-based vs. permutation-based).
12. **Association Rules (Apriori)** — market-basket-style mining over the 50 binary multi-select survey items, with an interactive support/confidence/lift bubble chart.

All targets and parameters (k, depth, alpha, support, lift, etc.) are interactively selectable in the sidebar/section controls.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy via GitHub + Streamlit Community Cloud

1. Push this folder's contents to a new GitHub repository (`git init`, `git add .`, `git commit -m "Diagnostic dashboard"`, `git push`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repo/branch, and set the main file to `app.py`.
4. Deploy. The first build installs `requirements.txt` automatically.

## Data

`synthetic_survey_responses.csv` ships at the **repo root**, next to `app.py` (the loader also checks a
`data/` subfolder as a fallback). Keep the CSV at the root if you upload files individually through
GitHub's web UI, since that flow does not reliably preserve subfolders.

Columns prefixed with `_` (`_true_persona_segment`, `_is_noisy_responder`, `_is_outlier`) are synthetic
ground-truth/data-quality labels included for transparency and validation — they are not collected from
real respondents.

## Project structure

```
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── synthetic_survey_responses.csv
```
