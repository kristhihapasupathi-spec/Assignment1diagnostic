from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, r2_score, mean_absolute_error,
                              confusion_matrix, silhouette_score, adjusted_rand_score,
                              roc_curve, auc)
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="UAE Mental Health Diagnostic Analytics", layout="wide")

# ============================================================ CONSTANTS
SCALE_NUM = ["B2_stress_severity_1to10", "C1_stigma_comfort_1to5", "D3_recommend_likelihood_1to5",
             "E1_ai_comfort_1to5", "E4_ai_rec_likelihood_1to5"]
SCALE_LABELS = {
    "B2_stress_severity_1to10": "Stress severity (1-10)",
    "C1_stigma_comfort_1to5": "Comfort discussing MH (1-5)",
    "D3_recommend_likelihood_1to5": "Would recommend app (1-5)",
    "E1_ai_comfort_1to5": "Comfort with AI (1-5)",
    "E4_ai_rec_likelihood_1to5": "Likely to try AI advice (1-5)",
}
KEY_CAT = ["A1_age_group", "A4_occupation_status", "B1_stress_frequency", "B4_coping_ability",
           "C2_stigma_judged", "D1_interest_in_tool", "E5_trust_preference"]
CHAID_CATS = ["A1_age_group", "A2_residency_status", "A3_emirate", "A4_occupation_status", "A5_gender",
              "B1_stress_frequency", "B4_coping_ability", "C2_stigma_judged", "E5_trust_preference",
              "F4_language_preference", "Stress_Tier"]
MULTI_PREFIXES = ["B3", "B5", "C3", "D2", "E2", "E3", "F3"]
MULTI_Q_TITLES = {"B3": "Top stress causes", "B5": "Who they turn to", "C3": "Barriers to seeking help",
                   "D2": "Most valued support types", "E2": "Biggest AI concerns",
                   "E3": "What increases AI trust", "F3": "Preferred access channel"}
CLASS_TARGETS = ["F2_would_pay", "F1_would_use", "D1_interest_in_tool"]
RISK_INDEX_COMPONENTS = ["B2_stress_severity_1to10", "C1_stigma_comfort_1to5"]
AI_TRUST_COMPONENTS = ["E1_ai_comfort_1to5", "E4_ai_rec_likelihood_1to5"]

AGE_MID = {"12-14": 13, "15-17": 16, "18-20": 19, "21-24": 22.5, "25-27": 26}
COPING_MAP = {"No, not really": 1, "Not sure": 2, "Somewhat": 3, "Yes, definitely": 4}
JUDGED_MAP = {"Not at all": 1, "Not really": 2, "Somewhat": 3, "Yes, a lot": 4}
INTEREST_MAP = {"Definitely not": 1, "Probably not": 2, "Not sure": 3, "Probably": 4, "Definitely": 5}


def cramers_v(chi2, n, r, k):
    return float(np.sqrt((chi2 / n) / (min(r - 1, k - 1) or 1)))


def chi_square_assoc(df, predictor, target):
    ct = pd.crosstab(df[predictor], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = cramers_v(chi2, ct.values.sum(), ct.shape[0], ct.shape[1])
    return chi2, p, dof, v, ct


_HERE = Path(__file__).resolve().parent
_CSV_CANDIDATES = [
    _HERE / "synthetic_survey_responses.csv",
    _HERE / "data" / "synthetic_survey_responses.csv",
]


@st.cache_data
def load_data():
    for p in _CSV_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        st.error("Could not find `synthetic_survey_responses.csv` (repo root or `data/` folder).")
        st.stop()

    # ---- feature engineering ----
    df["Age_numeric"] = df["A1_age_group"].map(AGE_MID)
    df["Coping_numeric"] = df["B4_coping_ability"].map(COPING_MAP)
    df["Stigma_judged_numeric"] = df["C2_stigma_judged"].map(JUDGED_MAP)
    df["Interest_numeric"] = df["D1_interest_in_tool"].map(INTEREST_MAP)
    df["Stress_Tier"] = pd.cut(df["B2_stress_severity_1to10"], bins=[0, 3, 6, 10],
                                labels=["Low", "Medium", "High"])

    def z(s):
        return (s - s.mean()) / s.std(ddof=0)

    df["Risk_Index"] = (z(df["B2_stress_severity_1to10"]) + z(6 - df["C1_stigma_comfort_1to5"])
                         + z(df["Stigma_judged_numeric"]) + z(5 - df["Coping_numeric"])) / 4
    df["AI_Trust_Index"] = (z(df["E1_ai_comfort_1to5"]) + z(df["E4_ai_rec_likelihood_1to5"])) / 2
    df["Stress_x_LowCoping"] = df["B2_stress_severity_1to10"] * (5 - df["Coping_numeric"])
    return df


df = load_data()
multi_cols = [c for c in df.columns if any(c.startswith(p + "_") for p in MULTI_PREFIXES)]
ENGINEERED_NUM = SCALE_NUM + ["Age_numeric", "Coping_numeric", "Stigma_judged_numeric",
                               "Risk_Index", "AI_Trust_Index"]

st.title("UAE Mental Health & EQ Survey — Diagnostic Analytics")
st.caption(f"Synthetic respondent dataset (n={len(df):,}), ages 12-27 — diagnostic deep-dive beyond descriptive reporting.")

SECTIONS = [
    "1. Overview & Descriptive Analysis",
    "2. Noise vs. Signal Diagnostics",
    "3. Factor Impact Mapping",
    "4. Hypothesis Validation",
    "5. Feature Engineering",
    "6. Dynamic Segmentation (Drill-Down)",
    "7. Regression Analysis",
    "8. Classification & Data Mining",
    "9. Decision Tree (Gini / Information Gain)",
    "10. CHAID (Chi-Square Interaction Detector)",
    "11. Random Forest vs Gradient Boosting",
    "12. Association Rules (Apriori)",
]
section = st.sidebar.radio("Diagnostic section", SECTIONS)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset is synthetic: 5 latent personas, ~8.7% inconsistent ('noisy') responders, "
    "~2.3% seeded outliers, realistic skewed segment mix."
)

# ============================================================ SECTION 1
if section.startswith("1."):
    st.header("Overview & Descriptive Analysis")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Respondents", f"{len(df):,}")
    c2.metric("Avg stress (1-10)", f"{df['B2_stress_severity_1to10'].mean():.1f}")
    c3.metric("Feel judged (a lot/somewhat)", f"{df['C2_stigma_judged'].isin(['Yes, a lot','Somewhat']).mean()*100:.0f}%")
    c4.metric("Would use app", f"{(df['F1_would_use']=='Yes, definitely').mean()*100:.0f}%")
    c5.metric("Willing to pay outright", f"{(df['F2_would_pay']=='Yes, willing to pay').mean()*100:.0f}%")

    st.subheader("Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        order = ["12-14", "15-17", "18-20", "21-24", "25-27"]
        cnt = df["A1_age_group"].value_counts().reindex(order).reset_index()
        cnt.columns = ["Age group", "Respondents"]
        fig = px.bar(cnt, x="Age group", y="Respondents", color="Respondents", color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cnt = df["A3_emirate"].value_counts().reset_index()
        cnt.columns = ["Emirate", "Respondents"]
        fig = px.bar(cnt, x="Emirate", y="Respondents", color="Respondents", color_continuous_scale="Greens")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        cnt = df["A2_residency_status"].value_counts().reset_index()
        cnt.columns = ["Residency", "Respondents"]
        fig = px.bar(cnt, x="Respondents", y="Residency", orientation="h", color="Respondents", color_continuous_scale="Reds")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stress, Stigma & AI-Trust Distributions")
    cols = st.columns(len(SCALE_NUM))
    for c, colname in zip(cols, SCALE_NUM):
        with c:
            fig = px.histogram(df, x=colname, nbins=10, color_discrete_sequence=["#8172B2"])
            fig.update_layout(title=SCALE_LABELS[colname], title_font_size=11, height=260,
                               margin=dict(l=10, r=10, t=40, b=10), showlegend=False, yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Drivers (multi-select questions)")
    sel = st.selectbox("Choose a question", list(MULTI_Q_TITLES.keys()),
                        format_func=lambda k: f"{k} – {MULTI_Q_TITLES[k]}")
    cols_sel = [c for c in multi_cols if c.startswith(sel + "_")]
    rates = (df[cols_sel].mean().sort_values(ascending=True) * 100).reset_index()
    rates.columns = ["Option", "Pct"]
    rates["Option"] = rates["Option"].apply(lambda c: c.split("_", 1)[1])
    fig = px.bar(rates, x="Pct", y="Option", orientation="h", color="Pct", color_continuous_scale="Teal")
    fig.update_layout(title=MULTI_Q_TITLES[sel], xaxis_title="% of respondents selecting", height=420,
                       coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Skewness & Data-Quality Check")
    colA, colB = st.columns(2)
    with colA:
        fig = px.histogram(df, x="B2_stress_severity_1to10", nbins=10, marginal="box", color_discrete_sequence=["#DD8452"])
        fig.update_layout(title=f"Stress severity — skew = {df['B2_stress_severity_1to10'].skew():.2f}", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        flagged = (df[["_is_noisy_responder", "_is_outlier"]].mean() * 100).reset_index()
        flagged.columns = ["Flag", "Pct"]
        flagged["Flag"] = flagged["Flag"].map({"_is_noisy_responder": "Noisy responders", "_is_outlier": "Seeded outliers"})
        fig = px.bar(flagged, x="Flag", y="Pct", color="Flag", color_discrete_sequence=["#937860", "#CCB974"], text="Pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(title="Data-quality flags (transparency)", showlegend=False, height=380, yaxis_title="% of respondents")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================ SECTION 2
elif section.startswith("2"):
    st.header("Noise vs. Signal Diagnostics")
    st.markdown(
        "Distinguishes **random noise** (a group's average differs from the overall average just by "
        "chance) from a **systemic deviation** (a real, statistically significant difference between groups)."
    )

    kpi = st.selectbox("KPI to diagnose", SCALE_NUM, format_func=lambda c: SCALE_LABELS[c])
    group_dim = st.selectbox("Group by", ["A1_age_group", "A3_emirate", "A4_occupation_status", "_true_persona_segment"])

    grand_mean = df[kpi].mean()
    grand_std = df[kpi].std()
    grp = df.groupby(group_dim)[kpi].agg(["mean", "std", "count"]).reset_index()
    grp["se"] = grp["std"] / np.sqrt(grp["count"])
    ucl = grand_mean + 1.96 * (grand_std / np.sqrt(grp["count"].median()))
    lcl = grand_mean - 1.96 * (grand_std / np.sqrt(grp["count"].median()))
    grp["status"] = np.where((grp["mean"] > ucl) | (grp["mean"] < lcl), "Systemic deviation", "Within normal range (noise)")

    groups_vals = [g[kpi].values for _, g in df.groupby(group_dim)]
    f_stat, p_val = stats.f_oneway(*groups_vals)

    c1, c2, c3 = st.columns(3)
    c1.metric("Grand mean", f"{grand_mean:.2f}")
    c2.metric("ANOVA F-statistic", f"{f_stat:.2f}")
    c3.metric("p-value", f"{p_val:.4f}")

    if p_val < 0.05:
        st.success(f"Differences across **{group_dim}** are statistically significant (p={p_val:.4f}) — "
                    "this looks like a real, systemic pattern rather than random noise.")
    else:
        st.info(f"Differences across **{group_dim}** are not statistically significant (p={p_val:.4f}) — "
                "consistent with random noise around the overall average.")

    fig = go.Figure()
    fig.add_hrect(y0=lcl, y1=ucl, fillcolor="LightGreen", opacity=0.25, line_width=0,
                  annotation_text="Expected range (noise band)", annotation_position="top left")
    colors = grp["status"].map({"Systemic deviation": "#C44E52", "Within normal range (noise)": "#4C72B0"})
    fig.add_trace(go.Scatter(x=grp[group_dim], y=grp["mean"], mode="markers+text",
                              marker=dict(size=14, color=colors),
                              error_y=dict(type="data", array=1.96 * grp["se"]),
                              text=grp["status"], textposition="top center", name="Group mean"))
    fig.add_hline(y=grand_mean, line_dash="dash", line_color="grey", annotation_text="Grand mean")
    fig.update_layout(title=f"{SCALE_LABELS[kpi]} by {group_dim}", height=480, xaxis_tickangle=-30,
                       yaxis_title=SCALE_LABELS[kpi])
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(grp.rename(columns={group_dim: "Group", "mean": "Mean", "count": "N", "status": "Diagnosis"})
                 [["Group", "Mean", "N", "Diagnosis"]].round(2), use_container_width=True)

    st.subheader("Respondent-level noise (data-quality flags)")
    st.caption(
        "Separate from group-level systemic deviation above: these flags mark individual respondents whose "
        "answers were deliberately seeded as inconsistent ('noisy') or extreme ('outlier') during data generation, "
        "for transparency."
    )
    flag_rate = df.groupby(group_dim)[["_is_noisy_responder", "_is_outlier"]].mean().reset_index()
    flag_rate["_is_noisy_responder"] *= 100
    flag_rate["_is_outlier"] *= 100
    fig2 = px.bar(flag_rate, x=group_dim, y=["_is_noisy_responder", "_is_outlier"], barmode="group",
                  labels={"value": "% of respondents", "variable": "Flag"})
    fig2.update_layout(height=380, xaxis_tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================ SECTION 3
elif section.startswith("3"):
    st.header("Factor Impact Mapping")
    st.markdown("How do variations in one operational factor relate to variations in another?")

    corr_method = st.radio("Correlation method", ["Pearson", "Spearman"], horizontal=True)
    func = stats.pearsonr if corr_method == "Pearson" else stats.spearmanr

    n = len(ENGINEERED_NUM)
    r_mat = pd.DataFrame(np.eye(n), index=ENGINEERED_NUM, columns=ENGINEERED_NUM)
    p_mat = pd.DataFrame(np.zeros((n, n)), index=ENGINEERED_NUM, columns=ENGINEERED_NUM)
    for i, a in enumerate(ENGINEERED_NUM):
        for j, b in enumerate(ENGINEERED_NUM):
            if i <= j:
                r, p = func(df[a], df[b])
                r_mat.loc[a, b] = r_mat.loc[b, a] = r
                p_mat.loc[a, b] = p_mat.loc[b, a] = p

    nice = {c: SCALE_LABELS.get(c, c.replace("_", " ")) for c in ENGINEERED_NUM}
    r_disp = r_mat.rename(index=nice, columns=nice)

    st.subheader("Correlation matrix")
    fig = px.imshow(r_disp, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Driver network — significant relationships (|r| ≥ threshold, p < 0.05)")
    thresh = st.slider("Minimum |r| to show as a link", 0.05, 0.6, 0.15, 0.05)
    edges = []
    for i, a in enumerate(ENGINEERED_NUM):
        for j, b in enumerate(ENGINEERED_NUM):
            if i < j and abs(r_mat.loc[a, b]) >= thresh and p_mat.loc[a, b] < 0.05:
                edges.append((a, b, r_mat.loc[a, b]))

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {f: (np.cos(a), np.sin(a)) for f, a in zip(ENGINEERED_NUM, angles)}
    fig2 = go.Figure()
    for a, b, r in edges:
        x0, y0 = pos[a]; x1, y1 = pos[b]
        fig2.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                   line=dict(width=1 + 6 * abs(r), color="#C44E52" if r < 0 else "#4C72B0"),
                                   opacity=0.6, hoverinfo="text", text=f"{nice[a]} ↔ {nice[b]}: r={r:.2f}",
                                   showlegend=False))
    xs = [pos[f][0] for f in ENGINEERED_NUM]; ys = [pos[f][1] for f in ENGINEERED_NUM]
    fig2.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=[nice[f] for f in ENGINEERED_NUM],
                               textposition="top center", marker=dict(size=22, color="#55A868"), showlegend=False))
    fig2.update_layout(height=560, xaxis=dict(visible=False), yaxis=dict(visible=False),
                        title=f"{len(edges)} significant relationships at |r| ≥ {thresh}")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top relationships, ranked by strength")
    pairs = []
    for i, a in enumerate(ENGINEERED_NUM):
        for j, b in enumerate(ENGINEERED_NUM):
            if i < j:
                pairs.append({"Factor A": nice[a], "Factor B": nice[b], "r": r_mat.loc[a, b], "p-value": p_mat.loc[a, b]})
    pairs_df = pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False).head(10).reset_index(drop=True)
    pairs_df["Direction"] = np.where(pairs_df["r"] > 0, "As A↑, B↑", "As A↑, B↓")
    pairs_df["Significant (p<0.05)"] = pairs_df["p-value"] < 0.05
    st.dataframe(pairs_df.round(3), use_container_width=True)

# ============================================================ SECTION 4
elif section.startswith("4"):
    st.header("Hypothesis Validation")
    st.markdown("Testing specific assumptions about underlying operational problems against the data.")

    results = []

    r, p = stats.spearmanr(df["B2_stress_severity_1to10"], df["Interest_numeric"])
    results.append({
        "Hypothesis": "H1: Higher stress severity is associated with greater interest in the support tool",
        "Test": "Spearman correlation", "Statistic": r, "p-value": p,
        "Verdict": "Supported" if p < 0.05 and r > 0 else ("Not supported" if p >= 0.05 else "Rejected (opposite direction)"),
        "Takeaway": f"Stress and interest move {'together' if r > 0 else 'oppositely'} (ρ={r:.2f})."
    })

    chi2, p2, dof, v, ct = chi_square_assoc(df, "C2_stigma_judged", "F2_would_pay")
    pay_high_judged = df[df["C2_stigma_judged"].isin(["Yes, a lot", "Somewhat"])]["F2_would_pay"].eq("Yes, willing to pay").mean()
    pay_low_judged = df[df["C2_stigma_judged"].isin(["Not at all", "Not really"])]["F2_would_pay"].eq("Yes, willing to pay").mean()
    results.append({
        "Hypothesis": "H2: Feeling judged about mental health reduces willingness to pay",
        "Test": "Chi-square test of independence", "Statistic": chi2, "p-value": p2,
        "Verdict": "Supported" if p2 < 0.05 and pay_high_judged < pay_low_judged else "Not supported",
        "Takeaway": f"Pay-rate when judged a lot/somewhat = {pay_high_judged*100:.0f}% vs "
                    f"{pay_low_judged*100:.0f}% when not judged (Cramér's V={v:.2f})."
    })

    r3, p3 = stats.pearsonr(df["E1_ai_comfort_1to5"], df["E4_ai_rec_likelihood_1to5"])
    results.append({
        "Hypothesis": "H3: Comfort with AI predicts likelihood of following AI recommendations",
        "Test": "Pearson correlation", "Statistic": r3, "p-value": p3,
        "Verdict": "Supported" if p3 < 0.05 and r3 > 0 else "Not supported",
        "Takeaway": f"Strong {'positive' if r3>0 else 'negative'} relationship (r={r3:.2f})."
    })

    groups4 = [g["D3_recommend_likelihood_1to5"].values for _, g in df.groupby("B4_coping_ability")]
    f4, p4 = stats.f_oneway(*groups4)
    results.append({
        "Hypothesis": "H4: Coping ability is associated with recommend-likelihood",
        "Test": "One-way ANOVA", "Statistic": f4, "p-value": p4,
        "Verdict": "Supported" if p4 < 0.05 else "Not supported",
        "Takeaway": "Recommend-likelihood differs meaningfully across coping-ability groups." if p4 < 0.05
                    else "No statistically significant difference across coping-ability groups."
    })

    for res in results:
        color = "🟢" if res["Verdict"] == "Supported" else ("🟡" if "Not supported" in res["Verdict"] else "🔴")
        with st.container(border=True):
            st.markdown(f"**{res['Hypothesis']}**")
            c1, c2, c3 = st.columns([1, 1, 2])
            c1.metric(res["Test"], f"{res['Statistic']:.3f}")
            c2.metric("p-value", f"{res['p-value']:.4f}")
            c3.markdown(f"{color} **{res['Verdict']}** — {res['Takeaway']}")

# ============================================================ SECTION 5
elif section.startswith("5"):
    st.header("Feature Engineering")
    st.markdown("Derived features built from the raw survey responses to power the downstream models.")

    feat_table = pd.DataFrame([
        {"Feature": "Age_numeric", "Formula": "Midpoint of age-group bracket", "Why": "Turns an ordinal bracket into a continuous predictor for correlation/regression."},
        {"Feature": "Coping_numeric", "Formula": "B4_coping_ability mapped 1–4", "Why": "Ordinal-encodes a Likert-style text response."},
        {"Feature": "Stigma_judged_numeric", "Formula": "C2_stigma_judged mapped 1–4", "Why": "Ordinal-encodes perceived judgment for use in indices and correlations."},
        {"Feature": "Stress_Tier", "Formula": "B2 binned into Low (≤3) / Medium (4-6) / High (7-10)", "Why": "Simplifies a 1-10 scale into actionable risk tiers for segmentation/CHAID."},
        {"Feature": "Risk_Index", "Formula": "Mean z-score of [stress, stigma discomfort, feeling judged, poor coping]", "Why": "Single composite 'mental health risk' score combining 4 related signals."},
        {"Feature": "AI_Trust_Index", "Formula": "Mean z-score of [AI comfort, AI-recommendation likelihood]", "Why": "Single composite measure of trust/openness toward an AI-based tool."},
        {"Feature": "Stress_x_LowCoping", "Formula": "B2_stress_severity × (5 − Coping_numeric)", "Why": "Interaction term capturing compounding risk when high stress meets poor coping."},
    ])
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

    st.subheader("Engineered feature distributions")
    colA, colB = st.columns(2)
    with colA:
        fig = px.histogram(df, x="Risk_Index", nbins=25, color_discrete_sequence=["#C44E52"])
        fig.update_layout(title="Risk_Index distribution", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig = px.histogram(df, x="AI_Trust_Index", nbins=25, color_discrete_sequence=["#4C72B0"])
        fig.update_layout(title="AI_Trust_Index distribution", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk vs. AI-Trust, colored by willingness to pay")
    fig = px.scatter(df, x="Risk_Index", y="AI_Trust_Index", color="F2_would_pay", opacity=0.6,
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The composite indices separate respondents more cleanly than any single raw item — "
               "this is the feature space used for dynamic segmentation in the next section.")

# ============================================================ SECTION 6
elif section.startswith("6"):
    st.header("Dynamic Segmentation (Drill-Down Clustering)")
    st.markdown("Dynamically splits high-level metrics into granular sub-segments using unsupervised clustering, "
                "then lets you drill into any one segment and re-cluster it further.")

    base_feats = ["Risk_Index", "AI_Trust_Index"]
    k1 = st.slider("Top-level segments (k)", 2, 6, 3)
    X1 = StandardScaler().fit_transform(df[base_feats])
    km1 = KMeans(n_clusters=k1, random_state=42, n_init=10).fit(X1)
    df_seg = df.copy()
    df_seg["Segment_L1"] = km1.labels_.astype(str)
    sil1 = silhouette_score(X1, km1.labels_)

    c1, c2 = st.columns(2)
    c1.metric("Top-level silhouette score", f"{sil1:.2f}")
    c2.metric("Segments", k1)

    fig = px.scatter(df_seg, x="Risk_Index", y="AI_Trust_Index", color="Segment_L1",
                      color_discrete_sequence=px.colors.qualitative.Set1, opacity=0.65,
                      title="Level 1 — top-level segmentation")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    prof1 = df_seg.groupby("Segment_L1")[base_feats + ["B2_stress_severity_1to10"]].mean().round(2)
    prof1["N"] = df_seg.groupby("Segment_L1").size()
    st.dataframe(prof1, use_container_width=True)

    st.subheader("Drill down into a segment")
    drill = st.selectbox("Select a Level-1 segment to break into sub-segments", sorted(df_seg["Segment_L1"].unique()))
    sub = df_seg[df_seg["Segment_L1"] == drill].copy()
    sub_feats = SCALE_NUM
    k2 = st.slider("Sub-segments (k) within this segment", 2, 4, 2)

    if len(sub) > k2 * 3:
        X2 = StandardScaler().fit_transform(sub[sub_feats])
        km2 = KMeans(n_clusters=k2, random_state=42, n_init=10).fit(X2)
        sub["Segment_L2"] = km2.labels_.astype(str)
        sil2 = silhouette_score(X2, km2.labels_)

        st.metric(f"Sub-segmentation silhouette (segment {drill}, n={len(sub)})", f"{sil2:.2f}")

        pca2 = PCA(n_components=2).fit_transform(X2)
        sub_plot = sub.copy()
        sub_plot["pc1"], sub_plot["pc2"] = pca2[:, 0], pca2[:, 1]
        fig2 = px.scatter(sub_plot, x="pc1", y="pc2", color="Segment_L2",
                           color_discrete_sequence=px.colors.qualitative.Set2,
                           title=f"Level 2 — sub-segments of Segment {drill} (PCA projection)")
        fig2.update_layout(height=460)
        st.plotly_chart(fig2, use_container_width=True)

        prof2 = sub.groupby("Segment_L2")[sub_feats].mean().round(2)
        prof2["N"] = sub.groupby("Segment_L2").size()
        st.dataframe(prof2.rename(columns=SCALE_LABELS), use_container_width=True)

        st.subheader("Drill-down map")
        labels = ["All respondents"] + [f"Segment {s}" for s in sorted(df_seg["Segment_L1"].unique())] + \
                 [f"{drill}.{s2}" for s2 in sorted(sub["Segment_L2"].unique())]
        parents = [""] + ["All respondents"] * k1 + [f"Segment {drill}"] * k2
        values = [len(df_seg)] + [int((df_seg["Segment_L1"] == s).sum()) for s in sorted(df_seg["Segment_L1"].unique())] + \
                 [int((sub["Segment_L2"] == s2).sum()) for s2 in sorted(sub["Segment_L2"].unique())]
        fig3 = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total"))
        fig3.update_layout(height=520, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Selected segment too small to sub-cluster further with this k.")

# ============================================================ SECTION 7
elif section.startswith("7"):
    st.header("Regression Analysis")

    reg_target = st.selectbox("Target (continuous)", SCALE_NUM, format_func=lambda c: SCALE_LABELS[c])
    reg_feat_num = [c for c in ENGINEERED_NUM if c != reg_target]
    if reg_target in RISK_INDEX_COMPONENTS:
        reg_feat_num = [c for c in reg_feat_num if c != "Risk_Index"]  # Risk_Index is built from this target — avoid leakage
    if reg_target in AI_TRUST_COMPONENTS:
        reg_feat_num = [c for c in reg_feat_num if c != "AI_Trust_Index"]  # AI_Trust_Index is built from this target — avoid leakage
    reg_feat_cat = KEY_CAT
    model_choice = st.radio("Model", ["Linear Regression", "Random Forest Regressor"], horizontal=True)

    X = df[reg_feat_num + reg_feat_cat]
    y = df[reg_target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), reg_feat_cat),
                              ("num", "passthrough", reg_feat_num)])
    if model_choice == "Linear Regression":
        model = Pipeline([("pre", pre), ("reg", LinearRegression())])
    else:
        model = Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=200, random_state=42))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2_score(y_test, pred):.3f}")
    c2.metric("MAE", f"{mean_absolute_error(y_test, pred):.3f}")
    c3.metric("Test rows", len(y_test))

    colA, colB = st.columns(2)
    with colA:
        fig = px.scatter(x=y_test, y=pred, labels={"x": "Actual", "y": "Predicted"}, opacity=0.6,
                          title="Actual vs. Predicted")
        lo, hi = float(min(y_test.min(), pred.min())), float(max(y_test.max(), pred.max()))
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="red", dash="dash"))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        if model_choice == "Linear Regression":
            cat_names = model.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(reg_feat_cat)
            names = list(cat_names) + reg_feat_num
            coefs = model.named_steps["reg"].coef_
            imp = pd.DataFrame({"Feature": names, "Coefficient": coefs}).sort_values("Coefficient", key=abs, ascending=False).head(10)
            fig = px.bar(imp, x="Coefficient", y="Feature", orientation="h", color="Coefficient", color_continuous_scale="RdBu")
            fig.update_layout(title="Top standardized-effect coefficients", height=420, coloraxis_showscale=False)
        else:
            cat_names = model.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(reg_feat_cat)
            names = list(cat_names) + reg_feat_num
            imp = pd.DataFrame({"Feature": names, "Importance": model.named_steps["reg"].feature_importances_}) \
                    .sort_values("Importance", ascending=False).head(10)
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis")
            fig.update_layout(title="Top feature importances", height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Predicting **{SCALE_LABELS[reg_target]}** from engineered numeric indices + key categorical drivers.")

# ============================================================ SECTION 8
elif section.startswith("8"):
    st.header("Classification & Data Mining")

    target = st.selectbox("Target (categorical)", CLASS_TARGETS)
    feat_cat = [c for c in KEY_CAT if c != target]  # avoid leaking target into its own features
    feat_num = ENGINEERED_NUM

    X = df[feat_num + feat_cat]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat),
                              ("num", "passthrough", feat_num)])
    models = {
        "Logistic Regression": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))]),
        "Random Forest": Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))]),
    }
    rows = []
    preds = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        preds[name] = p
        rows.append({"Model": name, "Accuracy": accuracy_score(y_test, p), "Macro F1": f1_score(y_test, p, average="macro")})
    baseline_acc = y_test.value_counts(normalize=True).max()
    cmp_df = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(cmp_df, x="Model", y=["Accuracy", "Macro F1"], barmode="group",
                     title=f"Model comparison (majority-class baseline = {baseline_acc:.3f})")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        chosen = st.selectbox("Confusion matrix for", list(models.keys()))
        labels_sorted = sorted(y.unique())
        cm = confusion_matrix(y_test, preds[chosen], labels=labels_sorted)
        fig = px.imshow(cm, x=labels_sorted, y=labels_sorted, text_auto=True, color_continuous_scale="Blues",
                         labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(title=f"Confusion matrix — {chosen}", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature importance (Random Forest)")
    rf = models["Random Forest"]
    cat_names = rf.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(feat_cat)
    names = list(cat_names) + feat_num
    imp = pd.DataFrame({"Feature": names, "Importance": rf.named_steps["clf"].feature_importances_}) \
            .sort_values("Importance", ascending=False).head(12)
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Plasma")
    fig.update_layout(height=460, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mined pattern: highest-propensity sub-group")
    target_class = st.selectbox("Class of interest", labels_sorted)
    rule_dim = st.selectbox("Slice by", [c for c in KEY_CAT if c != target] + ["Stress_Tier"])
    rate_tbl = df.groupby(rule_dim)[target].apply(lambda s: (s == target_class).mean()).sort_values(ascending=False).reset_index()
    rate_tbl.columns = [rule_dim, "Rate"]
    overall_rate = (df[target] == target_class).mean()
    fig = px.bar(rate_tbl, x=rule_dim, y="Rate", color="Rate", color_continuous_scale="Sunset")
    fig.add_hline(y=overall_rate, line_dash="dash", annotation_text=f"Overall rate ({overall_rate:.2f})")
    fig.update_layout(title=f"P({target} = '{target_class}') by {rule_dim}", height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================ SECTION 9
elif section.startswith("9"):
    st.header("Decision Tree — Gini Impurity & Information Gain")

    target9 = st.selectbox("Target (categorical)", CLASS_TARGETS, key="dt_target")
    feat_cat9 = [c for c in KEY_CAT if c != target9]
    feat_num9 = ENGINEERED_NUM
    max_depth = st.slider("Max depth", 2, 5, 3)

    X = df[feat_num9 + feat_cat9]
    y = df[target9]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat9),
                              ("num", "passthrough", feat_num9)])
    X_enc = pre.fit_transform(X)
    feature_names = list(pre.named_transformers_["cat"].get_feature_names_out(feat_cat9)) + feat_num9

    X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
    dt = DecisionTreeClassifier(max_depth=max_depth, criterion="gini", random_state=42, min_samples_leaf=20)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
    c2.metric("Tree depth used", dt.get_depth())
    c3.metric("Leaves", dt.get_n_leaves())

    st.subheader("Tree structure")
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(dt, feature_names=feature_names, class_names=[str(c) for c in le.classes_],
              filled=True, rounded=True, fontsize=7, ax=ax, max_depth=max_depth)
    st.pyplot(fig)

    st.subheader("Node-level Gini impurity & information gain")
    tree_ = dt.tree_
    rows = []

    def entropy_of_value(value_row, n):
        p = value_row / n
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    for node_id in range(tree_.node_count):
        n_samples = tree_.n_node_samples[node_id]
        gini = tree_.impurity[node_id]
        value = tree_.value[node_id][0]
        ent = entropy_of_value(value, n_samples)
        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]
        if is_leaf:
            info_gain = np.nan
            split_feat = "— (leaf)"
        else:
            left, right = tree_.children_left[node_id], tree_.children_right[node_id]
            n_l, n_r = tree_.n_node_samples[left], tree_.n_node_samples[right]
            child_gini = (n_l / n_samples) * tree_.impurity[left] + (n_r / n_samples) * tree_.impurity[right]
            info_gain = gini - child_gini
            split_feat = feature_names[tree_.feature[node_id]]
        rows.append({"Node": node_id, "Split feature": split_feat, "N samples": int(n_samples),
                     "Gini impurity": gini, "Entropy": ent, "Information gain": info_gain})
    node_df = pd.DataFrame(rows)
    st.dataframe(node_df.round(4), use_container_width=True, hide_index=True)

    st.subheader("Top splits ranked by information gain")
    top_splits = node_df.dropna(subset=["Information gain"]).sort_values("Information gain", ascending=False).head(8)
    fig2 = px.bar(top_splits, x="Information gain", y=top_splits["Node"].astype(str) + ": " + top_splits["Split feature"],
                  orientation="h", color="Information gain", color_continuous_scale="Cividis")
    fig2.update_layout(height=420, yaxis_title="Node: feature", coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Gini impurity measures how mixed a node's classes are (0 = pure). Information gain is the "
               "reduction in impurity achieved by a split — the tree greedily picks the split with the highest gain at each node.")

# ============================================================ SECTION 10
elif section.startswith("10"):
    st.header("CHAID — Chi-Square Automatic Interaction Detector")
    st.markdown("A simplified 2-level CHAID: at each node, picks the categorical predictor most significantly "
                "associated with the target (lowest chi-square p-value) and splits on it, recursing one level deeper.")

    target10 = st.selectbox("Target (categorical)", CLASS_TARGETS, key="chaid_target")
    candidates = [c for c in CHAID_CATS if c != target10]
    min_node = st.slider("Minimum node size to keep splitting", 30, 150, 50, 10)
    alpha = st.slider("Significance threshold (alpha)", 0.01, 0.10, 0.05, 0.01)

    def best_split(data, predictors, target, alpha):
        best = None
        for pcol in predictors:
            if data[pcol].nunique() < 2:
                continue
            try:
                chi2, p, dof, v, ct = chi_square_assoc(data, pcol, target)
            except Exception:
                continue
            if (ct.values.sum() == 0) or (ct.shape[0] < 2):
                continue
            if best is None or p < best["p"]:
                best = {"feature": pcol, "chi2": chi2, "p": p, "v": v}
        if best is not None and best["p"] < alpha:
            return best
        return None

    target_classes = sorted(df[target10].unique())
    nodes, edges_chaid, split_log = [], [], []
    root_id = "Root"
    nodes.append({"id": root_id, "label": f"All (n={len(df)})", "parent": "", "n": len(df)})

    split1 = best_split(df, candidates, target10, alpha)
    if split1 is None:
        st.warning("No categorical predictor is significantly associated with this target at the chosen alpha.")
    else:
        split_log.append({"Level": 1, "Node": "Root", "Split on": split1["feature"], "chi2": split1["chi2"],
                           "p-value": split1["p"], "Cramér's V": split1["v"]})
        for cat1, sub1 in df.groupby(split1["feature"]):
            node1_id = f"{split1['feature']}={cat1}"
            nodes.append({"id": node1_id, "label": f"{cat1} (n={len(sub1)})", "parent": root_id, "n": len(sub1)})
            remaining = [c for c in candidates if c != split1["feature"]]
            if len(sub1) >= min_node:
                split2 = best_split(sub1, remaining, target10, alpha)
            else:
                split2 = None
            if split2 is not None:
                split_log.append({"Level": 2, "Node": node1_id, "Split on": split2["feature"], "chi2": split2["chi2"],
                                   "p-value": split2["p"], "Cramér's V": split2["v"]})
                for cat2, sub2 in sub1.groupby(split2["feature"]):
                    node2_id = f"{node1_id} | {split2['feature']}={cat2}"
                    nodes.append({"id": node2_id, "label": f"{cat2} (n={len(sub2)})", "parent": node1_id, "n": len(sub2)})

        # compute majority class per node by re-filtering on its path
        def slice_for(node_id):
            if node_id == root_id:
                return df
            parts = node_id.split(" | ")
            mask = pd.Series(True, index=df.index)
            for part in parts:
                feat, val = part.split("=", 1)
                mask &= df[feat].astype(str) == val
            return df[mask]

        for n in nodes:
            sl = slice_for(n["id"])
            if len(sl) > 0:
                n["majority"] = sl[target10].mode().iloc[0]
                n["purity"] = (sl[target10] == n["majority"]).mean()
            else:
                n["majority"], n["purity"] = "—", 0.0

        st.subheader("CHAID split decisions")
        st.dataframe(pd.DataFrame(split_log).round(4), use_container_width=True, hide_index=True)

        st.subheader("Segment tree (size = node N, color = dominant class purity)")
        labels = [n["label"] for n in nodes]
        ids = [n["id"] for n in nodes]
        parents = [n["parent"] for n in nodes]
        values = [n["n"] for n in nodes]
        custom = [f"{n['majority']} ({n['purity']*100:.0f}%)" for n in nodes]
        fig = go.Figure(go.Treemap(labels=labels, ids=ids, parents=parents, values=values,
                                    customdata=custom, branchvalues="total",
                                    hovertemplate="%{label}<br>N=%{value}<br>Majority class: %{customdata}<extra></extra>",
                                    marker=dict(colors=[n["purity"] for n in nodes], colorscale="RdYlGn", cmin=0, cmax=1)))
        fig.update_layout(height=560, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Target = **{target10}**. Each box's color shows how dominant its majority class is "
                    "(greener = purer/more predictable segment).")

# ============================================================ SECTION 11
elif section.startswith("11"):
    st.header("Random Forest vs. Gradient Boosted Trees")

    target11 = st.selectbox("Target (categorical)", CLASS_TARGETS, key="rfgbt_target")
    feat_cat11 = [c for c in KEY_CAT if c != target11]
    feat_num11 = ENGINEERED_NUM

    X = df[feat_num11 + feat_cat11]
    y = df[target11]
    le11 = LabelEncoder()
    y_enc = le11.fit_transform(y)
    classes11 = le11.classes_

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat11),
                              ("num", "passthrough", feat_num11)])
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

    rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])
    gbt = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(random_state=42, max_depth=4))])
    rf.fit(X_train, y_train); gbt.fit(X_train, y_train)
    rf_pred = rf.predict(X_test); gbt_pred = gbt.predict(X_test)
    rf_proba = rf.predict_proba(X_test); gbt_proba = gbt.predict_proba(X_test)

    baseline_acc = pd.Series(y_test).value_counts(normalize=True).max()
    cmp = pd.DataFrame([
        {"Model": "Random Forest", "Accuracy": accuracy_score(y_test, rf_pred), "Macro F1": f1_score(y_test, rf_pred, average="macro")},
        {"Model": "Gradient Boosted Trees", "Accuracy": accuracy_score(y_test, gbt_pred), "Macro F1": f1_score(y_test, gbt_pred, average="macro")},
    ])

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(cmp, x="Model", y=["Accuracy", "Macro F1"], barmode="group",
                     title=f"Accuracy / F1 (baseline = {baseline_acc:.3f})")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if len(classes11) == 2:
            y_test_bin = (y_test == 1).astype(int)
            fig = go.Figure()
            for name, proba in [("Random Forest", rf_proba), ("Gradient Boosted Trees", gbt_proba)]:
                fpr, tpr, _ = roc_curve(y_test_bin, proba[:, 1])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc(fpr, tpr):.2f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Chance"))
            fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            y_test_oh = label_binarize(y_test, classes=range(len(classes11)))
            fig = go.Figure()
            for name, proba in [("Random Forest", rf_proba), ("Gradient Boosted Trees", gbt_proba)]:
                fpr, tpr, _ = roc_curve(y_test_oh.ravel(), proba.ravel())
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (micro-AUC={auc(fpr, tpr):.2f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Chance"))
            fig.update_layout(title="Micro-average ROC (multi-class)", height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature importance comparison")
    cat_names = rf.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(feat_cat11)
    names = list(cat_names) + feat_num11
    rf_imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
    perm = permutation_importance(gbt, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)
    gbt_imp = pd.Series(perm.importances_mean, index=X.columns.tolist())
    gbt_imp_expanded = pd.Series(0.0, index=names)
    for col in feat_num11:
        gbt_imp_expanded[col] = gbt_imp.get(col, 0.0)
    for col in feat_cat11:
        share = gbt_imp.get(col, 0.0)
        matching = [n for n in cat_names if n.startswith(col + "_")]
        for m in matching:
            gbt_imp_expanded[m] = share / max(len(matching), 1)

    top_feats = rf_imp.sort_values(ascending=False).head(10).index
    comp_imp = pd.DataFrame({"Random Forest": rf_imp[top_feats], "Gradient Boosted (permutation)": gbt_imp_expanded[top_feats]}).reset_index()
    comp_imp.columns = ["Feature", "Random Forest", "Gradient Boosted (permutation)"]
    fig = px.bar(comp_imp, x=["Random Forest", "Gradient Boosted (permutation)"], y="Feature", orientation="h", barmode="group")
    fig.update_layout(height=460, xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Random Forest importance = mean impurity decrease; Gradient Boosted importance = permutation "
               "importance (drop in accuracy when a feature's values are shuffled), aggregated back to raw categorical columns.")

# ============================================================ SECTION 12
elif section.startswith("12"):
    st.header("Association Rules (Apriori)")
    st.markdown("Market-basket-style mining over the 50 binary multi-select survey items to find co-occurring patterns.")

    min_support = st.slider("Minimum support", 0.02, 0.20, 0.05, 0.01)
    min_lift = st.slider("Minimum lift", 1.0, 3.0, 1.2, 0.1)

    basket = df[multi_cols].astype(bool)
    freq = apriori(basket, min_support=min_support, use_colnames=True, max_len=3)
    if freq.empty:
        st.warning("No itemsets found at this support threshold — try lowering it.")
    else:
        rules = association_rules(freq, metric="lift", min_threshold=min_lift)
        rules = rules[(rules["antecedents"].apply(len) <= 2) & (rules["consequents"].apply(len) == 1)]
        rules = rules.sort_values("lift", ascending=False).head(20).copy()

        if rules.empty:
            st.warning("No rules at these thresholds — try lowering minimum support or lift.")
        else:
            rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
            rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
            show = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
            show["support"] = show["support"].round(3)
            show["confidence"] = show["confidence"].round(3)
            show["lift"] = show["lift"].round(2)

            c1, c2, c3 = st.columns(3)
            c1.metric("Itemsets found", len(freq))
            c2.metric("Rules shown", len(show))
            c3.metric("Top lift", f"{show['lift'].max():.2f}")

            fig = px.scatter(show, x="support", y="confidence", size="lift", color="lift",
                              hover_data=["antecedents", "consequents"], color_continuous_scale="Magma",
                              title="Rules: support vs. confidence (bubble size/color = lift)")
            fig.update_layout(height=460)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(show, use_container_width=True, hide_index=True)
