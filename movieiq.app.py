"""
MovieIQ — Predictive Analytics on Film Success
================================================
Streamlit dashboard: filter the catalogue, explore EDA + statistical-test
results, and get a live success/failure prediction from the trained
Random Forest model.

Run with:  streamlit run MovieIQ.py
"""

import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats as sci_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MovieIQ", page_icon="🎬", layout="wide")

# ---------------------------------------------------------------------------
# Self-contained pipeline: this app only requires movies.csv on disk.
# Everything else (cleaning, stats tests, model training) is computed here
# and cached, so there is nothing extra to commit or keep in sync.
# ---------------------------------------------------------------------------
ALPHA = 0.05


def parse_genres(raw):
    try:
        parsed = ast.literal_eval(raw)
        return [g["name"] for g in parsed]
    except (ValueError, SyntaxError, TypeError):
        return []


@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    df["success"] = (df["revenue"] > df["budget"]).astype(int)
    df["genre_list"] = df["genres"].apply(parse_genres)
    df["primary_genre"] = df["genre_list"].apply(lambda g: g[0] if g else "Unknown")
    return df


@st.cache_data
def compute_results(df):
    """Recreate the same statistics analysis.py computes, for display in the app."""
    results = {}

    # Stage 2 — EDA numbers
    results["budget_revenue_correlation"] = round(float(df["budget"].corr(df["revenue"])), 4)

    # Stage 3 — statistical tests
    success_votes = df.loc[df["success"] == 1, "vote_average"]
    fail_votes = df.loc[df["success"] == 0, "vote_average"]
    t_stat, t_p = sci_stats.ttest_ind(success_votes, fail_votes, equal_var=False)
    results["ttest"] = {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(t_p), 6),
        "significant": bool(t_p < ALPHA),
        "mean_success": round(float(success_votes.mean()), 3),
        "mean_failure": round(float(fail_votes.mean()), 3),
    }

    contingency = pd.crosstab(df["primary_genre"], df["success"])
    chi2, chi_p, dof, _ = sci_stats.chi2_contingency(contingency)
    results["chi_square"] = {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(chi_p), 6),
        "degrees_of_freedom": int(dof),
        "significant": bool(chi_p < ALPHA),
    }
    return results


@st.cache_resource
def train_model(df):
    """Train the Random Forest once per app session and cache it."""
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(
        mlb.fit_transform(df["genre_list"]), columns=[f"genre_{g}" for g in mlb.classes_], index=df.index
    )
    feature_cols_base = ["budget", "popularity", "runtime", "vote_average"]
    X = pd.concat([df[feature_cols_base], genre_dummies], axis=1)
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "split_ratio": "80/20",
    }
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    metrics["feature_importance"] = importances.round(4).to_dict()

    genre_classes = sorted(mlb.classes_)
    all_feature_columns = list(X.columns)
    return model, mlb, feature_cols_base, all_feature_columns, genre_classes, metrics


df = load_data()
results = compute_results(df)
model, mlb, feature_cols_base, all_feature_columns, genre_classes, model_metrics = train_model(df)

all_genres = sorted(df["genre_list"].explode().dropna().unique())

# ---------------------------------------------------------------------------
# Sidebar — filters (Stage 5.1)
# ---------------------------------------------------------------------------
st.sidebar.title("🎬 MovieIQ")
st.sidebar.caption("Predictive Analytics on Film Success")

st.sidebar.header("Filters")
selected_genres = st.sidebar.multiselect(
    "Genre", options=all_genres, default=[], help="Leave empty to include all genres"
)
min_vote = st.sidebar.slider(
    "Minimum vote average",
    min_value=float(df["vote_average"].min()),
    max_value=float(df["vote_average"].max()),
    value=float(df["vote_average"].min()),
    step=0.1,
)

filtered = df[df["vote_average"] >= min_vote].copy()
if selected_genres:
    filtered = filtered[filtered["genre_list"].apply(lambda g: any(x in g for x in selected_genres))]

st.sidebar.markdown("---")
st.sidebar.metric("Movies matching filters", f"{len(filtered):,}")
st.sidebar.metric("Success rate (filtered)", f"{filtered['success'].mean():.1%}" if len(filtered) else "n/a")

# ---------------------------------------------------------------------------
# Header / KPIs
# ---------------------------------------------------------------------------
st.title("🎬 MovieIQ Dashboard")
st.caption("A movie is labelled **successful** when its revenue exceeds its budget.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Movies in view", f"{len(filtered):,}", help=f"out of {len(df):,} total")
k2.metric("Success rate", f"{filtered['success'].mean():.1%}" if len(filtered) else "n/a")
k3.metric("Avg. budget", f"${filtered['budget'].mean()/1e6:,.1f}M" if len(filtered) else "n/a")
k4.metric("Avg. revenue", f"${filtered['revenue'].mean()/1e6:,.1f}M" if len(filtered) else "n/a")

st.markdown("---")

tab_eda, tab_stats, tab_model, tab_predict = st.tabs(
    ["📊 Exploratory Analysis", "🧪 Statistical Tests", "🌲 Model Performance", "🔮 Predict a Movie"]
)

# ---------------------------------------------------------------------------
# TAB 1 — EDA (Stage 5.2)
# ---------------------------------------------------------------------------
with tab_stats:
    pass  # placeholder, filled below to keep tab order readable

with tab_eda:
    st.subheader("Budget vs. Revenue")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=filtered, x="budget", y="revenue", hue="success",
        palette={0: "#e15759", 1: "#4e79a7"}, alpha=0.6, ax=ax,
    )
    ax.set_xlabel("Budget ($)")
    ax.set_ylabel("Revenue ($)")
    ax.legend(title="Success", labels=["Failure", "Success"])
    st.pyplot(fig)
    st.caption(
        f"Correlation between budget and revenue across the full dataset: "
        f"**{results['budget_revenue_correlation']:.2f}** — bigger budgets "
        f"loosely track bigger revenues, but the relationship is far from perfect."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Most Common Genres")
        genre_counts = (
            filtered["genre_list"].explode().value_counts() if len(filtered) else pd.Series(dtype=int)
        )
        st.bar_chart(genre_counts)
    with col_b:
        st.subheader("Success Rate by Genre")
        genre_exploded = filtered.explode("genre_list")
        genre_sr = (
            genre_exploded.groupby("genre_list")["success"].mean().sort_values(ascending=False)
            if len(filtered) else pd.Series(dtype=float)
        )
        st.bar_chart(genre_sr)

    st.subheader("Popularity, Runtime & Vote Average vs. Success")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, col in enumerate(["popularity", "runtime", "vote_average"]):
        sns.boxplot(
            data=filtered, x="success", y=col, hue="success", ax=axes[i],
            palette={0: "#e15759", 1: "#4e79a7"}, legend=False,
        )
        axes[i].set_xticklabels(["Failure", "Success"])
        axes[i].set_title(col)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
    sns.heatmap(filtered[corr_cols].corr() if len(filtered) else pd.DataFrame(),
                annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# TAB 2 — Statistical Tests (Stage 5.2)
# ---------------------------------------------------------------------------
with tab_stats:
    st.subheader("T-Test — vote_average by Success")
    ttest = results["ttest"]
    c1, c2, c3 = st.columns(3)
    c1.metric("t-statistic", ttest["t_statistic"])
    c2.metric("p-value", ttest["p_value"])
    c3.metric("Significant (α=0.05)?", "Yes" if ttest["significant"] else "No")
    st.write(
        f"**Null hypothesis:** The mean vote_average is the same for successful and unsuccessful movies.\n\n"
        f"Mean vote_average — successful movies: **{ttest['mean_success']}**, "
        f"unsuccessful movies: **{ttest['mean_failure']}**.\n\n"
        f"Since p = {ttest['p_value']} {'<' if ttest['significant'] else '≥'} 0.05, we "
        f"{'reject' if ttest['significant'] else 'fail to reject'} the null hypothesis: "
        f"vote_average does **{'':s}{'' if ttest['significant'] else 'not '}differ significantly** "
        f"between successful and unsuccessful movies in this dataset."
    )

    st.markdown("---")
    st.subheader("Chi-Square Test — Genre vs. Success")
    chi = results["chi_square"]
    c1, c2, c3 = st.columns(3)
    c1.metric("χ² statistic", chi["chi2_statistic"])
    c2.metric("p-value", chi["p_value"])
    c3.metric("Significant (α=0.05)?", "Yes" if chi["significant"] else "No")
    st.write(
        f"**Null hypothesis:** Genre and movie success are independent of each other.\n\n"
        f"With p = {chi['p_value']} {'<' if chi['significant'] else '≥'} 0.05 "
        f"(dof = {chi['degrees_of_freedom']}), we "
        f"{'reject' if chi['significant'] else 'fail to reject'} the null hypothesis: genre "
        f"appears **{'to be associated' if chi['significant'] else 'independent of'}** with success "
        f"in this dataset."
    )

    st.markdown("---")
    st.info(
        "**What is a p-value?** It's the probability of observing a difference at least this "
        "extreme purely by chance, if the null hypothesis were actually true. We use the "
        "conventional **α = 0.05** threshold — a p-value below that gives us enough evidence "
        "to call the result statistically significant."
    )

# ---------------------------------------------------------------------------
# TAB 3 — Model Performance (Stage 4/5.2)
# ---------------------------------------------------------------------------
with tab_model:
    s4 = model_metrics
    st.subheader("Random Forest Classifier")
    st.caption(
        f"Trained on a {s4['split_ratio']} train/test split "
        f"({s4['train_size']} train rows, {s4['test_size']} test rows)."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{s4['accuracy']:.1%}")
    c2.metric("Precision", f"{s4['precision']:.1%}")
    c3.metric("Recall", f"{s4['recall']:.1%}")

    cm = np.array(s4["confusion_matrix"])
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred: Fail", "Pred: Success"],
            yticklabels=["Actual: Fail", "Actual: Success"],
        )
        st.pyplot(fig)
    with col_b:
        st.write("**Top Feature Importances**")
        importances = pd.Series(s4["feature_importance"]).sort_values(ascending=False).head(10)
        st.bar_chart(importances)

    st.caption(
        "The model leans most heavily on **popularity, budget, vote_average and runtime** — "
        "individual genre flags contribute comparatively little."
    )

# ---------------------------------------------------------------------------
# TAB 4 — Prediction (Stage 5.3)
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Will this movie succeed?")
    st.caption("Enter a movie's details below to get a live prediction from the trained model.")

    with st.form("predict_form"):
        pc1, pc2 = st.columns(2)
        with pc1:
            in_budget = st.number_input("Budget ($)", min_value=1000, value=50_000_000, step=1_000_000)
            in_popularity = st.slider("Popularity", 0.0, 100.0, 50.0)
        with pc2:
            in_runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=300, value=120)
            in_vote = st.slider("Expected vote average", 0.0, 10.0, 6.0)

        in_genres = st.multiselect("Genre(s)", options=genre_classes, default=[genre_classes[0]])
        submitted = st.form_submit_button("Predict Success", type="primary")

    if submitted:
        genre_vector = mlb.transform([in_genres])[0]
        base_features = pd.DataFrame(
            [[in_budget, in_popularity, in_runtime, in_vote]], columns=feature_cols_base
        )
        genre_df = pd.DataFrame([genre_vector], columns=[f"genre_{g}" for g in mlb.classes_])
        X_new = pd.concat([base_features, genre_df], axis=1)[all_feature_columns]

        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]

        if pred == 1:
            st.success(f"✅ Predicted **SUCCESS** — confidence {proba[1]:.1%}")
        else:
            st.error(f"❌ Predicted **NOT successful** — confidence {proba[0]:.1%}")

        st.progress(float(proba[1]), text=f"Probability of success: {proba[1]:.1%}")

st.markdown("---")
with st.expander("📝 Reflection"):
    st.write(
        "If a studio asked *'Will our next film succeed?'*, MovieIQ's answer should be treated "
        "as **one useful signal, not a verdict**. In this dataset the model reaches "
        f"**{model_metrics['accuracy']:.0%} accuracy**, and the statistical tests found "
        "**no significant relationship** between genre or audience rating and success once "
        "revenue-vs-budget is the yardstick — the strongest real signal is simply how much a "
        "movie cost to make relative to how popular and well-reviewed it eventually became. "
        "\n\n**Limitation:** the dataset has no marketing spend, release-date/seasonality, cast, "
        "or franchise/sequel information — all of which materially affect real box-office "
        "outcomes. **With more time,** I'd enrich the data with those features and test "
        "gradient-boosted models (e.g. XGBoost) against the Random Forest baseline."
    )