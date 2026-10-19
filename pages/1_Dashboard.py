
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="T20Pulse | Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"


# --------------------------------------------------
# 2. LOAD SAVED DATA
# --------------------------------------------------

@st.cache_data
def load_dashboard_data():

    matches = pd.read_csv(
        OUTPUTS_DIR / "match_summary.csv",
        parse_dates=["match_date"]
    )

    model_comparison = pd.read_csv(
        OUTPUTS_DIR / "validation_model_comparison.csv"
    )

    test_results = pd.read_csv(
        OUTPUTS_DIR / "final_test_results.csv"
    )

    phase_results = pd.read_csv(
        OUTPUTS_DIR / "phase_wise_evaluation.csv"
    )

    return matches, model_comparison, test_results, phase_results


try:
    matches, model_comparison, test_results, phase_results = (
        load_dashboard_data()
    )

except Exception as exc:
    st.error(f"Could not load dashboard data: {exc}")
    st.info(
        "Check that the CSV files exist inside the project's outputs folder."
    )
    st.stop()


# --------------------------------------------------
# 3. PAGE HEADER
# --------------------------------------------------

st.title("📊 Cricket Analytics Dashboard")

st.write(
    "Explore International T20 match outcomes, target scores, "
    "venue statistics, model performance, and win-probability "
    "behaviour across different stages of a chase."
)

st.caption(
    "Historical match-level data and saved model evaluation results. "
    "Charts update according to the selected filters where applicable."
)

st.divider()


# --------------------------------------------------
# 4. FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

all_teams = sorted(
    set(matches["batting_team"].dropna().astype(str))
    | set(matches["bowling_team"].dropna().astype(str))
)

all_venues = sorted(
    matches["venue"].dropna().astype(str).unique()
)

selected_team = st.sidebar.selectbox(
    "Filter by team",
    ["All teams"] + all_teams
)

selected_venue = st.sidebar.selectbox(
    "Filter by venue",
    ["All venues"] + all_venues
)

filtered_matches = matches.copy()

if selected_team != "All teams":

    filtered_matches = filtered_matches[
        (filtered_matches["batting_team"] == selected_team)
        | (filtered_matches["bowling_team"] == selected_team)
    ]

if selected_venue != "All venues":

    filtered_matches = filtered_matches[
        filtered_matches["venue"] == selected_venue
    ]

if filtered_matches.empty:
    st.warning("No matches are available for these filters.")
    st.stop()


# --------------------------------------------------
# 5. KEY PERFORMANCE INDICATORS
# --------------------------------------------------

st.subheader("Dataset Overview")

total_matches = len(filtered_matches)

chase_win_rate = (
    filtered_matches["chasing_team_wins"].mean()
)

average_target = filtered_matches["target_runs"].mean()

unique_venues = filtered_matches["venue"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Matches analyzed",
    f"{total_matches:,}"
)

col2.metric(
    "Chase success rate",
    f"{chase_win_rate:.1%}"
)

col3.metric(
    "Average target",
    f"{average_target:.1f} runs"
)

col4.metric(
    "Venues represented",
    f"{unique_venues:,}"
)

st.caption(
    "These statistics describe matches represented in the saved "
    "match-state dataset, after the project’s eligibility filters."
)


# --------------------------------------------------
# 6. MATCH OUTCOME DISTRIBUTION
# --------------------------------------------------

st.divider()
st.subheader("Match Outcome Analysis")

left, right = st.columns(2)

with left:

    outcome_counts = (
        filtered_matches["outcome"]
        .value_counts()
        .rename_axis("Outcome")
        .reset_index(name="Matches")
    )

    outcome_fig = px.pie(
        outcome_counts,
        names="Outcome",
        values="Matches",
        hole=0.48,
        title="Chasing Team vs Defending Team Wins"
    )

    outcome_fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    st.plotly_chart(
        outcome_fig,
        use_container_width=True
    )

with right:

    target_fig = px.histogram(
        filtered_matches,
        x="target_runs",
        color="outcome",
        barmode="overlay",
        opacity=0.65,
        nbins=25,
        marginal="box",
        title="Target Score Distribution",
        labels={
            "target_runs": "Target score (runs)",
            "outcome": "Match outcome"
        }
    )

    st.plotly_chart(
        target_fig,
        use_container_width=True
    )


# --------------------------------------------------
# 7. TARGET SCORE ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("Target Score Insights")

target_analysis = filtered_matches.copy()

target_analysis["target_band"] = pd.cut(
    target_analysis["target_runs"],
    bins=[0, 100, 130, 160, 190, 220, 260, float("inf")],
    labels=[
        "1–100",
        "101–130",
        "131–160",
        "161–190",
        "191–220",
        "221–260",
        "261+"
    ],
    include_lowest=True
)

target_summary = (
    target_analysis.groupby(
        "target_band",
        observed=False
    )
    .agg(
        matches=("match_id", "nunique"),
        chase_win_rate=("chasing_team_wins", "mean"),
        average_target=("target_runs", "mean")
    )
    .reset_index()
)

target_summary["chase_win_percentage"] = (
    target_summary["chase_win_rate"] * 100
)

target_fig = px.bar(
    target_summary,
    x="target_band",
    y="chase_win_rate",
    hover_data=[
        "matches",
        "average_target"
    ],
    title="Observed Chase Success by Target Range",
    labels={
        "target_band": "Target range",
        "chase_win_rate": "Chase win rate",
        "matches": "Matches",
        "average_target": "Average target"
    }
)

target_fig.update_yaxes(tickformat=".0%", range=[0, 1])

st.plotly_chart(
    target_fig,
    use_container_width=True
)

st.caption(
    "This chart is descriptive. It does not show the model's predictions "
    "or establish that target size alone causes an outcome."
)


# --------------------------------------------------
# 8. VENUE ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("Venue Performance")

venue_summary = (
    filtered_matches.groupby("venue", as_index=False)
    .agg(
        matches=("match_id", "nunique"),
        average_target=("target_runs", "mean"),
        chase_win_rate=("chasing_team_wins", "mean")
    )
)

venue_summary["chase_win_percentage"] = (
    venue_summary["chase_win_rate"] * 100
)

minimum_matches = st.slider(
    "Minimum matches per venue",
    min_value=1,
    max_value=30,
    value=5
)

popular_venues = (
    venue_summary[
        venue_summary["matches"] >= minimum_matches
    ]
    .sort_values("matches", ascending=False)
    .head(15)
)

if popular_venues.empty:

    st.info("No venues satisfy the selected minimum match count.")

else:

    venue_fig = px.bar(
        popular_venues.sort_values(
            "matches",
            ascending=True
        ),
        x="matches",
        y="venue",
        orientation="h",
        color="chase_win_rate",
        color_continuous_scale="Viridis",
        hover_data=[
            "average_target",
            "chase_win_percentage"
        ],
        title="Most Frequently Represented Venues",
        labels={
            "matches": "Matches",
            "venue": "Venue",
            "chase_win_rate": "Chase win rate",
            "average_target": "Average target",
            "chase_win_percentage": "Chase win rate (%)"
        }
    )

    st.plotly_chart(
        venue_fig,
        use_container_width=True
    )

st.dataframe(
    venue_summary.sort_values(
        "matches",
        ascending=False
    ).round(2),
    hide_index=True,
    use_container_width=True
)


# --------------------------------------------------
# 9. MODEL COMPARISON
# --------------------------------------------------

st.divider()
st.subheader("Machine Learning Model Comparison")

st.write(
    "Four classification models were evaluated using the same "
    "chronological validation partition."
)

available_metrics = [
    metric for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "Log Loss",
        "Brier Score"
    ]
    if metric in model_comparison.columns
]

selected_metric = st.selectbox(
    "Select evaluation metric",
    available_metrics,
    index=available_metrics.index("Log Loss")
    if "Log Loss" in available_metrics else 0
)

ascending = selected_metric in ["Log Loss", "Brier Score"]

comparison_sorted = model_comparison.sort_values(
    selected_metric,
    ascending=ascending
)

model_fig = px.bar(
    comparison_sorted,
    x="Model",
    y=selected_metric,
    color="Model",
    text_auto=".4f",
    title=f"Model Comparison: {selected_metric}",
    labels={
        selected_metric: selected_metric,
        "Model": "Machine learning model"
    }
)

st.plotly_chart(
    model_fig,
    use_container_width=True
)

st.dataframe(
    comparison_sorted.round(4),
    hide_index=True,
    use_container_width=True
)

st.info(
    "XGBoost was selected because it achieved the strongest overall "
    "validation results among the four tested models, including the "
    "lowest validation log loss."
)


# --------------------------------------------------
# 10. FINAL TEST RESULTS
# --------------------------------------------------

st.divider()
st.subheader("Final Test Evaluation")

st.write(
    "The following results come from the held-out chronological "
    "test partition, which was not used for model selection."
)

test_model_options = test_results["Model"].tolist()

selected_test_model = st.selectbox(
    "Select probability version",
    test_model_options,
    index=(
        test_model_options.index("XGBoost - Raw")
        if "XGBoost - Raw" in test_model_options
        else 0
    )
)

test_row = test_results[
    test_results["Model"] == selected_test_model
].iloc[0]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{test_row['Accuracy']:.2%}")
c2.metric("ROC-AUC", f"{test_row['ROC-AUC']:.4f}")
c3.metric("Log Loss", f"{test_row['Log Loss']:.4f}")
c4.metric("Brier Score", f"{test_row['Brier Score']:.4f}")

st.dataframe(
    test_results.round(4),
    hide_index=True,
    use_container_width=True
)


# --------------------------------------------------
# 11. PHASE-WISE EVALUATION
# --------------------------------------------------

st.divider()
st.subheader("Prediction Performance Across Chase Phases")

phase_metrics = [
    metric for metric in [
        "Accuracy",
        "ROC-AUC",
        "Log Loss",
        "Brier Score"
    ]
    if metric in phase_results.columns
]

phase_metric = st.selectbox(
    "Select phase-wise evaluation metric",
    phase_metrics,
    index=phase_metrics.index("Log Loss")
    if "Log Loss" in phase_metrics else 0
)

phase_order = [
    "Start of chase",
    "After 6 overs",
    "After 12 overs",
    "After 18 overs"
]

phase_fig = px.line(
    phase_results,
    x="Phase",
    y=phase_metric,
    color="Probability version",
    markers=True,
    category_orders={"Phase": phase_order},
    hover_data=["Matches"],
    title=f"{phase_metric} by Chase Phase",
    labels={
        "Phase": "Match stage",
        phase_metric: phase_metric,
        "Probability version": "Model output"
    }
)

st.plotly_chart(
    phase_fig,
    use_container_width=True
)

st.dataframe(
    phase_results.round(4),
    hide_index=True,
    use_container_width=True
)

st.caption(
    "Later-stage evaluation includes only matches that remained in "
    "progress at that checkpoint. Therefore, the number and composition "
    "of evaluated matches differ by phase."
)


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "T20Pulse | International T20 Cricket Analytics. "
    "Historical analysis and model predictions are estimates, "
    "not guarantees of match outcomes."
)

st.markdown(
    "[Cricsheet data source](https://cricsheet.org/downloads/) "
    " · "
    "[Licensing information](https://cricsheet.org/licensing/)"
)
