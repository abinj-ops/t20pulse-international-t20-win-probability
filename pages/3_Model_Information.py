
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="T20Pulse | Model Information",
    page_icon="🧠",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"


# --------------------------------------------------
# 2. LOAD EVALUATION RESULTS
# --------------------------------------------------

@st.cache_data
def load_evaluation_data():

    validation = pd.read_csv(
        OUTPUTS_DIR / "validation_model_comparison.csv"
    )

    test = pd.read_csv(
        OUTPUTS_DIR / "final_test_results.csv"
    )

    phase = pd.read_csv(
        OUTPUTS_DIR / "phase_wise_evaluation.csv"
    )

    return validation, test, phase


try:
    validation_df, test_df, phase_df = load_evaluation_data()

except Exception as exc:
    st.error(f"Unable to load model evaluation files: {exc}")
    st.stop()


required_columns = {
    "Model",
    "Accuracy",
    "ROC-AUC",
    "Log Loss",
    "Brier Score"
}

if not required_columns.issubset(validation_df.columns):
    st.error("The validation results file is missing required columns.")
    st.stop()


# --------------------------------------------------
# 3. PAGE HEADER
# --------------------------------------------------

st.title("🧠 Machine Learning Model Information")

st.write(
    "Explore the machine learning algorithms tested in T20Pulse, "
    "their validation performance, the model selection process, "
    "and the final test results."
)

st.info(
    "XGBoost was selected using validation results. "
    "The final test data was kept separate from model selection "
    "and probability calibration."
)

st.divider()


# --------------------------------------------------
# 4. PURPOSE OF THE MODEL
# --------------------------------------------------

st.header("1. Prediction Objective")

st.write("""
The primary task is binary classification.

The model receives a snapshot of an International T20
second-innings chase and estimates the probability that
the batting team will eventually win.
""")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Model inputs")

    st.markdown("""
    - Target score
    - Current score
    - Wickets lost
    - Legal balls completed
    - Balls remaining
    - Runs needed
    - Wickets remaining
    - Current run rate
    - Required run rate
    - Batting team
    - Bowling team
    - Venue
    """)

with c2:
    st.subheader("Prediction target")

    st.markdown("""
    **`chasing_team_wins`**

    - `1`: Chasing team eventually wins.
    - `0`: Defending team eventually wins.

    The classifier estimates the probability of class `1`.
    The probability for the defending team is displayed as
    the complement of the chasing team's probability.
    """)

st.divider()


# --------------------------------------------------
# 5. ALGORITHMS EVALUATED
# --------------------------------------------------

st.header("2. Algorithms Evaluated")

st.write(
    "Four classification algorithms were trained using the same "
    "training period and evaluated on the same chronological "
    "validation partition."
)

model_details = [
    {
        "name": "Logistic Regression",
        "type": "Linear classification",
        "description": (
            "A baseline classifier that models the relationship "
            "between input features and the probability of a binary "
            "outcome. Numerical features were standardized."
        ),
        "strength": (
            "Fast to train and provides a useful baseline "
            "for comparison."
        ),
        "limitation": (
            "May not capture complex nonlinear relationships "
            "between runs required, wickets, and balls remaining."
        )
    },
    {
        "name": "Random Forest",
        "type": "Bagged decision trees",
        "description": (
            "Combines predictions from multiple decision trees "
            "trained using randomized samples and feature choices."
        ),
        "strength": (
            "Captures nonlinear relationships and interactions "
            "between match-state features."
        ),
        "limitation": (
            "May be less effective than well-tuned boosting "
            "models on some structured datasets."
        )
    },
    {
        "name": "Extra Trees",
        "type": "Randomized tree ensemble",
        "description": (
            "Uses an ensemble of trees with additional randomness "
            "in the selection of split thresholds."
        ),
        "strength": (
            "Provides a useful comparison against Random Forest "
            "with a different tree-randomization strategy."
        ),
        "limitation": (
            "Its probability estimates may require evaluation "
            "and calibration for the intended application."
        )
    },
    {
        "name": "XGBoost",
        "type": "Gradient-boosted decision trees",
        "description": (
            "Builds trees sequentially, with later trees helping "
            "to correct errors made by the preceding ensemble."
        ),
        "strength": (
            "Can learn complex nonlinear patterns and interactions "
            "in structured match-state data."
        ),
        "limitation": (
            "Requires appropriate parameter choices and careful "
            "evaluation to avoid overfitting."
        )
    }
]

for start in range(0, len(model_details), 2):

    col1, col2 = st.columns(2)

    for column, model_info in zip(
        [col1, col2],
        model_details[start:start + 2]
    ):

        with column:

            with st.container(border=True):

                st.subheader(model_info["name"])

                st.caption(model_info["type"])

                st.write(model_info["description"])

                st.markdown("**Strength**")
                st.write(model_info["strength"])

                st.markdown("**Limitation**")
                st.write(model_info["limitation"])


st.divider()


# --------------------------------------------------
# 6. VALIDATION MODEL COMPARISON
# --------------------------------------------------

st.header("3. Validation Performance Comparison")

st.write("""
The validation dataset was used to compare models and choose
the leading candidate before probability calibration and
final testing.
""")

metric_options = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "Log Loss",
    "Brier Score"
]

available_metrics = [
    metric for metric in metric_options
    if metric in validation_df.columns
]

selected_metric = st.selectbox(
    "Choose a metric to compare",
    available_metrics,
    index=available_metrics.index("Log Loss")
    if "Log Loss" in available_metrics else 0
)

lower_is_better = selected_metric in [
    "Log Loss",
    "Brier Score"
]

sorted_validation = validation_df.sort_values(
    selected_metric,
    ascending=lower_is_better
)

fig = px.bar(
    sorted_validation,
    x="Model",
    y=selected_metric,
    color="Model",
    text_auto=".4f",
    title=f"Validation Comparison — {selected_metric}",
    labels={
        "Model": "Machine learning algorithm",
        selected_metric: selected_metric
    }
)

fig.update_layout(showlegend=False)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    sorted_validation.round(4),
    hide_index=True,
    use_container_width=True
)


# --------------------------------------------------
# 7. WHY XGBOOST WAS SELECTED
# --------------------------------------------------

st.header("4. Selected Model: XGBoost")

xgb_validation = validation_df[
    validation_df["Model"] == "XGBoost"
]

if not xgb_validation.empty:

    xgb = xgb_validation.iloc[0]

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Validation Accuracy",
        f"{float(xgb['Accuracy']):.2%}"
    )

    m2.metric(
        "Validation ROC-AUC",
        f"{float(xgb['ROC-AUC']):.4f}"
    )

    m3.metric(
        "Validation Log Loss",
        f"{float(xgb['Log Loss']):.4f}"
    )

    m4.metric(
        "Validation Brier Score",
        f"{float(xgb['Brier Score']):.4f}"
    )

st.markdown("""
**Selection rationale**

XGBoost was selected because it achieved the best validation
results among the four tested candidates on the project's
primary probability metrics, Log Loss and Brier Score.
It also achieved the highest validation accuracy and ROC-AUC
in the recorded comparison.

This makes it the leading model for the current version
of T20Pulse; it does not guarantee superior performance on
every future match.
""")

st.divider()


# --------------------------------------------------
# 8. FINAL TEST RESULTS
# --------------------------------------------------

st.header("5. Final Test Results")

st.write("""
After model selection, the selected XGBoost model was evaluated
on the separate chronological test partition. Its probabilities
were also evaluated before and after isotonic calibration.
""")

st.dataframe(
    test_df.round(4),
    hide_index=True,
    use_container_width=True
)

raw_result = test_df[
    test_df["Model"] == "XGBoost - Raw"
]

if not raw_result.empty:

    raw = raw_result.iloc[0]

    st.subheader("Raw XGBoost")

    a, b, c, d = st.columns(4)

    a.metric("Accuracy", f"{float(raw['Accuracy']):.2%}")
    b.metric("ROC-AUC", f"{float(raw['ROC-AUC']):.4f}")
    c.metric("Log Loss", f"{float(raw['Log Loss']):.4f}")
    d.metric("Brier Score", f"{float(raw['Brier Score']):.4f}")


# --------------------------------------------------
# 9. PROBABILITY CALIBRATION
# --------------------------------------------------

st.divider()

st.header("6. Probability Calibration")

st.write("""
The project fitted an isotonic regression calibrator using
predictions from the dedicated calibration period. The final
test set was not used to fit the calibrator.
""")

st.markdown("""
Calibration attempts to improve the correspondence between
predicted probabilities and observed outcome frequencies.

A model can rank winning situations correctly without producing
well-calibrated probabilities. Therefore, ROC-AUC alone is not
enough for a live win-probability application.
""")

st.markdown("""
**Observed project result:** On the full final test partition,
raw XGBoost achieved a Log Loss of 0.2784 and a Brier Score of
0.0883. The calibrated version achieved 0.2816 and 0.0895,
respectively. The raw version therefore remains the default
for this project.
""")

if not phase_df.empty:

    phase_metric_options = [
        metric for metric in [
            "Accuracy",
            "ROC-AUC",
            "Log Loss",
            "Brier Score"
        ]
        if metric in phase_df.columns
    ]

