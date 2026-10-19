
from pathlib import Path

import json
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="T20Pulse | Methodology",
    page_icon="⚙️",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"


# --------------------------------------------------
# 2. LOAD SAVED PROJECT INFORMATION
# --------------------------------------------------

def load_json(path):
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


project_info = load_json(
    OUTPUTS_DIR / "project_info.json"
)

try:
    model_comparison = pd.read_csv(
        OUTPUTS_DIR / "validation_model_comparison.csv"
    )
except (FileNotFoundError, pd.errors.EmptyDataError):
    model_comparison = pd.DataFrame()

try:
    test_results = pd.read_csv(
        OUTPUTS_DIR / "final_test_results.csv"
    )
except (FileNotFoundError, pd.errors.EmptyDataError):
    test_results = pd.DataFrame()


# --------------------------------------------------
# 3. PAGE HEADER
# --------------------------------------------------

st.title("⚙️ Project Methodology")

st.write(
    "This page documents the end-to-end methodology used "
    "to build T20Pulse, from historical data collection "
    "to live win-probability prediction."
)

st.info(
    "The current model predicts the outcome of a standard "
    "men's International T20 second-innings chase. It does "
    "not directly ingest live match feeds."
)


# --------------------------------------------------
# 4. METHODOLOGY OVERVIEW
# --------------------------------------------------

st.header("Methodology Overview")

st.markdown("""
The project follows an end-to-end data science workflow:

**Data Collection → EDA → Feature Engineering → Preprocessing
→ Train/Validation Split → Model Training → Calibration
→ Final Evaluation → Streamlit Application**
""")

st.divider()


# --------------------------------------------------
# 5. DATA COLLECTION
# --------------------------------------------------

st.header("1. Data Collection")

st.markdown("""
**Data source:** Cricsheet

The project uses historical men's International T20
ball-by-ball match records downloaded in JSON format.

Each match file contains match metadata, innings,
overs, deliveries, and the recorded result.
""")

st.markdown("**Main data fields used:**")

data_fields = pd.DataFrame({
    "Field or section": [
        "info.dates",
        "info.teams",
        "info.venue",
        "info.outcome.winner",
        "innings[].team",
        "innings[].overs",
        "deliveries[].runs.total",
        "deliveries[].extras",
        "deliveries[].wickets"
    ],
    "Purpose": [
        "Chronological data splitting",
        "Identify participating teams",
        "Venue feature",
        "Derive final outcome label",
        "Identify batting innings",
        "Iterate over match deliveries",
        "Calculate cumulative runs",
        "Determine whether a delivery is legal",
        "Count wickets lost"
    ]
})

st.dataframe(
    data_fields,
    hide_index=True,
    use_container_width=True
)

st.markdown("""
The downloaded archive contained 3,627 match files.
The initial filtering stage retained 3,371 candidate matches.
Additional checks on target metadata and special scoring
situations resulted in a final dataset of 3,224 matches.
""")

st.markdown(
    "[Cricsheet downloads](https://cricsheet.org/downloads/) "
    " · "
    "[JSON format documentation](https://cricsheet.org/format/json/) "
    " · "
    "[Licensing](https://cricsheet.org/licensing/)"
)

st.divider()


# --------------------------------------------------
# 6. DATA FILTERING
# --------------------------------------------------

st.header("2. Match Filtering and Data Quality")

st.markdown("""
The first version of the project focuses on standard
20-over men's International T20 matches with an ordinary,
recorded result and exactly two innings.

The implementation excludes:
""")

st.markdown("""
- Matches with D/L or other special outcome methods.
- Ties, no-results, and awarded or otherwise special results.
- Matches with super overs or non-standard innings accounting.
- Matches with missing, inconsistent, or non-standard chase targets.
- Cases involving selected penalty-run or miscounted-over records.
""")

st.warning(
    "These filters define the model's supported scope. "
    "They do not cover every possible International T20 match situation."
)

st.divider()


# --------------------------------------------------
# 7. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

st.header("3. Exploratory Data Analysis")

st.markdown("""
EDA was performed before training to understand the
characteristics of the matches represented in the dataset.
""")

st.markdown("""
**Analyses performed:**

- Target-score distributions by match outcome.
- Number of chasing-team wins versus defending-team wins.
- Observed chase-win rates at different innings checkpoints.
- Relationships between runs needed and wickets remaining.
- Match and venue representation.
- Distribution and consistency of engineered features.
""")

st.markdown("**Example EDA finding from this dataset:**")

eda_left, eda_right = st.columns(2)

eda_left.metric(
    "Mean target in successful chases",
    "128.05 runs"
)

eda_right.metric(
    "Mean target in unsuccessful chases",
    "175.18 runs"
)

st.caption(
    "These figures describe historical associations in the "
    "eligible dataset. They do not establish that target size "
    "alone determines the match outcome."
)

st.divider()


# --------------------------------------------------
# 8. FEATURE ENGINEERING
# --------------------------------------------------

st.header("4. Match-State Feature Engineering")

st.markdown("""
Raw delivery records were transformed into repeated
snapshots of the second innings. Each snapshot represents
the match state before a particular delivery.
""")

st.markdown("""
The score, wickets, legal-ball count, and run-rate features
are updated as deliveries are processed chronologically.
""")

feature_table = pd.DataFrame({
    "Feature": [
        "target_runs",
        "current_score",
        "wickets_lost",
        "legal_balls_bowled",
        "balls_remaining",
        "runs_needed",
        "wickets_remaining",
        "current_run_rate",
        "required_run_rate",
        "batting_team",
        "bowling_team",
        "venue"
    ],
    "Description": [
        "Target score for the chase",
        "Cumulative runs scored",
        "Wickets lost by the chasing team",
        "Legal deliveries completed",
        "120 minus legal balls completed",
        "Target minus current score, bounded at zero",
        "Ten minus wickets lost",
        "Current score per six legal balls",
        "Runs needed per six balls remaining",
        "Team batting in the second innings",
        "Team defending the target",
        "Match venue"
    ]
})

st.dataframe(
    feature_table,
    hide_index=True,
    use_container_width=True
)

st.markdown("**Feature calculations**")

st.latex(
    r"\text{Balls Remaining} = 120 - \text{Legal Balls Bowled}"
)

st.latex(
    r"\text{Runs Needed} = \max(\text{Target} - \text{Current Score}, 0)"
)

st.latex(
    r"\text{Current Run Rate} = "
    r"\frac{6 \times \text{Current Score}}{\text{Legal Balls Bowled}}"
)

st.latex(
    r"\text{Required Run Rate} = "
    r"\frac{6 \times \text{Runs Needed}}{\text{Balls Remaining}}"
)

st.markdown("""
The final target variable is `chasing_team_wins`:

- `1`: the chasing team eventually won.
- `0`: the defending team eventually won.

The final result is used as the label and not as a model input.
""")

st.divider()


# --------------------------------------------------
# 9. PREPROCESSING
# --------------------------------------------------

st.header("5. Data Preprocessing")

st.markdown("""
The twelve model input features consist of nine numerical
features and three categorical features.
""")

pre_left, pre_right = st.columns(2)

with pre_left:
    st.subheader("Numerical features")

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
    """)

with pre_right:
    st.subheader("Categorical features")

    st.markdown("""
    - Batting team
    - Bowling team
    - Venue
    """)

st.markdown("**Preprocessing techniques**")

preprocessing_table = pd.DataFrame({
    "Technique": [
        "Median imputation",
        "Most-frequent imputation",
        "One-hot encoding",
        "Standard scaling"
    ],
    "Purpose": [
        "Handle missing numeric values",
        "Handle missing categorical values",
        "Convert team and venue labels to numeric features",
        "Standardize numerical features for Logistic Regression"
    ]
})

st.dataframe(
    preprocessing_table,
    hide_index=True,
    use_container_width=True
)

st.markdown("""
Preprocessing is implemented using Scikit-learn pipelines
and a ColumnTransformer. The transformations are fitted
on training data and reused on subsequent partitions.
Random Forest, Extra Trees, and XGBoost use the same
categorical encoding, without requiring numerical scaling.
""")

st.divider()


# --------------------------------------------------
# 10. TRAIN-VALIDATION-CALIBRATION-TEST SPLIT
# --------------------------------------------------

st.header("6. Chronological Data Splitting")

st.markdown("""
Distinct match dates were sorted chronologically.
The partitions were created using the earlier dates for
training and progressively later dates for validation,
calibration, and final testing.

All match states from a given match remain in the same
partition.
""")

split_data = [
    {
        "Partition": "Training",
        "Percentage": "Approximately 70%",
        "Matches": project_info.get("training_matches", 0),
        "Rows": project_info.get("training_state_rows", 0),
        "Purpose": "Fit preprocessing and models"
    },
    {
        "Partition": "Validation",
        "Percentage": "Approximately 10%",
        "Matches": project_info.get("validation_matches", 0),
        "Rows": project_info.get("validation_state_rows", 0),
        "Purpose": "Compare model performance"
    },
    {
        "Partition": "Calibration",
        "Percentage": "Approximately 10%",
        "Matches": project_info.get("calibration_matches", 0),
        "Rows": project_info.get("calibration_state_rows", 0),
        "Purpose": "Fit probability calibrator"
    },
    {
        "Partition": "Final test",
        "Percentage": "Approximately 10%",
        "Matches": project_info.get("test_matches", 0),
        "Rows": project_info.get("test_state_rows", 0),
        "Purpose": "Final evaluation"
    }
]

st.dataframe(
    pd.DataFrame(split_data),
    hide_index=True,
    use_container_width=True
)

st.success(
    "The project verified zero overlapping match IDs "
    "between all four data partitions."
)

st.divider()


# --------------------------------------------------
# 11. MODEL TRAINING
# --------------------------------------------------

st.header("7. Model Training and Comparison")

st.markdown("""
Four classification algorithms were trained using the
training partition and compared using the same validation
partition.
""")

model_descriptions = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "XGBoost"
    ],
    "Role": [
        "Baseline linear classifier",
        "Ensemble of randomized decision trees",
        "Ensemble with randomized split selection",
        "Gradient-boosted decision trees"
    ]
})

st.dataframe(
    model_descriptions,
    hide_index=True,
    use_container_width=True
)

if not model_comparison.empty:

    st.subheader("Validation Results")

    display_columns = [
        column
        for column in [
            "Model",
            "Accuracy",
            "ROC-AUC",
            "Log Loss",
            "Brier Score"
        ]
        if column in model_comparison.columns
    ]

    st.dataframe(
        model_comparison[display_columns]
        .sort_values("Log Loss")
        .round(4),
        hide_index=True,
        use_container_width=True
    )

st.markdown("""
XGBoost was selected because it achieved the strongest
overall validation metrics among the four tested candidates,
including the lowest validation log loss and Brier score.
""")

st.divider()


# --------------------------------------------------
# 12. PROBABILITY CALIBRATION
# --------------------------------------------------

st.header("8. Probability Calibration")

st.markdown("""
An isotonic regression calibrator was fitted using
probabilities produced on the separate calibration partition.

The purpose is to adjust the model's raw probability estimates
to improve agreement between predicted probabilities and
observed frequencies.
""")

st.markdown("""
Calibration is evaluated separately on the final test data.
The calibrated output is retained as an experimental option
because it does not improve every metric in the current
test results.
""")

st.divider()


# --------------------------------------------------
# 13. FINAL EVALUATION
# --------------------------------------------------

st.header("9. Final Model Evaluation")

st.markdown("""
The final test partition was not used for model selection or
fitting the probability calibrator. It contains match states
from later matches that were held out during development.
""")

if not test_results.empty:

    st.dataframe(
        test_results.round(4),
        hide_index=True,
        use_container_width=True
    )

    raw_rows = test_results[
        test_results["Model"] == "XGBoost - Raw"
    ]

    if not raw_rows.empty:

        raw = raw_rows.iloc[0]

        metric_1, metric_2, metric_3 = st.columns(3)

        metric_1.metric(
            "Raw XGBoost accuracy",
            f"{float(raw['Accuracy']):.2%}"
        )

        metric_2.metric(
            "Raw XGBoost ROC-AUC",
            f"{float(raw['ROC-AUC']):.4f}"
        )

        metric_3.metric(
            "Raw XGBoost log loss",
            f"{float(raw['Log Loss']):.4f}"
        )

st.markdown("""
The primary probability metrics are log loss and Brier score.
ROC-AUC measures ranking quality, while accuracy and F1-score
measure classification performance at the chosen threshold.
""")

st.divider()


# --------------------------------------------------
# 14. LIMITATIONS
# --------------------------------------------------

st.header("10. Methodological Limitations")

st.markdown("""
- The application supports standard men's T20 International
  chases represented in the eligible historical data.
- Rain-shortened and DLS-adjusted games are not modeled in
  the current version.
- The initial model does not include batter-specific,
  bowler-specific, weather, or pitch-condition features.
- Multiple match-state rows from the same match are
  correlated. State-level scores should not be interpreted
  as if every row were an independent match.
- Historical performance does not guarantee accurate
  predictions on every future match.
- The model requires the user to provide the match state;
  automatic live-score integration is not included.
""")

st.divider()

st.caption(
    "T20Pulse | Methodology and model evaluation documentation"
)
