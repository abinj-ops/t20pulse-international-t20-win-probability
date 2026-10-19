
from pathlib import Path
import json

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="T20Pulse | Home",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"


# --------------------------------------------------
# 2. LOAD PROJECT INFORMATION
# --------------------------------------------------

def load_json_file(file_path):
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


project_info = load_json_file(
    OUTPUTS_DIR / "project_info.json"
)

try:
    match_data = pd.read_csv(
        OUTPUTS_DIR / "match_summary.csv"
    )
except (FileNotFoundError, pd.errors.EmptyDataError):
    match_data = pd.DataFrame()

try:
    test_results = pd.read_csv(
        OUTPUTS_DIR / "final_test_results.csv"
    )
except (FileNotFoundError, pd.errors.EmptyDataError):
    test_results = pd.DataFrame()


# --------------------------------------------------
# 3. PAGE STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .hero {
        padding: 2rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(
            120deg,
            #10251F 0%,
            #164B38 55%,
            #237653 100%
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #E0F2E9;
        font-size: 1.12rem;
        line-height: 1.7;
    }

    .section-card {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 12px;
        padding: 1.15rem;
        margin-bottom: 0.6rem;
        min-height: 140px;
    }

    .section-card h4 {
        margin-top: 0;
    }

    .small-note {
        color: #9CA3AF;
        font-size: 0.86rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

st.sidebar.title("🏏 T20Pulse")

st.sidebar.write(
    "International T20 Cricket Analytics"
)

st.sidebar.divider()

st.sidebar.markdown("### Project Navigation")
st.sidebar.write(
    "Select a page from the navigation menu above "
    "to explore the dashboard, methodology, models, "
    "and prediction system."
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning • Cricket Analytics • Python"
)


# --------------------------------------------------
# 5. HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🏏 T20Pulse</h1>
    <p>
        International T20 Live Win Probability Prediction
    </p>
    <p>
        Transforming historical cricket data into
        meaningful match insights using exploratory data
        analysis, feature engineering, and machine learning.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 6. PROJECT OVERVIEW
# --------------------------------------------------

st.header("Project Overview")

st.write("""
T20Pulse is a code-first data science project designed to
estimate the probability that a team chasing a target in
a men's International T20 match will eventually win.

The system analyzes historical ball-by-ball match data,
constructs snapshots of the game at different stages,
and trains machine learning models to learn patterns
associated with successful and unsuccessful chases.

Users can enter a match situation—including the target,
current score, wickets lost, and balls completed—and
receive a model-generated win-probability estimate.
""")

st.info(
    "This version predicts the outcome of a second-innings "
    "chase. It does not ingest live cricket feeds automatically; "
    "users enter the match situation manually."
)


# --------------------------------------------------
# 7. PROJECT STATISTICS
# --------------------------------------------------

st.header("Project at a Glance")

total_matches = project_info.get(
    "dataset_matches",
    len(match_data)
)

total_states = sum([
    project_info.get("training_state_rows", 0),
    project_info.get("validation_state_rows", 0),
    project_info.get("calibration_state_rows", 0),
    project_info.get("test_state_rows", 0)
])

test_matches = project_info.get("test_matches", 0)

raw_test = None

if not test_results.empty and "Model" in test_results.columns:
    raw_rows = test_results[
        test_results["Model"] == "XGBoost - Raw"
    ]

    if not raw_rows.empty:
        raw_test = raw_rows.iloc[0]

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Matches analyzed",
    f"{total_matches:,}"
)

m2.metric(
    "Match-state rows",
    f"{total_states:,}"
)

m3.metric(
    "Selected model",
    project_info.get("selected_model", "XGBoost")
)

if raw_test is not None:
    m4.metric(
        "Held-out test accuracy",
        f"{float(raw_test['Accuracy']):.2%}"
    )
else:
    m4.metric(
        "Held-out matches",
        f"{test_matches:,}"
    )

st.caption(
    "Test accuracy is calculated across match-state rows "
    "from held-out matches, not as a percentage of matches "
    "predicted correctly in a single match-level test."
)


# --------------------------------------------------
# 8. PROBLEM STATEMENT
# --------------------------------------------------

st.header("Problem Statement")

st.write("""
Cricket match outcomes depend on several interacting factors,
including runs required, wickets remaining, the number of
deliveries left, scoring rate, opposition, and venue.

A match scorecard describes what has happened, but it does not
directly quantify how the current situation relates to the
eventual result.

The objective is to build a data-driven system that learns
from historical International T20 matches and estimates the
chasing team's likelihood of winning from its current state.
""")

st.divider()


# --------------------------------------------------
# 9. AIM AND OBJECTIVES
# --------------------------------------------------

st.header("Aim of the Project")

st.write("""
To develop an interactive machine learning application
that estimates the live win probability of the chasing
team in International T20 cricket using historical
match data and match-state features.
""")

st.header("Project Objectives")

objectives = [
    (
        "Data collection",
        "Obtain historical men's T20 International "
        "ball-by-ball records from Cricsheet."
    ),
    (
        "Exploratory data analysis",
        "Analyze target scores, chase outcomes, wickets, "
        "and match situations through visualizations."
    ),
    (
        "Data preprocessing",
        "Construct match-state features, encode team and "
        "venue categories, and scale numerical variables "
        "where required."
    ),
    (
        "Model comparison",
        "Train and compare Logistic Regression, Random "
        "Forest, Extra Trees, and XGBoost."
    ),
    (
        "Probability evaluation",
        "Evaluate models using ROC-AUC, Log Loss, Brier "
        "Score, and other classification metrics."
    ),
    (
        "Application development",
        "Provide an interface for entering chase details "
        "and displaying the model's estimated win probability."
    )
]

for title, description in objectives:
    st.markdown(f"**{title}**")
    st.write(description)


# --------------------------------------------------
# 10. TECHNOLOGY STACK
# --------------------------------------------------

st.header("Technology Stack")

technology = pd.DataFrame({
    "Component": [
        "Programming language",
        "Data processing",
        "Data visualization",
        "Machine learning",
        "Probability calibration",
        "Web application",
        "Model persistence",
        "Development environment"
    ],
    "Technology": [
        "Python",
        "Pandas, NumPy",
        "Matplotlib, Seaborn",
        "Scikit-learn, XGBoost",
        "Isotonic Regression",
        "Streamlit",
        "Joblib",
        "Google Colab"
    ]
})

st.dataframe(
    technology,
    hide_index=True,
    use_container_width=True
)


# --------------------------------------------------
# 11. PROJECT WORKFLOW
# --------------------------------------------------

st.header("Project Workflow")

workflow = [
    "01  Historical data collection",
    "02  Exploratory data analysis",
    "03  Match-state feature engineering",
    "04  Encoding and scaling",
    "05  Model training and comparison",
    "06  Probability calibration",
    "07  Final test evaluation",
    "08  Interactive prediction application"
]

for item in workflow:
    st.markdown(f"- {item}")


# --------------------------------------------------
# 12. PRELIMINARY MODEL FINDINGS
# --------------------------------------------------

st.header("Preliminary Model Findings")

if raw_test is not None:

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Test ROC-AUC",
        f"{float(raw_test['ROC-AUC']):.4f}"
    )

    c2.metric(
        "Test Log Loss",
        f"{float(raw_test['Log Loss']):.4f}"
    )

    c3.metric(
        "Test Brier Score",
        f"{float(raw_test['Brier Score']):.4f}"
    )

    st.write("""
    XGBoost was selected after comparison against
    Logistic Regression, Random Forest, and Extra Trees
    using the validation dataset. The raw XGBoost
    probabilities currently provide the preferred
    overall test results.
    """)

else:
    st.write(
        "Model evaluation results will be displayed "
        "once the saved evaluation report is available."
    )


# --------------------------------------------------
# 13. SCOPE AND LIMITATIONS
# --------------------------------------------------

st.header("Scope and Limitations")

st.write("""
- The model focuses on men's International T20 matches.
- The current model uses manually supplied match states.
- The first version focuses on standard second-innings chases.
- DLS-adjusted matches and super-over outcomes are outside
  the model's supported scope.
- Predictions are statistical estimates, not guarantees.
- Match-state rows from the same match are correlated;
  evaluation metrics should therefore be interpreted
  with that limitation in mind.
""")


# --------------------------------------------------
# 14. DATA SOURCE AND FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
**Data source:** Cricsheet — historical cricket data

[Cricsheet Downloads](https://cricsheet.org/downloads/)
&nbsp; • &nbsp;
[Cricsheet Licensing](https://cricsheet.org/licensing/)
""")

st.markdown(
    '<p class="small-note">'
    'T20Pulse is an educational data science project and '
    'is not affiliated with or endorsed by Cricsheet or '
    'any cricket governing body.'
    '</p>',
    unsafe_allow_html=True
)
