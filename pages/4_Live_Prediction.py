
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="T20Pulse | Live Prediction",
    page_icon="🏏",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "live_win_probability.joblib"


# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model_artifact():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    artifact = load_model_artifact()

    model = artifact["model"]
    calibrator = artifact["calibrator"]

    feature_columns = artifact["feature_columns"]
    teams = sorted(artifact["teams"])
    venues = sorted(artifact["venues"])

    max_balls = int(artifact["max_balls"])

except Exception as exc:
    st.error(f"Unable to load the saved model: {exc}")
    st.stop()


# --------------------------------------------------
# 3. PAGE HEADER
# --------------------------------------------------

st.title("🏏 Live Win Probability Predictor")

st.write("""
Enter the current situation in a standard men's International
T20 second-innings chase. T20Pulse will use the trained XGBoost
model to estimate the chasing team's probability of winning.
""")

st.info("""
This is a manual-input prediction tool, not a live-score feed.
Rain-shortened/DLS matches and super-over outcomes are outside
the current model's scope.
""")


# --------------------------------------------------
# 4. INPUT FORM
# --------------------------------------------------

st.header("Match Situation")

if len(teams) < 2:
    st.error("The saved model does not contain two or more teams.")
    st.stop()

default_batting = "India" if "India" in teams else teams[0]

default_bowling_candidates = [
    team for team in teams
    if team != default_batting
]

default_bowling = (
    "Australia"
    if "Australia" in default_bowling_candidates
    else default_bowling_candidates[0]
)

venue_options = venues if venues else ["Unknown"]

with st.form("live_prediction_form"):

    team_col1, team_col2 = st.columns(2)

    with team_col1:
        batting_team = st.selectbox(
            "Batting team (chasing)",
            teams,
            index=teams.index(default_batting)
        )

    with team_col2:
        bowling_options = [
            team for team in teams
            if team != batting_team
        ]

        default_opponent = (
            "Australia"
            if "Australia" in bowling_options
            else bowling_options[0]
        )

        bowling_team = st.selectbox(
            "Bowling team (defending)",
            bowling_options,
            index=bowling_options.index(default_opponent)
        )

    score_col1, score_col2 = st.columns(2)

    with score_col1:
        target_runs = st.number_input(
            "Target score",
            min_value=1,
            max_value=400,
            value=180,
            step=1
        )

        current_score = st.number_input(
            "Current score",
            min_value=0,
            max_value=400,
            value=95,
            step=1
        )

        wickets_lost = st.number_input(
            "Wickets lost",
            min_value=0,
            max_value=10,
            value=3,
            step=1
        )

    with score_col2:
        legal_balls_bowled = st.number_input(
            "Legal balls completed",
            min_value=0,
            max_value=max_balls,
            value=72,
            step=1,
            help=(
                "For example, 75 legal balls means "
                "12 overs and 3 balls. Wides and no-balls "
                "do not count as legal balls."
            )
        )

        venue = st.selectbox(
            "Match venue",
            venue_options
        )

        probability_mode = st.radio(
            "Probability method",
            [
                "Raw XGBoost",
                "Calibrated XGBoost"
            ],
            index=0,
            help=(
                "Raw XGBoost is the default because it performed "
                "slightly better on the complete held-out test "
                "partition using Log Loss and Brier Score."
            )
        )

    submitted = st.form_submit_button(
        "Predict Win Probability",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# 5. VALIDATE AND PREDICT
# --------------------------------------------------

if submitted:

    target_runs = int(target_runs)
    current_score = int(current_score)
    wickets_lost = int(wickets_lost)
    legal_balls_bowled = int(legal_balls_bowled)

    # Completed match states are handled separately.
    if current_score >= target_runs:

        st.success(
            f"{batting_team} have reached the target."
        )

        chase_probability = 1.0
        show_prediction = True
        terminal_state = True

    elif wickets_lost >= 10:

        st.info(
            "The chasing team is all out before reaching the target."
        )

        chase_probability = 0.0
        show_prediction = True
        terminal_state = True

    elif legal_balls_bowled >= max_balls:

        if current_score == target_runs - 1:

            st.warning(
                "The scores are tied after 20 overs. "
                "This model does not predict a super-over outcome."
            )

            show_prediction = False
            terminal_state = True

        else:

            st.info(
                "The scheduled 20 overs have been completed "
                "without the chasing team reaching the target."
            )

            chase_probability = 0.0
            show_prediction = True
            terminal_state = True

    else:

        # Derive match-state features exactly as in training.
        balls_remaining = max_balls - legal_balls_bowled

        runs_needed = max(
            target_runs - current_score,
            0
        )

        wickets_remaining = 10 - wickets_lost

        current_run_rate = (
            current_score * 6 / legal_balls_bowled
            if legal_balls_bowled > 0
            else 0.0
        )

        required_run_rate = (
            runs_needed * 6 / balls_remaining
            if balls_remaining > 0
            else 0.0
        )

        # Assemble the exact model input columns.
        match_state = {
            "target_runs": target_runs,
            "current_score": current_score,
            "wickets_lost": wickets_lost,
            "legal_balls_bowled": legal_balls_bowled,
            "balls_remaining": balls_remaining,
            "runs_needed": runs_needed,
            "wickets_remaining": wickets_remaining,
            "current_run_rate": current_run_rate,
            "required_run_rate": required_run_rate,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue
        }

        input_df = pd.DataFrame(
            [match_state]
        )[feature_columns]

        # Obtain the real model probabilities.
        raw_probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        calibrated_probability = float(
            calibrator.predict([raw_probability])[0]
        )

        raw_probability = float(
            np.clip(raw_probability, 0.0, 1.0)
        )

        calibrated_probability = float(
            np.clip(calibrated_probability, 0.0, 1.0)
        )

        if probability_mode == "Raw XGBoost":
            chase_probability = raw_probability
        else:
            chase_probability = calibrated_probability

        show_prediction = True
        terminal_state = False

    # --------------------------------------------------
    # 6. DISPLAY RESULT
    # --------------------------------------------------

    if show_prediction:

        defending_probability = 1.0 - chase_probability

        st.divider()
        st.header("Prediction Result")

        left, right = st.columns(2)

        left.metric(
            f"{batting_team} — Win Probability",
            f"{chase_probability:.1%}"
        )

        right.metric(
            f"{bowling_team} — Win Probability",
            f"{defending_probability:.1%}"
        )

        st.progress(chase_probability)

        if not terminal_state:

            st.caption(
                f"Raw XGBoost probability: {raw_probability:.1%}  |  "
                f"Calibrated probability: {calibrated_probability:.1%}"
            )

            st.subheader("Match Situation")

            remaining_balls = max_balls - legal_balls_bowled

            required_runs = max(
                target_runs - current_score,
                0
            )

            remaining_wickets = 10 - wickets_lost

            current_rr = (
                current_score * 6 / legal_balls_bowled
                if legal_balls_bowled > 0
                else 0.0
            )

            required_rr = (
                required_runs * 6 / remaining_balls
                if remaining_balls > 0
                else 0.0
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Runs Needed", required_runs)
            m2.metric("Balls Remaining", remaining_balls)
            m3.metric("Wickets Remaining", remaining_wickets)
            m4.metric("Required Run Rate", f"{required_rr:.2f}")

            st.write(
                f"**Current run rate:** {current_rr:.2f}"
            )

        st.caption(
            "Predictions are historical-data-based estimates, "
            "not guarantees. The model's reliability can vary "
            "by match situation, team, and venue."
        )


# --------------------------------------------------
# 7. MODEL DETAILS
# --------------------------------------------------

with st.expander("How does this prediction work?"):

    st.markdown("""
    1. The app accepts the current chase situation.
    2. It calculates runs needed, wickets remaining, balls remaining,
       current run rate, and required run rate.
    3. The saved preprocessing pipeline encodes categorical inputs.
    4. XGBoost estimates the probability that the chasing team wins.
    5. The selected probability version is displayed.

    The model was trained on historical men's International T20
    match states. It does not currently ingest live match feeds.
    """)
