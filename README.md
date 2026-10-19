# T20Pulse — International T20 Win Probability

## Overview

T20Pulse is an end-to-end data science project that estimates
the probability that a team chasing a target in a men's
International T20 cricket match will win.

The project covers exploratory data analysis, feature
engineering, preprocessing, model comparison, probability
calibration, evaluation, and web application development.

## Selected Model

XGBoost Classifier.

The model was selected after comparing Logistic Regression,
Random Forest, Extra Trees, and XGBoost.

## Application Features

- Project overview and objectives
- Interactive cricket analytics dashboard
- Methodology documentation
- Machine learning model comparison
- Live win-probability prediction from manually entered
  second-innings match states

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Plotly
- Streamlit
- Joblib

## Project Structure

    app.py
    pages/
        1_Dashboard.py
        2_Methodology.py
        3_Model_Information.py
        4_Live_Prediction.py
    models/
        live_win_probability.joblib
    outputs/
        match_summary.csv
        validation_model_comparison.csv
        final_test_results.csv
        phase_wise_evaluation.csv
        project_info.json
        model_metadata.json
    requirements.txt

## Run Locally

Install the dependencies:

    pip install -r requirements.txt

Start the application from the project root:

    streamlit run app.py

## Dataset

Historical men's International T20 ball-by-ball data is
obtained from Cricsheet.

Downloads: https://cricsheet.org/downloads/
Licensing: https://cricsheet.org/licensing/

Appropriate source attribution and applicable licensing
requirements should be respected.

## Limitations

The current model focuses on standard second-innings chases.
DLS-adjusted matches and super-over outcomes are outside the
supported scope. Predictions are estimates, not guarantees.
The app currently accepts manual match-state inputs rather
than retrieving live scores automatically.

## Evaluation

See the CSV files in the outputs directory for the recorded
validation comparison, final test metrics, and phase-wise
evaluation.
