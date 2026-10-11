# Student Pass/Fail Predictor

A small web app that estimates whether a secondary-school student is likely to **pass or fail** a course, built with **scikit-learn** and **FastAPI**.

The user fills in a form with background information about a student (family, study habits, free time, absences, and the first-period grade). The app returns a Pass/Fail prediction together with a probability.

## How it works

- **Data:** [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (Mathematics course, 395 students).
- **Target:** `Passing Status` = 1 if the final grade `G3 >= 10`, otherwise 0 (about 67% pass / 33% fail).
- **Features:** 31 inputs (demographics, family, school, lifestyle, absences) plus the first-period grade `G1`.
- **Why `G1` is included but `G2` is not:** `G2` is too close to the final grade and makes prediction almost trivial. Using only `G1` keeps this an *early risk estimate* that could be used early in the term.
- **Preprocessing:** `OneHotEncoder` for categorical columns, `StandardScaler` for the final feature matrix.
- **Model:** Logistic Regression, chosen after comparing 9 classifiers with 5-fold stratified cross-validation (metric: F1 score of the *fail* class).

### Key findings

| Scenario | What happened |
|---|---|
| No previous grades | Accuracy about 0.68, barely above the 0.67 baseline. Background data alone is a weak predictor. |
| With `G1` | Accuracy about 0.73-0.80 and much better detection of failing students. |
| 5-fold CV (with `G1`) | Logistic Regression had the best and most stable result (mean F1 of the fail class about 0.76). |

The test set is small (79 students, about 26 failing), so single-split results are noisy. Cross-validation was used to compare models.

## Project structure

```
.
├── app.py               # FastAPI application
├── student_model.pkl    # Saved model + encoder + scaler
├── model_test.py        # Quick script to test the saved model
├── testdatascaled.csv   # Scaled test data used by model_test.py
├── notebooks/           # EDA and model training notebooks
├── static/              # CSS / JS
├── templates/           # HTML templates (index.html)
└── requirements.txt
```

## Run locally

```bash
# 1. Clone the repository
git clone https://github.com/haticeikrakalkan/FastAPI-Student-Predictor.git
cd FastAPI-Student-Predictor

# 2. (Optional) create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app:app --reload
```

Then open <http://127.0.0.1:8000> in your browser.
Interactive API docs are available at <http://127.0.0.1:8000/docs>.

> **Note:** `student_model.pkl` was saved with **scikit-learn 1.7.2**. If you see version warnings or errors, install that exact version: `pip install scikit-learn==1.7.2`.

## API

### `POST /predict`

Takes a JSON body with the 31 student features and returns:

```json
{
  "prediction": 1,
  "passed": true,
  "label": "Pass",
  "probability": 0.78
}
```

## Limitations

- Small dataset (395 students) from two Portuguese schools, so the model may not generalize to other schools or countries.
- Predictions are statistical estimates, not a judgment about any individual student. It should not be used for real decisions about students.

## Tech stack

Python, pandas, scikit-learn, FastAPI, Uvicorn, Jinja2
