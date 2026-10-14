import pickle
import pandas as pd

def test_model():
    with open("student_model.pkl","rb") as f:
        saved_data = pickle.load(f)

    model = saved_data["model"]
    X_test_scaled = pd.read_csv("testdatascaled.csv")
    prediction = model.predict(X_test_scaled)

    assert len(prediction) == len(X_test_scaled)
    assert set(prediction) <= {0,1}