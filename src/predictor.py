"""Module for including classes in the predictor for heart disease."""

class IsHeartDisease:
  def predict_heart(self, patient):
    model = joblib.load('20_outcomes/random_forest_model.pkl')
    predictions = model.predict(patient)
