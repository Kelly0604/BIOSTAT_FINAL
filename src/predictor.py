"""Module for including classes in the predictor for heart disease."""

class IsHeartDisease:
  def predict_heart(self, patient):
    model = joblib.load('20_outcomes/random_forest_model.pkl')
    predictions = model.predict(patient)
    predicted_class = predictions[0]
    if predicted_class == 1:
            return "The patient is likely diagnosed of heart disease."
        else:
            return "The patient is not likely diagnosed of heart disease."
