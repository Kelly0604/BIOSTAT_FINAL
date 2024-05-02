"""Module for including classes in the predictor for heart disease."""
import joblib

class IsHeartDisease:
    def __init__(self, model_path: str = '20_outcomes/random_forest_model.pkl'):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        try:
            model = joblib.load(self.model_path)
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"The model file was not found at {self.model_path}.")
        except Exception as e:
            raise Exception(f"Failed to load the model due to an error: {e}")

    def predict_heart(self, patient):
        predictions = self.model.predict([patient])
        predicted_class = predictions[0]
        if predicted_class == 1:
            return "The patient is likely diagnosed with heart disease."
        else:
            return "The patient is not likely diagnosed with heart disease."
