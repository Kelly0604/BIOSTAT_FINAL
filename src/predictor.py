"""Module for predicting heart disease using a trained Random Forest model."""

from typing import Any, List
import joblib
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)

class HeartDiseasePredictor:
    """Class for predicting heart disease using a Random Forest model."""

    def __init__(self, model_path: str = "20_outcomes/random_forest_model.pkl") -> None:
        """Initializes the HeartDiseasePredictor class with the path to a trained model."""
        self.model_path: str = model_path
        self.model: Any = self.load_model()

    def load_model(self) -> Any:
        """Loads the machine learning model from a file."""
        try:
            model = joblib.load(self.model_path)
            logging.info("Model loaded successfully.")
            return model
        except FileNotFoundError as e:
            logging.error(f"The model file was not found at {self.model_path}.")
            raise FileNotFoundError(f"The model file was not found at {self.model_path}.") from e
        except Exception as e:
            logging.error(f"Failed to load the model due to an error: {e}")
            raise Exception(f"Failed to load the model due to an error: {e}") from e

    def predict_heart_disease(self, patient_data: List[float]) -> str:
        """Predicts whether the patient has heart disease based on the input features.
        
        Args:
            patient_data (List[float]): A list of float numbers representing patient features.

        Returns:
            str: A message indicating whether the patient is likely diagnosed with heart disease.
        """
        try:
            prediction = self.model.predict([patient_data])[0]
            return "The patient is likely diagnosed with heart disease." if prediction == 1 else "The patient is not likely diagnosed with heart disease."
        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            raise Exception(f"An error occurred during prediction: {e}") from e

if __name__ == "__main__":
    predictor = HeartDiseasePredictor("20_outcomes/random_forest_model.pkl")
    example_patient = [0.5, 1.2, 0.3, 0.6, 0.2, 0.9, 1.1, 0.4, 0.7, 0.1, 0.3, 0.8, 0.2]
    print(predictor.predict_heart_disease(example_patient))
