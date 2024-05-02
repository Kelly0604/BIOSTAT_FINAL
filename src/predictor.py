"""Module for including classes in the predictor for heart disease."""

from typing import Any, List

import joblib


class IsHeartDisease:
    """Class for predicting heart disease."""

    def __init__(
        self, model_path: str = "20_outcomes/random_forest_model.pkl"
    ) -> None:
        """Initialize the IsHeartDisease class."""
        self.model_path: str = model_path
        self.model: Any = self.load_model()

    def load_model(self) -> Any:
        """Load the machine learning model."""
        try:
            model = joblib.load(self.model_path)
            return model
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"The model file was not found at {self.model_path}."
            ) from e
        except Exception as e:
            raise Exception(
                f"Failed to load the model due to an error: {e}"
            ) from e

    def predict_heart(self, patient: List[float]) -> str:
        """Predict if the patient has heart disease."""
        predictions = self.model.predict([patient])
        predicted_class = predictions[0]
        if predicted_class == 1:
            return "The patient is likely diagnosed with heart disease."
        else:
            return "The patient is not likely diagnosed with heart disease."
