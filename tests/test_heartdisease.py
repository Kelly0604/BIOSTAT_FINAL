"""Unit tests for the HeartDiseaseClassifier."""


import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from heart_disease import HeartDiseaseClassifier


class TestHeartDiseaseClassifier(unittest.TestCase):
   """Test cases for the HeartDiseaseClassifier class."""


   def test_data_loading_and_preparation(self) -> None:
       """Test that data is loaded and prepared correctly."""
       with patch("pandas.read_csv") as mocked_read_csv:
           mocked_read_csv.return_value = pd.DataFrame(
               {
                   "HeartDisease": np.random.randint(0, 2, size=100),
                   **{f"Feature{i}": np.random.rand(100) for i in range(40)},
               }
           )
           classifier = HeartDiseaseClassifier()
           data = "00_data/preprocessed_data.csv"
           X_train, X_test, y_train, y_test = (
               classifier.load_and_prepare_data(data)
           )


           self.assertTrue(X_train.shape[0] > 0 and X_test.shape[0] > 0)
           self.assertTrue(y_train.shape[0] > 0 and y_test.shape[0] > 0)
           self.assertEqual(
               X_train.shape[1],
               X_test.shape[1],
               "Features count mismatch between training and testing sets",
           )


   def test_model_training(self) -> None:
       """Test the model is trained and classifier has a model."""
       with patch("builtins.input", return_value="yes"), patch(
           "heart_disease.HeartDiseaseClassifier.load_and_prepare_data",
           return_value=(
               np.array([[0] * 40] * 80),
               np.array([[0] * 40] * 20),
               np.array([0] * 80),
               np.array([0] * 20),
           ),
       ), patch("sklearn.ensemble.RandomForestClassifier.fit"):
           classifier = HeartDiseaseClassifier()
           classifier.train_model()
           self.assertIsNotNone(classifier.model)


   def test_prediction(self) -> None:
       """Test the prediction output format and content."""
       with patch("builtins.input", return_value="yes"), patch(
           "heart_disease.HeartDiseaseClassifier.load_and_prepare_data",
           return_value=(
               np.array([[0] * 40] * 80),
               np.array([[0] * 40] * 20),
               np.array([0] * 80),
               np.array([0] * 20),
           ),
       ), patch(
           "sklearn.ensemble.RandomForestClassifier.predict_proba",
           return_value=np.array([[0.3, 0.7]]),
       ):
           classifier = HeartDiseaseClassifier()
           classifier.train_model()


           feature_names = [
               "Sex_0",
               "Sex_1",
               "ChestPainType_0",
               "ChestPainType_1",
               "ChestPainType_2",
               "ChestPainType_3",
               "FastingBS_0",
               "FastingBS_1",
               "RestingECG_0",
               "RestingECG_1",
               "RestingECG_2",
               "ExerciseAngina_0",
               "ExerciseAngina_1",
               "ST_Slope_0",
               "ST_Slope_1",
               "ST_Slope_2",
               "Age_0",
               "Age_1",
               "Age_2",
               "Age_3",
               "Age_4",
               "RestingBP_0",
               "RestingBP_1",
               "RestingBP_2",
               "RestingBP_3",
               "RestingBP_4",
               "Cholesterol_0",
               "Cholesterol_1",
               "Cholesterol_2",
               "Cholesterol_3",
               "Cholesterol_4",
               "MaxHR_0",
               "MaxHR_1",
               "MaxHR_2",
               "MaxHR_3",
               "MaxHR_4",
               "Oldpeak_0",
               "Oldpeak_1",
               "Oldpeak_2",
               "Oldpeak_3",
               "Oldpeak_4",
           ]
           X_new = pd.DataFrame(
               [np.zeros(len(feature_names))], columns=feature_names
           )


           result, probability = classifier.predict(X_new)


           self.assertIsInstance(result, str)
           self.assertIsInstance(probability, float)
           self.assertTrue(
               0 <= probability <= 1, "Probability should be between 0 and 1"
           )


   def test_model_not_trained_exception(self) -> None:
       """Test that prediction raises an exception if not trained."""
       classifier = HeartDiseaseClassifier()
       with self.assertRaises(Exception) as context:
           classifier.predict(np.array([[0] * 41]))
       self.assertIn("Model has not been trained yet", str(context.exception))




if __name__ == "__main__":
   unittest.main()
