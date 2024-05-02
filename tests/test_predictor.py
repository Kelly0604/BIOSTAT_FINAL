"""This module includes test examples for the file predictor.py."""

import unittest
from typing import List

from predictor import IsHeartDisease


class TestIsHeartDisease(unittest.TestCase):
    """Test suite for the IsHeartDisease class."""

    def setUp(self) -> None:
        """Setup the IsHeartDisease instance with a known model path."""
        self.predictor: IsHeartDisease = IsHeartDisease()

    def test_predict_heart_positive(self) -> None:
        """Test predict_heart method for a patient with heart disease."""
        # Sample patient data where the model should predict heart disease (1)
        patient_with_disease: List[float] = [
            63,
            1,
            3,
            145,
            233,
            1,
            0,
            150,
            0,
            2.3,
            0,
            0,
            1,
        ]  # Example features
        self.assertEqual(
            self.predictor.predict_heart(patient_with_disease),
            "The patient is likely diagnosed with heart disease.",
        )

    def test_predict_heart_negative(self) -> None:
        """Test predict_heart method for a patient without heart disease."""
        # Sample patient data, the model should predict no heart disease (0)
        patient_without_disease: List[float] = [
            56,
            0,
            2,
            140,
            294,
            0,
            1,
            153,
            0,
            1.3,
            1,
            0,
            2,
        ]  # Example features
        self.assertEqual(
            self.predictor.predict_heart(patient_without_disease),
            "The patient is not likely diagnosed with heart disease.",
        )


if __name__ == "__main__":
    unittest.main()
