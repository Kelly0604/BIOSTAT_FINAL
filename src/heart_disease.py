"""Train RandomForest model to predict prescence of heart disease."""


import warnings
from typing import Any, Tuple, Union


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Ignore specific warning categories or all warnings
warnings.filterwarnings("ignore")




class HeartDiseaseClassifier:
   """Heart Disease Classifier."""


   def __init__(self) -> None:
       """Initialize the classifier."""
       self.model: Union[None, RandomForestClassifier] = None
       self.scaler: StandardScaler = StandardScaler()


   def load_and_prepare_data(
       self, url: str
   ) -> Tuple[
       np.ndarray[np.float64, Any],
       np.ndarray[np.float64, Any],
       np.ndarray[np.int32, Any],
       np.ndarray[np.int32, Any],
   ]:
       """Load and prepare data from a CSV file."""
       try:
           data = pd.read_csv(url)
       except Exception as e:
           print(f"Failed to load data: {e}")
           return (
               np.empty((0, 0), dtype=np.float64),
               np.empty((0, 0), dtype=np.float64),
               np.empty((0, 0), dtype=np.int32),
               np.empty((0, 0), dtype=np.int32),
           )
       X = data.drop("HeartDisease", axis=1)
       y = data["HeartDisease"]
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.20, random_state=42, stratify=y
       )
       self.scaler.fit(X_train)
       return (
           self.scaler.transform(X_train),
           self.scaler.transform(X_test),
           y_train.to_numpy(dtype=np.int32),
           y_test.to_numpy(dtype=np.int32),
       )


   def train_model(self) -> None:
       """Train the RandomForest model."""
       confirm = input(
           "Proceed with loading and training the model? (yes/no): "
       )
       if confirm.lower() != "yes":
           print("Training canceled.")
           return


       data_url = (
           "https://raw.githubusercontent.com/Kelly0604/BIOSTAT_FINAL"
           "/main/00_data/preprocessed_data.csv"
       )
       X_train, X_test, y_train, y_test = self.load_and_prepare_data(data_url)


       if X_train.size == 0:
           return


       self.model = RandomForestClassifier(n_estimators=100, random_state=42)
       self.model.fit(X_train, y_train)
       accuracy = self.model.score(X_test, y_test)
       print(f"Model trained with accuracy: {accuracy:.2f}")


   def predict(self, X_new: np.ndarray[np.float64, Any]) -> Tuple[str, float]:
       """Predict the probability of heart disease for new data."""
       if self.model is None:
           raise Exception("Model has not been trained yet.")


       try:
           X_new = np.array(X_new, dtype=np.float64).reshape(1, -1)
           X_new = self.scaler.transform(X_new)
       except Exception as e:
           print(f"Error processing input data: {e}")
           return "", 0.0


       probabilities = self.model.predict_proba(X_new)
       prediction = probabilities[0][1]
       result = (
           "Positive for heart disease"
           if prediction >= 0.5
           else "Negative for heart disease"
       )
       return result, prediction




if __name__ == "__main__":
   classifier = HeartDiseaseClassifier()
   classifier.train_model()


   print("Enter patient data for prediction:")
   example_1 = np.array(
       list(
           map(
               int,
               input(
                   "Enter the patient data as"
                   "a comma-separated list of binary values: "
               ).split(","),
           )
       ),
       dtype=float,
   )
   result, probability = classifier.predict(example_1)
   if result:
       print(f"{result} (Probability: {probability:.2f})")
