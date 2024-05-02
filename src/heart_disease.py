import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

class HeartDiseaseClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def load_and_prepare_data(self, url):
        """Load and prepare data from a CSV file."""
        data = pd.read_csv(url)
        X = data.drop("HeartDisease", axis=1)
        y = data["HeartDisease"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        self.scaler.fit(X_train)
        return self.scaler.transform(X_train), self.scaler.transform(X_test), y_train, y_test

    def train_model(self, url):
        """Train the RandomForest model."""
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(url)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy}")

    def predict(self, X_new):
        """Predict the probability of heart disease for new data."""
        if self.model is not None:
            X_new = np.array(X_new).reshape(1, -1) 
            X_new = self.scaler.transform(X_new)
            probabilities = self.model.predict_proba(X_new)
            prediction = probabilities[0][1]
            result = "Positive for heart disease" if prediction >= 0.5 else "Negative for heart disease"
            return result, prediction
        else:
            raise Exception("Model has not been trained yet.")

if __name__ == "__main__":
    classifier = HeartDiseaseClassifier()
    data_url = "https://raw.githubusercontent.com/Kelly0604/BIOSTAT_FINAL/main/00_data/preprocessed_data.csv"
    classifier.train_model(data_url)

    example_1 = [True,False,True,False,False,False,True,False,True,False,False,True,False,True,False,False,False,True,False,False,False,False,False,False,True,False,False,False,True,False,False,False,False,False,True,False,False,True,False,False,False]  # Example feature set
    result, probability = classifier.predict(example_1)
    example_2 = [False,True,False,True,False,False,True,False,True,False,False,True,False,False,True,False,False,False,True,False,False,False,False,False,True,False,False,True,False,False,False,False,False,False,True,False,False,False,True,False,False]
    result2, probability2 = classifier.predict(example_2)
    print(f"{result} (Probability: {probability:.2f})")
    print(f"{result2} (Probability: {probability2:.2f})")
