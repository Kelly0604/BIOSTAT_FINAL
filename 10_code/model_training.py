"""Train and test the model."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the preprocessed data
data = pd.read_csv(
    "https://raw.githubusercontent.com/Kelly0604/BIOSTAT_FINAL/main/00_data/preprocessed_data.csv"
)

# Split the data into training and testing sets
X = data.drop(["HeartDisease"], axis=1)
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100)

# Train the model using the training data
model.fit(X_train, y_train)

# Test the model using the testing data
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

# Save the trained model
joblib.dump(model, '20_outcomes/random_forest_model.pkl')
