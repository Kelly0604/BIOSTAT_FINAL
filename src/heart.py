import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class HeartDiseaseClassifier:
    """A classifier to predict the presence of heart disease in a patient."""

    def __init__(self, input_shape: int) -> None:
        self.input_shape = input_shape

    @property
    def raw_model(self):
        print("Building heart disease prediction model...")
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.input_shape,)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def train_model(self, data_path, test_mode=False):
        data = pd.read_csv(data_path)
        X = data.drop('target', axis=1).values
        y = data['target'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(32)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

        model = self.raw_model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        epochs = 1 if test_mode else 10
        history = model.fit(train_ds, epochs=epochs, validation_data=test_ds)

        results = model.evaluate(test_ds)
        print(f"Test loss: {results[0]}, Test accuracy: {results[1]}")

        self.model = model
        np.save('scaler.npy', scaler.scale_)
        print("Model and scaler saved successfully.")
        return model

    def save_model(self, filepath="saved_models/heartDiseaseModel"):
        self.model.save(filepath)

    def predict(self, input_data, model_path="saved_models/heartDiseaseModel"):
        model = tf.keras.models.load_model(model_path)

        scaler = StandardScaler()
        scaler.scale_ = np.load('scaler.npy')
        input_scaled = scaler.transform([input_data])

        prediction = model.predict(input_scaled)
        print(f"Prediction (probability of having heart disease): {prediction[0]}")
        return prediction[0]

if __name__ == "__main__":
    classifier = HeartDiseaseClassifier(input_shape=13)
    classifier.train_model('/00_data/preprocessed_data.csv') 
    prediction = classifier.predict([0, 1, 0, 1, 125, 230, 1, 150, 1, 2.3, 2, 0, 2])
    print(f"Predicted probability: {prediction}")