"""Tests for the heart_disease module."""
import numpy as np
import pandas as pd
from heart_disease import HeartDiseaseClassifier


def test_data_loading_and_preparation():
    """Test that data is loaded and prepared correctly."""
    classifier = HeartDiseaseClassifier()
    data = '00_data/preprocessed_data.csv'
    X_train, X_test, y_train, y_test = classifier.load_and_prepare_data(data)

    assert X_train.shape[0] > 0 and X_test.shape[0] > 0
    assert y_train.shape[0] > 0 and y_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1], \
        "Features count mismatch between training and testing sets"

def test_model_training():
    """Test the model is trained without errors and classifier has a model."""
    classifier = HeartDiseaseClassifier()
    data = '00_data/preprocessed_data.csv'
    classifier.train_model(data)
    
    assert classifier.model is not None

def test_prediction():
    """Test the prediction output format and content."""
    classifier = HeartDiseaseClassifier()
    data = '00_data/preprocessed_data.csv'
    classifier.train_model(data)
    
    feature_names = [
        'Sex_0','Sex_1','ChestPainType_0',
        'ChestPainType_1','ChestPainType_2','ChestPainType_3',
        'FastingBS_0','FastingBS_1','RestingECG_0',
        'RestingECG_1','RestingECG_2','ExerciseAngina_0',
        'ExerciseAngina_1','ST_Slope_0','ST_Slope_1',
        'ST_Slope_2','Age_0','Age_1',
        'Age_2','Age_3','Age_4',
        'RestingBP_0','RestingBP_1',
        'RestingBP_2','RestingBP_3','RestingBP_4',
        'Cholesterol_0','Cholesterol_1',
        'Cholesterol_2','Cholesterol_3',
        'Cholesterol_4','MaxHR_0','MaxHR_1',
        'MaxHR_2','MaxHR_3','MaxHR_4',
        'Oldpeak_0','Oldpeak_1',
        'Oldpeak_2','Oldpeak_3',
        'Oldpeak_4'
    ]
    X_new = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    
    result, probability = classifier.predict(X_new)
    
    assert isinstance(result, str)
    assert isinstance(probability, float)
    assert 0 <= probability <= 1, "Probability should be between 0 and 1"


    
def test_model_not_trained_exception():
    """Test that prediction raises an exception if the model is not trained."""
    classifier = HeartDiseaseClassifier()
    try:
        classifier.predict([0]*41)
        raise AssertionError("Prediction not pass without trained model")
    except Exception as e:
        assert "Model has not been trained yet" in str(e)
