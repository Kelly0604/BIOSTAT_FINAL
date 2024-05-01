"""Data Cleaning and Preprocessing."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the data
data = pd.read_csv("https://raw.githubusercontent.com/Kelly0604/BIOSTAT_FINAL/main/00_data/heart_disease.csv")

# Preprocessing the data
def dataPreprocessing(input_df):
  """Convert all categorical columns to numeric."""
  cat_columns = input_df.select_dtypes(['object']).columns
  input_df[cat_columns] = input_df[cat_columns].apply(
     lambda x: pd.factorize(x)[0])
  return input_df

df = dataPreprocessing(data)
nan_values = data.isnull().sum()

# Data Bining
# Fill NaN values with the mean of each column
data_filled = data.fillna(data.mean())

# Bin the continuous features
data_bin = data_filled.copy()
continuous_f = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

for feature in continuous_f:
    bins = 5
    data_bin[feature] = pd.cut(data_filled[feature], bins=bins, labels=range(bins)).astype(np.int64)

# One-Hot Encoding
categorical_fts = ['Sex', 'ChestPainType', 'FastingBS', 
                   'RestingECG', 'ExerciseAngina', 'ST_Slope']
continuous_f = ['Age', 'RestingBP', 
                'Cholesterol', 'MaxHR', 'Oldpeak']

data_dumm = pd.get_dummies(data_bin, columns=categorical_fts + continuous_f)

# Save preprocessed data as CSV
data_dumm.to_csv("00_data/preprocessed_data.csv", index=False)