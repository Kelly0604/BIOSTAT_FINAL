"""Data Exploration and Visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the data
data = pd.read_csv(
    "https://raw.githubusercontent.com/Kelly0604/BIOSTAT_FINAL/main/00_data/heart_disease.csv"
)


# Preprocessing the data
def dataPreprocessing(input_df):
    """Convert all categorical columns to numeric."""
    cat_columns = input_df.select_dtypes(["object"]).columns
    input_df[cat_columns] = input_df[cat_columns].apply(
        lambda x: pd.factorize(x)[0]
    )
    return input_df


df = dataPreprocessing(data)

print(data.columns)


# Exploratory Data Analysis
def getDfSummary(input_data):
    """Give a slightly more robust initial EDA."""
    output_data = input_data.describe()
    output_data = output_data.transpose()

    # get the median of each column as well
    output_data["median"] = np.median(input_data, axis=0)

    # get distinct counts, first use the nunique function
    uniques = input_data.nunique(0)
    ph = uniques.to_frame(name="number_distinct")
    output_data = pd.merge(output_data, ph, left_index=True, right_index=True)

    numrows = len(input_data.index)
    output_data["number_nan"] = output_data["count"].apply(
        lambda x: numrows - x
    )

    return output_data


getDfSummary(df)

# Define the features
discrete_fts = [
    "Sex",
    "ChestPainType",
    "FastingBS",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
cts_fts = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
target = ["HeartDisease"]

# Separate data by target values
df_pos = data.loc[data.HeartDisease != 1].astype(object)
df_neg = data.loc[data.HeartDisease == 0].astype(object)

# Plot histograms for discrete features
fig, ax = plt.subplots(2, 4, figsize=(20, 12))
for n, f in enumerate(discrete_fts):
    ax[int(np.floor(n / 4)), n % 4].hist(
        [df_neg[f], df_pos[f]],
        bins=max(len(np.unique(df_neg[f])), len(np.unique(df_pos[f]))),
        label=["HeartDisease=0", "HeartDisease=1"],
    )
    ax[int(np.floor(n / 4)), n % 4].legend(loc="best")
    ax[int(np.floor(n / 4)), n % 4].set_title(f)
plt.show()

# Plot pairwise distributions for continuous features
sns.pairplot(data[cts_fts + target], hue=target[0])

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap="Blues", annot=True)
plt.show()
data.corr()
