# BIOSTAT_FINAL: Heart Disease Predictor

## Authors: Kelly Tong, Katherine Tian, Cassie Kang

#### Project Description

The Heart Disease Predictor aims to develop a user-friendly tool that utilizes the UCI Heart Disease Dataset to forecast individual risk of cardiovascular diseases. These conditions remain a leading cause of global mortality, emphasizing the need for effective early detection strategies. By inputting personal health metrics, users can receive a personalized assessment of their heart disease risk, facilitating timely healthcare interventions.

Our tool is built upon a database of 918 records from the Kaggle dataset, integrating crucial health indicators like age, sex, chest pain type, and more. We plan to employ machine learning technique to create a robust predictive model.

#### Goals and Impact
The primary objective of this project is to provide individuals with an accessible, accurate predictor for heart disease, with their specific health data. This predictive capability aims to support targeted monitoring and proactive management of cardiovascular health, potentially reducing the incidence and severity of heart disease.

#### Dataset Description

The dataset is sourced from Kaggle and can be found at: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data

The Heart Disease Dataset amalgamates five distinct datasets into a unified database comprising 918 records from an initial pool of 1190 observations. This consolidated dataset features 11 attributes, designed to facilitate the prediction of heart disease risk. These attributes encompass age, sex, type of chest pain, resting blood pressure, serum cholesterol levels, fasting blood sugar levels, results from resting electrocardiograms, maximum heart rate achieved, exercise-induced angina, ST segment depression (old peak), and the slope of the peak exercise ST segment, alongside the outcome variable indicating the presence or absence of heart disease.

This dataset is primed for utilization in research aimed at crafting machine learning models capable of assessing heart disease risk. Given the prominence of cardiovascular diseases as a leading global mortality factor, these models hold significant potential for the early identification and intervention of cardiovascular diseases in high-risk individuals, thereby contributing to improved management and outcomes.

#### Tool: Heart Disease Predictor

**Class - HeartDiseaseClassifier**

- Output whether the patient is likely to get a heart disease. Positive when larger than 50%, negative when smaller than 50%. Percentage of possibility of diagnosing will also be returned. 
  
- Provide warning if the patient is considered likely diagnosing heart disease.

#### User Guideline


#### Project Breakdown and Timeline
Phase 1: Data Exploration
  - Data exploratory analysis would be done on the dataset to see whether there are relationships between independent variables.
  - Visualizations will be generated to better understand the data

Phase 2: Data Cleaning 
- Data cleaning will be done to normalize, one-hot encoding, and replace null values in the dataset.
- Unecessary or irrelavant variables, as identified in the data exploration stage, might be dropped.

Phase 3: Training and Model Experiment 
- Train the data and split them into training, testing and validation set

Phase 4: Evaluating Model 
- The model will be evaluated based on their accuracy and AUC. The best model will be used to develop our user interface function. 
  
Phase 5: Develop user interface
- Allow user to predict their own chance of being diagnosed of heart disease using our model.
