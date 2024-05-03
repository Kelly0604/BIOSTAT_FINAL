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

**Step 1: Clone Repository**

Clone the repository using git clone _repo_url_. This will allow saving the processed data, model and heart disease predictor functions on local machine. 

**Step 2: Input User-Specific Information**

Input the user-specific information into the Heart Disease Classifier. All inputs should be categorized into binary or numeric bins as specified. For numeric variables, use the following bin definitions to classify each feature:

#### Feature: Age
- **Bin 0**: Age 28 to <38
- **Bin 1**: Age 38 to <48
- **Bin 2**: Age 48 to <58
- **Bin 3**: Age 58 to <68
- **Bin 4**: Age 68 and above

#### Feature: Resting Blood Pressure (RestingBP)
- **Bin 0**: BP < 40.0 mmHg
- **Bin 1**: BP 40.0 to <80.0 mmHg
- **Bin 2**: BP 80.0 to <120.0 mmHg
- **Bin 3**: BP 120.0 to <160.0 mmHg
- **Bin 4**: BP 160.0 to <200.0 mmHg

#### Feature: Cholesterol
- **Bin 0**: Cholesterol < 120.6 mg/dL
- **Bin 1**: Cholesterol 120.6 to <241.2 mg/dL
- **Bin 2**: Cholesterol 241.2 to <361.8 mg/dL
- **Bin 3**: Cholesterol 361.8 to <482.4 mg/dL
- **Bin 4**: Cholesterol 482.4 to <603.0 mg/dL

#### Feature: Maximum Heart Rate (MaxHR)
- **Bin 0**: MaxHR < 88.4 bpm
- **Bin 1**: MaxHR 88.4 to <116.8 bpm
- **Bin 2**: MaxHR 116.8 to <145.2 bpm
- **Bin 3**: MaxHR 145.2 to <173.6 bpm
- **Bin 4**: MaxHR 173.6 to <202.0 bpm

#### Feature: Oldpeak
- **Bin 0**: Oldpeak < -0.84
- **Bin 1**: Oldpeak -0.84 to <0.92
- **Bin 2**: Oldpeak 0.92 to <2.68
- **Bin 3**: Oldpeak 2.68 to <4.44
- **Bin 4**: Oldpeak 4.44 to <6.2

#### User Input Example:


**Step 3: Training and Predictions**

Run Training and Predicting functions in heart_disease.py to predict results for the user. Results will contain whether the user is considered negative or positive as well as probability of diagnosing heart disease. 

**Step 4: Tests**

Contributor to the repository can add tests examples in test_heart_disease.py file for pytest to check testing results automatically. Contributors can also create pull request for creators to review.

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
