# BIOSTAT_FINAL: Heart Disease Predictor

## Authors: Kelly Tong, Katherine Tian, Cassie Kang

##### Project Description

Our project employs the UCI Heart Disease Dataset, which includes 918 observations across 11 critical features, to develop a predictive machine learning framework. This comprehensive dataset features essential indicators of heart disease risk such as age, sex, and type of chest pain, pivotal for assessing the likelihood of cardiovascular conditions. With cardiovascular diseases being a leading cause of death globally, our goal is to leverage this vital data for early detection and effective management of heart disease risks in individuals at elevated risk.

We aim to customize our model to process user-specific data, providing personalized risk assessments to support targeted cardiovascular health monitoring and interventions. To ensure the precision and effectiveness of our predictive capabilities, we plan to conduct a thorough comparative analysis of various machine learning algorithms, including Logistic Regression, Support Vector Machine (SVM), Random Forest, Decision Tree, and XGBoost. This analysis will help us identify the most suitable algorithm for our dataset, enhancing the reliability and accuracy of our risk evaluations.

Our objective is to create a reliable predictive model that will serve as a foundational tool for advanced cardiovascular disease management and research, by meticulously evaluating and selecting the most efficient machine learning approach for heart disease prediction. As a result, users will be able to input their own personal data and receive an estimated result on whether they will be diagnosed a heart disease. 

##### Dataset Description

The dataset is sourced from Kaggle and can be found at: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data

The Heart Disease Dataset amalgamates five distinct datasets into a unified database comprising 918 records from an initial pool of 1190 observations. This consolidated dataset features 11 attributes, designed to facilitate the prediction of heart disease risk. These attributes encompass age, sex, type of chest pain, resting blood pressure, serum cholesterol levels, fasting blood sugar levels, results from resting electrocardiograms, maximum heart rate achieved, exercise-induced angina, ST segment depression (old peak), and the slope of the peak exercise ST segment, alongside the outcome variable indicating the presence or absence of heart disease.

This dataset is primed for utilization in research aimed at crafting machine learning models capable of assessing heart disease risk. Given the prominence of cardiovascular diseases as a leading global mortality factor, these models hold significant potential for the early identification and intervention of cardiovascular diseases in high-risk individuals, thereby contributing to improved management and outcomes.

##### Project Breakdown and Timeline

**Phase 1: ** Data Exploration
  - Data exploratory analysis would be done on the dataset to see whether there are relationships between independent variables.
  - Visualizations will be generated to better understand the data

**Phase 2: ** Data Cleaning 
- Data cleaning will be done to normalize, one-hot encoding, and replace null values in the dataset.
- Unecessary or irrelavant variables, as identified in the data exploration stage, might be dropped.

**Phase 3: ** Training and Model Experiment 
- Train the data and split them into training, testing and validation set
- Several machine learning models such as KNN, Random Forest, Regression, XGBoost etc. will be experimented with the data.

**Phase 4: ** Evaluating Model 
- Each machine learning model will be evaluated based on their accuracy and AUC. Their pros and cons will be considered during this phase.

**Phase 5: ** Prediction 
- The best evaluated and tested model will be selected to predict heart disease of an individual based on self characteristics.
- User interface will be incorporated to allow user to predict their own chance of being diagnosed of heart disease.
