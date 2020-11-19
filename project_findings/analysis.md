# Project : Forecasting Diabetes by diagnostic measures


# Table of contents

1.The objecttive of this project
2.Data details
3.Model and methods Overview
4.Domain : Diabetes types
5.Assumptions
6.Feature creattion and Data cleaning
7.Model selection
8.Challenges and Augmentations



# 1.The objective of the dataset 
To diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements.all patients here are females at least 21 years old of Pima Indian heritage

# 2. Data details
### Dataset
* Total samples size: 768 
    - Diabetes 268 / Non Diabetes 500  [1:2] Ratio
* Predictor variables: 8 variables
    - Pregnancies : The number of times pregnant
    - Glucose : Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    - BloodPressure : Blood Pressure (mm Hg)
    - SkinThickness : Triceps skin fold thickness (mm)
    - Insulin : 2-Hour serum insulin (mu U/ml)
    - BMI : Body mass index (weight in kg/(height in m)^2)
    - DiabetesPedigreeFunction : Diabetes Pedigree Function
    - Age : Age
* Labels : 
    - Outcome (1 for Diabetes, 0 for Non-Diabetes) 

# 3. Model and methods Overview

### Random Forest and Logistic Regression
### ROC and AUC for viewing F1 score
### Class_weight, Resampling for imbalanced data


# 4. Domain (Types of Diabetes)

## Diabetes Type 1
This type occurs most frequently in children and adolescents
Genetic/Ethnic influence Ex : American Indians(this dataset) , African Americans, Hispanics/Latinos etc 

## Diabetes Type 2
This type occurs in adults and accounts for around 90% of all diabetes cases
Tendency : Heart and blood vessel disease,Overweight 

## Diabetes Gestational diabetes
High blood glucose during pregnancy
Usually disappears after pregnancy but women affected and their children are at increased risk of developing type 2  later in life

[Reference link](https://www.idf.org/aboutdiabetes/what-is-diabetes.html?gclid=Cj0KCQjwyPbzBRDsARIsAFh15JbnJPIjlz4ByJoQ5NmP71B0fMTZAgn4v5Ht5VT51Iwpg9N9zKv7RE4aAo_9EALw_wcB)


# 5. Assumption

* “High blood pressure” + “BMI over 30” = Diabete?
* Being pregnant many times = Diabetes?

![feature_importance](/images/fearture_importance_rf.PNG)


# 6. Feature Engeneering and Data Cleaning


## Create new Features

* BMI group
    - Underweight below 18.5,  Normal weight = 18.5–24.9,  Overweight = 25–29.9 , Obesity  over 30 
[reference link for BMI](https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmi-m.htm)
* Glucose group
    - Normal Below 140 , Prediabetes 140 - 199 (mg/dL ), Diabetes over 200 
[reference link for Glucose](https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451)
* Pregnancies group
    - Zero or more than once

## Remove 

* Insulin and Skin Thickness features
    - Insulin(374) and Skin Thickness(227). Most of observations miss both of values. 

## Missing data

* Take Median values to fill out BMI, blood pressure, Glucose's missing values

## Handling imbalanced data 

* Set class_weights (for Logistic Regression)
* Resampling (SMOTE and Undersampling) 

![resampling_method](/images/resampling_method.PNG)


# 7. Model Selection

## 7.1 Target and Evaluation
* **Target : Improve F1 score (Precision and Recall Average), Not accuracy
* **Evaluation of models : ROC and AUC curve chart


![model_comparason](/images/model_comparason.PNG)
![ROC_AUC](/images/ROC_AUC.PNG)


## 7.2 Score details

![dataframe of scores](/images/)

![rocaud of rf](/images/roc_auc_rf.PNG)

![classifcation report of rf](/images/)


## 7.3 FP and FN studies

**False Positive(Model classify as Diabetes but it is actually NOT diabetes)**
* Age : older than 37 years old **88%**
* Pregnancies: Yes 77 %
* BMI_group_Obesity 77 %
* Glucose pre diabetes 77 %
**False Negative(Model classify as not Diabetes but it is actually diabetes)**
* Age : **20s 55%**,  30s 30%,  40s 15%
* Pregnancies : Yes 80% 
* **BMI obesity or overweight 95% (BMI_group_Obesity 55% , overweight 40% )**
* **Glucose Normal 90%**


# 8. Augumentation and Challenges


## With additional time I would do the following....
* Study deeper on features of FP and FN and create new features to improve F1 score
* Collect more observations since the dataset was too small.
* Ask experties' advise of types of diabetes. It would be interesting to know if labels are detailed (Type1, Type2, Gestational)
* Explore different methods for handling imbalanced data resampling (ex: ENN, Hybrid of Random over + under sampling)   
* Missing insulin data which could be important feature(based on domain info)
* Predicotor variable "Diabetes Pedigree Function". There was no explanation of what it was and how it was created.
* Handling outliers (especially "Blood Pressure" variable) 

![outliers](/images/outliers.PNG)




















