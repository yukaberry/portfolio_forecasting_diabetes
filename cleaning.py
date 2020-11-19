import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from utils.cleaning_utils import return_median
from utils.cleaning_utils import filling_median
from utils.cleaning_utils import bmi_cl
from utils.cleaning_utils import glucose_cl


df = pd.read_csv("data/diabetes.csv")

# Feature Engineering
# Creating new column for "Missing Not At Random" as "MNAR". 
# When Insulin is missing, SkinThickness is definetely missing.
# Using SkinThickenss' indices, creating new columns.
df.loc[(df["SkinThickness"] == 0 ), "MNAR"] = 1
df.loc[(df["SkinThickness"] >=1 ), "MNAR"] = 0

# Check if there are any ZERO values
df_columns=['Glucose','BloodPressure',"BMI"]
for i in df_columns:
    df[i].replace(0, np.nan, inplace= True)
#print(df.isnull().sum())

# Calculate and return median according to the Outcome
gl_value_for_0,gl_value_for_1 = return_median(df,"Glucose")
bp_value_for_0,bp_value_for_1 = return_median(df,"BloodPressure")
bmi_value_for_0,bmi_value_for_1 = return_median(df,"BMI")
# Replace zero with median value
filling_median(df,"Glucose",gl_value_for_0,gl_value_for_1)
filling_median(df,"BloodPressure",bp_value_for_0,bp_value_for_1)
filling_median(df,"BMI",bmi_value_for_0,bmi_value_for_1)
#print(df.isnull().sum())

# BMI Categories: Underweight = <18.5 Normal weight = 18.5–24.9 Overweight = 25–29.9 Obesity = BMI of 30 or greater
df['BMI_group'] = df.apply(lambda df: bmi_cl(df), axis=1)

# Glucose below 140 is nomal , 140 mg/dL and 199 is prediabetes, over 200 is considered as diabeties
df['Glucose_group'] = df.apply(lambda df: glucose_cl(df), axis=1)

# Pregnancies
df.loc[(df["Pregnancies"] == 0), "Pregnancies_MoreThanOnce"] = "0"
df.loc[(df["Pregnancies"] >= 1), "Pregnancies_MoreThanOnce"] = "1"

# Replace with dummy values
df = pd.get_dummies(df,columns=["BMI_group","Glucose_group"])

# Creating New  columns : Swap outcome column's value, 0 for Diabetes, 1 for Not Diabetes( Positive = 0 = Diabetes, Negative = 1= Not Diabetes),For ROC and AUC purpose 
df.loc[(df["Outcome"] == 0), "Outcome2"] =1
df.loc[(df["Outcome"] == 1), "Outcome2"] =0

# Change colums' order
df=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI','Age', 'MNAR','DiabetesPedigreeFunction',
       'Pregnancies_MoreThanOnce', 'BMI_group_Normal weight',
       'BMI_group_Obesity', 'BMI_group_Overweight', 'BMI_group_Underweight ',
       'Glucose_group_Normal', 'Glucose_group_Pre Diabetes','Outcome2']]


# Standarisation
num_features=["Pregnancies","Glucose","BloodPressure","BMI","Age","DiabetesPedigreeFunction",'SkinThickness', 'Insulin']

for col in num_features:
    scaler = StandardScaler()
    df[col] = scaler.fit_transform(np.array(df[col].values).reshape(-1,1))

# rename columns name with underscore("_") when they contain space(" ")
df.columns = [c.replace(' ', '_') for c in df.columns]

#save to csv file
df.to_csv("data/cleaned_all.csv",encoding='utf8',index=False)
df_cleaned_all = pd.read_csv("data/cleaned_all.csv")


# Resampling process
print(df_cleaned_all.Outcome2.value_counts())
under_sample_Diabetes= df_cleaned_all[df_cleaned_all["Outcome2"]==0]
under_sample_NonDiabetes= df_cleaned_all[df_cleaned_all["Outcome2"]==1]
under_sample = resample(under_sample_NonDiabetes, replace =False,n_samples=268,random_state=42)

# Concatnate two datasets
under_sample_df=pd.concat([under_sample,under_sample_Diabetes])

# check if 0 value and 1 value are the same number (268) and total observations are 536
print(under_sample_df.Outcome2.value_counts())
print(len(under_sample_df))

# Shuffle samples
under_sample_df = under_sample_df.sample(frac=1).reset_index(drop=True)
under_sample_df.head(10)

# save to csv file
under_sample_df.to_csv("data/under_sample_df.csv",encoding='utf8',index=False)
under_sample_df = pd.read_csv("data/under_sample_df.csv")
print(under_sample_df.shape)
print(under_sample_df.Outcome2.value_counts())





