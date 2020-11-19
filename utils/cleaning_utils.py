import pandas as pd
import numpy as np


def return_median(df,variable): 
    """calculate and return median according to the Outcome""" 

    temp_df = df[df[variable].notnull()]
    temp_df = temp_df[[variable, 'Outcome']].groupby(['Outcome'])[[variable]].median().reset_index()
    value_for_0 = temp_df.iloc[0][variable]
    value_for_1 = temp_df.iloc[1][variable]
    return value_for_0,value_for_1


def filling_median(df,variable,value_for_0,value_for_1):
    """filling median value"""

    df.loc[(df['Outcome'] == 0 ) & (df[variable].isnull()), variable] = value_for_0
    df.loc[(df['Outcome'] == 1 ) & (df[variable].isnull()), variable] = value_for_1
    return df


def bmi_cl(df):
    """BMI Categories: Underweight = <18.5 Normal weight = 18.5–24.9 Overweight = 25–29.9 Obesity = BMI of 30 or greater"""
    
    if 0<= df['BMI']<=18.5:
        return 'Underweight '
    elif 18.6 <= df['BMI']<=24.9:
        return "Normal weight"
    elif 25 <= df["BMI"]<=29.9:
        return "Overweight"
    elif df["BMI"]>=30:
        return "Obesity"


def glucose_cl(df):
    """Glucose below 140 is nomal , 140 mg/dL and 199 is prediabetes, over 200 is considered as diabeties"""

    if 0<= df['Glucose']<=139:
        return 'Normal'
    elif 140 <=df["Glucose"]<=199:
        return "Pre Diabetes"
