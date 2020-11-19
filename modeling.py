import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import model_selection
from sklearn.model_selection import cross_val_score

from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

import eli5
from eli5.sklearn import PermutationImportance

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_roc_curve



# load csv file
df_cleaned_all = pd.read_csv("data/cleaned_all.csv")

# Define X and y 
X = df_cleaned_all[['Pregnancies', 'Glucose', 'BloodPressure',
       'BMI', 'Age', 'MNAR', 'DiabetesPedigreeFunction',
       'Pregnancies_MoreThanOnce', 'BMI_group_Normal_weight',
       'BMI_group_Obesity', 'BMI_group_Overweight', 'BMI_group_Underweight_',
       'Glucose_group_Normal', 'Glucose_group_Pre_Diabetes']]
y = df_cleaned_all[["Outcome2"]]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model comparison
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC()))
models.append(('LSVC', LinearSVC()))
models.append(('RFC', RandomForestClassifier()))
models.append(('DTR', DecisionTreeRegressor()))
models.append(("XGB",XGBClassifier()))
models.append(("ADB", AdaBoostClassifier()))

seed = 7
results = []
names = []
X_model_comparison = X_train
Y_model_comparison = y_train

for name, model in models:
    kfold = model_selection.KFold(
        n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X_model_comparison, Y_model_comparison, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (
        name, cv_results.mean(), cv_results.std())
    print(msg)


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



# Show Feature imporntance  
# Random Forest
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rfc_model =RandomForestClassifier(random_state=0).fit(X_train, y_train)
perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)
print(eli5.format_as_text(eli5.explain_weights(perm, feature_names = val_X.columns.tolist(),top=20)))

# XGB
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
xgb_model =XGBClassifier(random_state=0).fit(X_train, y_train)
perm_xgb = PermutationImportance(xgb_model, random_state=1).fit(val_X, val_y)
print(eli5.format_as_text(eli5.explain_weights(perm_xgb, feature_names = val_X.columns.tolist(),top=20)))

# SVC
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
svc_model =SVC(random_state=0).fit(X_train, y_train)
perm_svc = PermutationImportance(svc_model, random_state=1).fit(val_X, val_y)
print(eli5.format_as_text(eli5.explain_weights(perm_svc, feature_names = val_X.columns.tolist(),top=20)))


# LSVC
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
lsvc_model =LinearSVC(random_state=0).fit(X_train, y_train)
perm_lsvc = PermutationImportance(lsvc_model, random_state=1).fit(val_X, val_y)
print(eli5.format_as_text(eli5.explain_weights(perm_lsvc, feature_names = val_X.columns.tolist(),top=20)))

#LR
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
lr_model =LogisticRegression(random_state=0).fit(X_train, y_train)
perm_lr = PermutationImportance(lr_model, random_state=1).fit(val_X, val_y)
print(eli5.format_as_text(eli5.explain_weights(perm_lr, feature_names = val_X.columns.tolist(),top=20)))





# Plot ROC and AUC

#Prep for LR
model_logistic = LogisticRegression().fit(X_train, y_train)
y_pred_logistic = model_logistic.decision_function(X_test)

#Prep for SVM
model_SVC = SVC().fit(X_train, y_train)
y_pred_SVC = model_SVC.decision_function(X_test)

# LR
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred_logistic)
auc_logistic = auc(logistic_fpr, logistic_tpr)

#SVM
svc_fpr, svc_tpr, threshold = roc_curve(y_test, y_pred_SVC)
auc_svc = auc(svc_fpr, svc_tpr)

# Plot Graph
plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svc_fpr, svc_tpr, linestyle='-', label='SVC (auc = %0.3f)' % auc_svc)
plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)
plt.legend()

#matplotlib
ax = plt.gca()

# prep for RFC
model_RFC = RandomForestClassifier()
model_RFC.fit(X_train, y_train)
RFC_display = plot_roc_curve(model_RFC, X_test, y_test, ax=ax, alpha=0.8)

#prep for XGB
model_XGB = XGBClassifier()
model_XGB.fit(X_train, y_train)
XGB_display = plot_roc_curve(model_XGB, X_test, y_test, ax=ax, alpha=0.8)

#Prep for ADB
model_ADB = AdaBoostClassifier()
model_ADB.fit(X_train, y_train)
ADB_display = plot_roc_curve(model_ADB, X_test, y_test, ax=ax, alpha=0.8)
plt.show()





