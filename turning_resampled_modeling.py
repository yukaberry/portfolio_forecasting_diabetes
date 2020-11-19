import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz
import pydotplus
import six
import os
from sklearn import tree

from utils.modeling_utils import generate_auc_roc_curve
from utils.modeling_utils import return_x_val_accuracy
from utils.modeling_utils import return_x_val_f1_macro


# load and split non resampled data
df_all = pd.read_csv("data/cleaned_all.csv")
X = df_all[['Pregnancies', 'Glucose', 'BloodPressure', 'BMI',
       'DiabetesPedigreeFunction', 'Age', 'BMI_group_Normal_weight',
       'BMI_group_Obesity', 'BMI_group_Overweight', 'BMI_group_Underweight_',
       'Glucose_group_Normal', 'Glucose_group_Pre_Diabetes',
       'Pregnancies_MoreThanOnce']]
y = df_all[["Outcome2"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# load and split resampled data
df_r = pd.read_csv("data/under_sample_df.csv")
X_r = df_r[['Pregnancies', 'Glucose', 'BloodPressure', 'BMI',
       'DiabetesPedigreeFunction', 'Age', 'BMI_group_Normal_weight',
       'BMI_group_Obesity', 'BMI_group_Overweight', 'BMI_group_Underweight_',
       'Glucose_group_Normal', 'Glucose_group_Pre_Diabetes',
       'Pregnancies_MoreThanOnce']]
y_r = df_r[["Outcome2"]]

# Baseline 
clf_r = RandomForestClassifier(random_state=42).fit(X_r,y_r)
y_pred_r =clf_r.predict(X_test)
print(classification_report(y_test, y_pred_r))

# visualize a tree (estimator_[0])out of random forest 
dotfile = six.StringIO()
i_tree = 0
feature_name=X_r.columns
lable_name=y_r.columns

for tree_in_forest in clf_r.estimators_:
    if (i_tree <1):        
        tree.export_graphviz(tree_in_forest, out_file=dotfile)
        pydotplus.graph_from_dot_data(dotfile.getvalue()).write_png('resampled_decision_tree_number'+ str(i_tree) +'.png')
        i_tree = i_tree + 1
    
        export_graphviz(tree_in_forest,
                feature_names=feature_name,
                filled=True,
                rounded=True)
        os.system('dot -Tpng tree.dot -o tree.png')

# Show Baseline auc roc chart 
generate_auc_roc_curve(clf_r, X_test,y_test)




# Gridsearch
weights_grid_rf_r= np.linspace(0.05, 0.95, 20)


gsc_rf_r = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights_grid_rf_r],
        'criterion':["gini","entropy"],
        'max_depth': [None,4,6,8],
        'min_samples_split':[2,4,6],
        'oob_score':[True]
    },
    scoring='f1_macro',
    cv=5,
    return_train_score=True
)

grid_result_rf_r = gsc_rf_r.fit(X_r, y_r)
print("Best parameters : %s" % grid_result_rf_r.best_params_)
print(grid_result_rf_r.best_score_)
print(grid_result_rf_r.cv_results_['mean_train_score']) # Need to set "True" to get results: GridSearchCV(return_train_score=True)




# kfold cross validation to compare baseline and turned model
# Baseline
return_x_val_accuracy(clf_r,X_r,y_r,kfold=5)
return_x_val_f1_macro(clf_r,X_r,y_r,kfold=5)
# Turned model
# Set parameters from gridsearch optimised params
turned_rfc_r=RandomForestClassifier(random_state=42)
turned_rfc_r.set_params(**grid_result_rf_r.best_params_)
print(turned_rfc_r.get_params)
return_x_val_accuracy(turned_rfc_r,X_r,y_r,kfold=5)
return_x_val_f1_macro(turned_rfc_r,X_r,y_r,kfold=5)
