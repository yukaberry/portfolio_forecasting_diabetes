import pandas as pd
import numpy as np


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
from sklearn import model_selection

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

import eli5
from eli5.sklearn import PermutationImportance


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import plotly.express as px
from matplotlib import pyplot as plt
%matplotlib inline

from sklearn.tree import export_graphviz
import pydotplus
import six
from sklearn import tree
import os

from mlxtend.plotting import plot_confusion_matrix

from utils.modeling_utils import generate_auc_roc_curve
from utils.modeling_utils import return_x_val_accuracy
from utils.modeling_utils import return_x_val_f1_macro


# load and split non resampled data
df_all = pd.read_csv("../data/cleaned_all.csv")
X = df_all[['Pregnancies', 'Glucose', 'BloodPressure', 'BMI',
       'DiabetesPedigreeFunction', 'Age', 'BMI_group_Normal_weight',
       'BMI_group_Obesity', 'BMI_group_Overweight', 'BMI_group_Underweight_',
       'Glucose_group_Normal', 'Glucose_group_Pre_Diabetes',
       'Pregnancies_MoreThanOnce']]
y = df_all[["Outcome2"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model comparason
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
plt.ylabel("Accuracy score")
plt.show()





# ROC and AUC chart
# Prep for LR
model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)
y_pred_logistic = model_logistic.decision_function(X_test)

#Prep for SVM
model_SVC = SVC()
model_SVC.fit(X_train, y_train)
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






# Baseline 
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)
print(classification_report(y_test, y_pred))





# visualize a tree (estimator_[0])out of random forest 
dotfile = six.StringIO()
i_tree = 0
feature_name=X.columns
lable_name=y.columns

for tree_in_forest in clf.estimators_:
    if (i_tree <1):        
        tree.export_graphviz(tree_in_forest, out_file=dotfile)
        pydotplus.graph_from_dot_data(dotfile.getvalue()).write_png('decision_tree_number'+ str(i_tree) +'.png')
        i_tree = i_tree + 1
    
        export_graphviz(tree_in_forest,
                feature_names=feature_name,
                filled=True,
                rounded=True)
        os.system('dot -Tpng tree.dot -o tree.png')

# Random Forest is a list of trees.Get that list using the estimators_ attribute. clf.estimators_[0]
# default estimatos in random forest is 100. see a result of "get_params", "n_estimators"
print(len(clf.estimators_))
print(clf.get_params)





# Show Baseline auc roc chart 
generate_auc_roc_curve(clf, X_test,y_test)




# Gridsearch
weights_grid_rf= np.linspace(0.05, 0.95, 20)


gsc_rf = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights_grid_rf],
        'criterion':["gini","entropy"],
        'max_depth': [None,4,6,8],
        'min_samples_split':[2,4,6],
        'oob_score':[True]
    },
    scoring='f1_macro',
    cv=5,
    return_train_score=True
)

grid_result_rf = gsc_rf.fit(X_train, y_train)
print("Best parameters : %s" % grid_result_rf.best_params_)
print(grid_result_rf.best_score_)
print(grid_result_rf.cv_results_['mean_train_score']) # Need to set "True" to get results: GridSearchCV(return_train_score=True)







# Turned model classification report
# Set parameters from gridsearch optimised params
turned_rfc=RandomForestClassifier()
turned_rfc.set_params(**grid_result_rf.best_params_)
print(turned_rfc.get_params)
turned_rfc.fit(X_train,y_train)
y_pred =turned_rfc.predict(X_test)
print(classification_report(y_test, y_pred))



# Turned model ROC AUD curve
generate_auc_roc_curve(turned_rfc, X_test)






# 5fold cross validation to compare baseline and turned model
# Baseline
clf_cv, clf_ac = return_x_val_accuracy(clf,X_train,y_train,kfold=5)
clf_cv_f1, clf_f1 = return_x_val_f1_macro(clf,X_train,y_train,kfold=5)
# Turned model
# Set parameters from gridsearch optimised params
turned_cv,turned_ac =return_x_val_accuracy(turned_rfc,X_train,y_train,kfold=5)
turned_cv_f1,turned_f1=return_x_val_f1_macro(turned_rfc,X_train,y_train,kfold=5)



# Table of model performances
results={'Classifier':["Random Forest Classifier","Turned Random Forest Classifier"],
'Accuracy':[str(clf_ac)[:6],str(turned_ac)[:6]],
'F1_macro':[str(clf_f1)[:6],str(turned_f1)[:6]]}

score_report_df =pd.DataFrame(data=results,columns=["Classifier","Accuracy","F1_macro"])
print("Base model and turned model, 5fold cross validation")
print(score_report_df)







# Confusion matrix
turned_rfc_confusion_matrix =confusion_matrix(y_test,y_pred)
class_names = ["Diabetes","Non Diabetes"]
fig,ax =plot_confusion_matrix(conf_mat = turned_rfc_confusion_matrix,colorbar = True,
                             show_absolute=False, show_normed=True,
                             class_names = class_names)
plt.show()






# Feature importance
perm = PermutationImportance(turned_rfc,).fit(X_train, y_train)
eli5.show_weights(perm, feature_names = X_train.columns.tolist(),top=20)
