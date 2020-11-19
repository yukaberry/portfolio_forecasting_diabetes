from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


def generate_auc_roc_curve(clf, X_test,y_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc=4)
    plt.show()
    pass


def return_x_val_accuracy(clf,X_train,y_train,kfold=5):
    """Return kfold cross validation scores of accuracy and its mean. kfold = 5 as default """
    cross_validation_score = cross_val_score(clf,X_train,y_train,cv=kfold)
    estimated_accuracy = (cross_validation_score.mean(), cross_validation_score.std() * 2)
    
    print("Estimate cross validation accuracy: %0.2f (+/- %0.2f)" % (cross_validation_score.mean(), cross_validation_score.std() * 2))
    print("Cross validation scores " + str(cross_validation_score))
    return cross_validation_score,estimated_accuracy


def return_x_val_f1_macro(clf,X_train,y_train,kfold=5):
    """Return kfold cross validation scores of f1_macro and its mean kfold = 5 as default """
    cross_validation_score_f1 = cross_val_score(clf,X_train,y_train,cv=kfold,scoring='f1_macro')
    estimated_f1 = (cross_validation_score_f1.mean(), cross_validation_score_f1.std() * 2)
    
    print("Estimate cross validation F1_macro: %0.2f (+/- %0.2f)" % (cross_validation_score_f1.mean(), cross_validation_score_f1.std() * 2))
    print("F1_macro Cross validation scores: " + str(cross_validation_score_f1))
    return cross_validation_score_f1,estimated_f1