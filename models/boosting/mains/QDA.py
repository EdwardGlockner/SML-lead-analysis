### IMPORTS

import os
import sys

# Check folders so it works for different OS:s
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting/mains", "")
sys.path.insert(1, os.path.join(dirname, "general_classes"))

from DataPreparation import DataPreparation
from Performance import Performance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from PCA import PCA

### FUNCTIONS ###

def normal_pred():
    # Set data
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, drop_cols = [], numpy_bool = True, gender = False, normalize = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Use data augmentation
    sm2 = SMOTE(k_neighbors = 8)
    X_res_a, Y_res_a = sm2.fit_resample(X_train, Y_train)
    X_train2 = np.concatenate((X_train,X_res_a))
    Y_train2 = np.concatenate((Y_train, Y_res_a))

    # Train the model and make predictions
    qda = QuadraticDiscriminantAnalysis(reg_param = 0)
    model = qda.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    # Get performance
    print(classification_report(Y_test,y_pred))
    perf = Performance(y_pred,Y_test)
    print(f"Kappa: {perf.cohen()}")


def cross_val():
    # Set data
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, drop_cols = [], numpy_bool = True, gender = False, normalize = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Use data augmentation
    sm = SMOTE(k_neighbors = 5)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))
    
    # Merge all the sets
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((Y_train, Y_test))
    
    # Use k-fold cross validation
    qda = QuadraticDiscriminantAnalysis(reg_param = 0.0)
    model = qda.fit(X_train, Y_train)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_train , Y_train, scoring="accuracy", cv=cv)
    for i in range(len(scores)):
        print(f"Iteration: {i}, accuracy: \t {scores[i]}")

    print(f"\nMean accuracy: \t {np.mean(scores)}")
    
    # Check which reg_param is the best
    params = [{'reg_param': np.linspace(0,1,100)}]
    Grids = GridSearchCV(qda, params, cv=4)
    Grids.fit(X_train, Y_train)
    reg_params = Grids.best_params_['reg_param']
    print("Best reg_param for model is " + str(reg_params))


### MAIN ###
def main():
    normal_pred()
    cross_val()
    
    

if __name__ == "__main__":
    main()
