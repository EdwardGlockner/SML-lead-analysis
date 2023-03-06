##########################################################
## IMPORTS

import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

##########################################################
## FIX PATHS AND OS

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/log_reg", "")
sys.path.insert(1, os.path.join(dirname, "general_classes"))

##########################################################
## LOCAL PACKAGES

from DataPreparation import DataPreparation
from Performance import Performance, print_combination, get_dict

##########################################################
## FUNCTIONS

def hyper_tuning():
    # Set data
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, numpy_bool = True, gender = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()
    
    # Use data augmentation
    sm = SMOTE(k_neighbors = 5)
    X_res_a, Y_res_a = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res_a))
    Y_train = np.concatenate((Y_train, Y_res_a))

    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
    logreg=LogisticRegression()
    logreg_cv=GridSearchCV(logreg,grid,cv=10)
    logreg_cv.fit(X_train,Y_train)

    print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
    print("accuracy :",logreg_cv.best_score_)


def normal_pred():
    # Set data
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, numpy_bool = True, gender = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()
    
    # Use data augmentation
    sm = SMOTE(k_neighbors = 5)
    X_res_a, Y_res_a = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res_a))
    Y_train = np.concatenate((Y_train, Y_res_a))

    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit the model and make predictions
    LR = LogisticRegression()
    model = LR.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    # Print performance
    print(classification_report(Y_test, y_pred))
    perf = Performance(y_pred, Y_test)
    print(f"Kappa: {perf.cohen()}")


def evaluation_cross_val(n_folds = 10):
    # Get the data sets
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, numpy_bool = True, gender = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Merge the data
    X = np.concatenate((X_train, X_test))
    Y = np.concatenate((Y_train, Y_test))
    
    data = get_dict()
    
    LR = LogisticRegression(max_iter=10000, penalty = 'l2')
    
    cross_val = KFold(n_splits = n_folds, shuffle= True, random_state=False)
    
    print("Calculating ","0%",end="\r")

    for fold, (index_train, index_val) in enumerate(cross_val.split(X)):     # cross_val.split() gives the indices for the training and validation data
        X_train_loop, X_val_loop = X[index_train], X[index_val]
        Y_train_loop, Y_val_loop = Y[index_train], Y[index_val]

        # Use SMOTE for over sampling
        sm = SMOTE()
        X_res, Y_res = sm.fit_resample(X_train_loop, Y_train_loop)
        X_train_loop = np.concatenate((X_train_loop, X_res))
        Y_train_loop = np.concatenate((Y_train_loop, Y_res))
        
        model = LR.fit(X_train_loop, Y_train_loop)

        Y_pred_loop = model.predict(X_val_loop)

        perf = Performance(Y_pred_loop, Y_val_loop)
        perf.combination(data)
        
        print("Calculating ",f"{round((fold+1)/n_folds*100,2)}%",end="\r")

    print("\r\n")
    print_combination(data)

##########################################################
## MAIN

def main():
    normal_pred()
    evaluation_cross_val()

##########################################################
## RUN CODE

if __name__ == "__main__":
    main()

##########################################################
## MAIN

def main():
    normal_pred()
    evaluation_cross_val()
    #hyper_tuning()

##########################################################
## RUN CODE

if __name__ == "__main__":
    main()
