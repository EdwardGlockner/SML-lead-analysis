##########################################################
## IMPORTS

import os
import sys
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

##########################################################
## FIXING PATH AND OS

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting/mains", "")
sys.path.insert(1, os.path.join(dirname, "general_classes/.."))

##########################################################
## LOCAL PACKAGES

from general_classes import *

##########################################################
## FUNCTIONS

def normal_pred():
    # Set data
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, numpy_bool = True, gender = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Use data augmentation
    sm = SMOTE(k_neighbors = 5)
    X_res_a, Y_res_a = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train,X_res_a))
    Y_train = np.concatenate((Y_train, Y_res_a))

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
    DataPrep = DataPreparation(path, numpy_bool = True, gender = False)
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
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
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


def evaluation_cross_val(n_folds = 10):
    # Get the data sets
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, numpy_bool = True, gender = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Merge the data
    X = np.concatenate((X_train, X_test))
    Y = np.concatenate((Y_train, Y_test))
    
    qda = QuadraticDiscriminantAnalysis()
    
    data = get_dict()   # data dict
    
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
        
        # Normalize the data
        scaler = preprocessing.StandardScaler().fit(X_train_loop)
        X_train_loop = scaler.transform(X_train_loop)
        X_val_loop = scaler.transform(X_val_loop)
        
        model = qda.fit(X_train_loop, Y_train_loop)

        Y_pred_loop = model.predict(X_val_loop)
        
        perf = Performance(Y_pred_loop, Y_val_loop)
        perf.combination(data)
        
        print("Calculating ",f"{round((fold+1)/n_folds*100,2)}%",end="\r")
    
    print("\r\n")
    print_combination(data)

##########################################################
## MAIN

def main():
    # normal_pred()
    evaluation_cross_val()

##########################################################
## RUN CODE    

if __name__ == "__main__":
    main()
