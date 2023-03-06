##########################################################
## IMPORTS

import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as Q
from sklearn.model_selection import KFold
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import sys
import os 

##########################################################
## FIXING PATH
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
sys.path.insert(1, os.path.join(dirname, "general_classes"))

##########################################################
## LOCAL PACKAGES

from general_classes import DataPreparation 

##########################################################
## GLOBALS

bar = "************************************************************"

############################################################
## FUNCTIONS

def training():
    
    # setting up dataprep instance
    dp = DataPreparation("./data/train.csv", clean=True)
    
    # combines the data due to the training
    X_train, X_test, Y_train, Y_test = dp.get_sets()    
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((Y_train, Y_test))
    
    # Use data augmentation
    sm = SMOTE(k_neighbors = 5)
    X_res_a, Y_res_a = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res_a))
    Y_train = np.concatenate((Y_train, Y_res_a))
    
    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    # Train the model and make predictions
    qda = Q(reg_param = 0.0)
    model = qda.fit(X_train, Y_train)
    
    return model

def predicting():
    
    dp = DataPreparation("./data/test.csv", test=True)
    
    
    scaler = preprocessing.StandardScaler().fit(dp.X_true)
    dp.X_true = scaler.transform(dp.X_true)
    
    model = training()
    
    preds = model.predict(dp.X_true)
    
    preds = [1 if pred == -1 else 0 for pred in preds]
    
    pred_str = ""
    
    for pred in preds:
        pred_str += str(pred) + ","
    
    pred_str = pred_str[:-1]
    
    print(pred_str)
    
    
if __name__ == "__main__":
    # training()
    predicting()
