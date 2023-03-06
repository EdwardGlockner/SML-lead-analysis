##########################################################
## IMPORTS
import numpy as np
import sys
import os
from imblearn.over_sampling import SMOTE

##########################################################
## FIXING PATH
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting/mains", "")
sys.path.insert(1,os.path.join(dirname, "general_classes"))

##########################################################
## LOCAL PACKAGES
from DataPreparation import DataPreparation
from Performance import Performance
sys.path.insert(1, os.path.join(dirname, "models/boosting"))
from AdaBoost import AdaBoost

##########################################################
## GLOBALS
clear = lambda : os.system("cls")

##########################################################
## MAIN
def main():

    # Fix data
    path_data = dirname + "/data/train.csv"
    drop_cols = []
    DataPrep = DataPreparation(path_data, numpy_bool = True)
    
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Create synthetic data
    sm = SMOTE(random_state = 42)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))
    
    # Merge all the data
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))
    
    # AdaBoost ML algortihm using 5 weak classifiers

    clf = AdaBoost(n_clf=5)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)

    # Analyze performance
    Perfor = Performance(y_pred, Y_test)
    accuracy = Perfor.accuracy()
    precision = Perfor.precision()
    recall = Perfor.recall()
    confusion = Perfor.confusion()
    f1 = Perfor.f1()
    cohen = Perfor.cohen()
    roc = Perfor.roc()

    print("Performance metrix\t\t") 
    print(f"Accuracy: \t{accuracy}")
    print(f"Precision: \t{precision}")
    print(f"Recall: \t{recall}")
    print(f"Confusion: \t{confusion}")
    print(f"f1: \t{f1}")
    print(f"Cohen: \t{cohen}")
    print(f"Roc: \t{roc}")


##########################################################
## RUN CODE
if __name__ == "__main__":
    main()


