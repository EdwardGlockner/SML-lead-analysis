##########################################################
## IMPORTS

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

##########################################################
## FIXING PATH

import sys
sys.path.append('.')

##########################################################
## LOCAL PACKAGES

from general_classes import *

##########################################################
## GRIDSEARCH

def tuning():
    
    dp = DataPreparation("./data/train.csv", clean=True)
    
    # combines the data due to the usage of k_fold
    X_train, X_test, Y_train, Y_test = dp.get_sets()    
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((Y_train, Y_test))
    
    # Hyperparameters that could be useful to tune
    criterion = ["gini", "entropy", "log_loss"]
    max_depth = [60, 70, 80, 90, 100, None]
    min_samples_split = [2,5,10]
    min_samples_leaf = [1,2,4]
    min_weight_fraction_leaf = [0.0, 0.5, 0.7, 1]
    # max_features = ["sqrt", "auto", None]
    max_leaf_nodes = [100, 200, None]
    min_impurity_decrease = [0.0, 0.5, 0.7, 1]

    hyperpara = dict(criterion=criterion, max_depth = max_depth, min_samples_split = min_samples_split,\
        min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf, max_leaf_nodes = max_leaf_nodes,\
        min_impurity_decrease = min_impurity_decrease)
    
    print("tuning...")
    model = GridSearchCV(RandomForestClassifier(n_estimators=100,bootstrap=True, max_features="auto"), hyperpara, cv=10, refit='balanced_accuracy', verbose=1, n_jobs=-1)     

    model.fit(X_train, Y_train)
    print("Tuned parameters: ")
    print(model.best_params_)
    
    
##########################################################
## MAIN

def main():
    # setting up dataprep instance
    dp = DataPreparation("./data/train.csv", clean=True)
    
    # combines the data due to the usage of k_fold
    X_train, X_test, Y_train, Y_test = dp.get_sets()    
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((Y_train, Y_test))
    
    # normalizing the data
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    
    # k-fold parameters
    N = 20
    kf = KFold(n_splits=N,random_state=None, shuffle=False)
    
    data = get_dict()   # data dict
    
    # the k-fold loops
    for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
        # visualizing progress to not drive the user mad
        print("Calculating ",f"{round((fold+1)/N*100,2)}%",end="\r")
        
        # k-fold selected data
        temp_X = X_train[train_index]
        temp_Y = Y_train[train_index]
        
        # SMOTE algo on the selected data
        sm = SMOTE(k_neighbors=5)
        X_res, Y_res = sm.fit_resample(temp_X,temp_Y)
        
        # this makes sure it is not validated using synthetic data
        temp_x_test = X_train[test_index]
        temp_y_test = Y_train[test_index]
        
        # adding the synthetic data to the training sets
        temp_X = np.concatenate((temp_X, X_res))
        temp_Y = np.concatenate((temp_Y, Y_res))
        
        # RFC using max cores
        model = RandomForestClassifier(n_estimators=100,n_jobs=-1, bootstrap=True, criterion="gini", max_depth=60, max_leaf_nodes=100, min_samples_leaf=2)        
        model.fit(temp_X, temp_Y)
        
        # predicting using the SMOTE-trained model on non-SMOTEd data
        y_pred = model.predict(temp_x_test)
        
        # setting up the perf instance
        perf = Performance(y_pred, temp_y_test)
        perf.combination(data)

    #############################
    # end of k-fold loop
    
    print("\r\n")
    print_combination(data)

##########################################################
## RUN CODE

if __name__ == "__main__":
    main()
    # tuning()