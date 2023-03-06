##########################################################
## IMPORTS
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
import sys
import os
from sklearn.model_selection import KFold
import sklearn.metrics as skl_me
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

##########################################################
## FIXING PATH
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting/mains", "")
sys.path.insert(1, os.path.join(dirname, "general_classes"))

##########################################################
## LOCAL PACKAGES
from DataPreparation import DataPreparation
from Performance import Performance

##########################################################
## FUNCTIONS
def hyper_tuning():
    """ Tuning with GridSearch """
    
    # Hyperparameters that could be useful to tune
    
    parameters = {
    "learning_rate": np.linspace(0.01, 3, 20).tolist(),
    }
    # Get the data sets
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, numpy_bool = True)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Use SMOTE for over sampling
    sm = SMOTE(random_state = 42)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))

    # ML model
    model = GridSearchCV(AdaBoostClassifier(), parameters, cv=10, n_jobs=-1, verbose=1)
    model.fit(X_train, Y_train)
    print(model.best_params_)

def normal_pred():
 
    # Set data
    path_data = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path_data, numpy_bool = True, gender = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Use data augmentation
    sm = SMOTE(k_neighbors = 5)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))

    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit the model and make predictions
    model = AdaBoostClassifier(n_estimators=50, learning_rate = 1).fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    # Print performance
    print("AdaBoostClassifier")
    print(classification_report(Y_test, y_pred))


def cross_val():
    # Set data 
    path_data = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path_data, numpy_bool = True)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Use data augmentation
    sm = SMOTE(k_neighbors = 5)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((Y_train, Y_test))
    
    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Fit the model and use k-fold cross validation
    model = AdaBoostClassifier()

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_train, Y_train, scoring="accuracy", cv=cv, n_jobs=-1, error_score='raise')
     
    for i in range(len(n_scores)):
        print(f"Iteration {i}, accuracy:\t {n_scores[i]}")

    print(f"Mean accuracy:\t {np.mean(n_scores)}")
    """
    n_cols = X_train.shape[1]
    best_subset, best_score = None, 0.0
    # enumerate all combinations of input features
    for subset in product([True, False], repeat=n_cols):
        # convert into column indexes
        ix = [i for i, x in enumerate(subset) if x]
        # check for now column (all False)
        if len(ix) == 0:
            continue
            # select columns
        X_new = X_train[:, ix]
        # define model
        model = AdaBoostClassifier()
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X_new, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        # summarize scores
        result = mean(scores)
        # report progress
        print('>f(%s) = %f ' % (ix, result))
        # check if it is better than the best so far
        if best_score is None or result >= best_score:
            # better result
            best_subset, best_score = ix, result
            # report best
    print('Done!')
    print('f(%s) = %f' % (best_subset, best_score))
 
    """

def evaluation_cross_val(n_folds = 10):
    # Get the data sets
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, numpy_bool = True, gender = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Merge the data
    X = np.concatenate((X_train, X_test))
    Y = np.concatenate((Y_train, Y_test))

    # Performance metrics
    accuracy = np.zeros(n_folds)
    balanced_accuracy = np.zeros(n_folds)
    precision_F = np.zeros(n_folds)
    precision_M = np.zeros(n_folds)
    recall_F = np.zeros(n_folds)
    recall_M = np.zeros(n_folds)
    F1_M = np.zeros(n_folds)
    F1_F = np.zeros(n_folds)
    cohen_kappa = np.zeros(n_folds)  

    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    Ada = AdaBoostClassifier()
    
    cross_val = KFold(n_splits = n_folds, shuffle= True, random_state=False)

    for i, (index_train, index_val) in enumerate(cross_val.split(X)):     
        # cross_val.split() gives the indices for the training and validation data
        X_train_loop, X_val_loop = X[index_train], X[index_val]
        Y_train_loop, Y_val_loop = Y[index_train], Y[index_val]

        # Use SMOTE for over sampling
        sm = SMOTE(random_state = 42)
        X_res, Y_res = sm.fit_resample(X_train_loop, Y_train_loop)
        X_train_loop = np.concatenate((X_train_loop, X_res))
        Y_train_loop = np.concatenate((Y_train_loop, Y_res))
        
        model = Ada.fit(X_train_loop, Y_train_loop)

        Y_pred_loop = model.predict(X_val_loop)

        
        accuracy[i] = skl_me.accuracy_score(Y_val_loop, Y_pred_loop)
        balanced_accuracy[i] = skl_me.balanced_accuracy_score(Y_val_loop, Y_pred_loop)
        precision_M[i] = skl_me.precision_score(Y_val_loop, Y_pred_loop, pos_label=1)
        precision_F[i] = skl_me.precision_score(Y_val_loop, Y_pred_loop, pos_label = -1)

        recall_F[i] = skl_me.recall_score(Y_val_loop, Y_pred_loop, pos_label=-1)
        recall_M[i] = skl_me.recall_score(Y_val_loop, Y_pred_loop, pos_label=1)
        F1_M[i] = skl_me.f1_score(Y_val_loop, Y_pred_loop, pos_label=1)
        F1_F[i] = skl_me.f1_score(Y_val_loop, Y_pred_loop, pos_label=-1)

        cohen_kappa[i] = skl_me.cohen_kappa_score(Y_val_loop, Y_pred_loop)

    print(f"Mean accuracy: {np.mean(accuracy)}")
    print(f"Mean balanced accuracy: {np.mean(balanced_accuracy)}")
    print(f"Mean recall F: {np.mean(recall_F)}")
    print(f"Mean recall M: {np.mean(recall_M)}")
    print(f"Mean F1 F: {np.mean(F1_F)}")
    print(f"Mean F1 M: {np.mean(F1_M)}")
    print(f"Mean cohen kappa: {np.mean(cohen_kappa)}")
    print(f"Mean precision M: {np.mean(precision_M)}")
    print(f"Mean precision F: {np.mean(precision_F)}")

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




