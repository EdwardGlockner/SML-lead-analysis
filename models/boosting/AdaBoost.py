##########################################################
## IMPORTS

import numpy as np
from DecisionStump import DecisionStump

##########################################################
## ADABOOST CLASS
class AdaBoost():
    def __init__(self,n_clf=5):
        """
        n_clf: number of classifiers (integer)
        """
        self.n_clf = n_clf


    def fit(self, X, Y):
        """
        X: training samples (pandas dataframe)
        Y: lables (pandas dataframe)
        """
        n_samples, n_features = X.shape

        # initialize the weights
        w = np.full(n_samples, (1/n_samples))
        
        # training begins
        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf') # want to minimize this on features and thresholds

            for feature_indx in range(n_features):
                X_column = X[:, feature_indx]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    polarity = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    
                    # error optimization
                    missclassified = w[Y != predictions] # missclassified weights
                    error = sum(missclassified)

                    
                    if error > 0.5:
                        error = 1 - error
                        polarity = -1

                    if error < min_error:
                        min_error = error

                        clf.polarity = polarity
                        clf.threshold = threshold
                        clf.feature_index = feature_indx

            epsilon = 1e-16 # don't divide by zero

            clf.alpha = 0.5 * np.log((1 - error) / (error + epsilon))

            predicts = clf.predict(X)
            w = w * np.exp(-clf.alpha * Y * predicts)
            w = w * np.sum(w) # normalize

            self.clfs.append(clf)


    def predict(self, X):
        """
        X: sample (pandas dataframe)
        """
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        Y_pred = np.sum(clf_preds, axis=0)
        Y_pred = np.sign(Y_pred)

        return Y_pred

