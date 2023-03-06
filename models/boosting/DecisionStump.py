##########################################################
## IMPORTS
import numpy as np

##########################################################
## DECISIONSTUMP CLASS
class DecisionStump():
    def __init__(self):
        self.polarity = 1 # sample should be classified as -1 or +1 for the given threshold
        self.feature_index = None
        self.threshold = None # split threshold
        self.alpha = None # performance


    def predict(self, X):
        """
        X: sample that should be predicted (pandas dataframe) 
        """
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index] # only this feature

        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1

        else: # -1
            predictions[X_column > self.threshold] = -1

        return predictions

