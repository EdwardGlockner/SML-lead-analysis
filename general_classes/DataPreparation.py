##########################################################
## IMPORTS

import pandas as pd
import sys
import os

##########################################################
## FIXING PATH

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/general_classes", "")

##########################################################
## GLOBALS

clear = lambda : os.system("cls")

##########################################################
## DataPreparation CLASS

class DataPreparation():
    def __init__(self, path_data, numpy_bool = False, gender=False,
                 random = False, clean=True, custom = False, test=False):
        """
        :param str path_data: absolute path to data
        :param bool numpy_bool: convert to numpy.ndarray or keep as pandas
        :param bool gender: keep gender labels as strings or not
        :param bool random: using a random seed for the validation set or not
        :param bool clean: clean colinear data or not
        :param bool custom: use custom parameters for the data features or not
        
        """
        
        # class attributes
        self.numpy_bool = numpy_bool
        self.gender = gender
        self.random = random

        # for OS-check
        try:
            if sys.platform == "darwin": # for macOS
                self.data = pd.read_csv(os.path.join(dirname, path_data)) 
            else:
                self.data = pd.read_csv(path_data)
        except OSError as e:
            print("FileNotFoundError: [Errno 2] No such file or directory")

        # preparing the data
        if clean:
            self.__limit_vars()
        elif custom:
            self.__customized_vars()
            
        if test:
            self.X_true = self.__create_pred_sets()
        else:
            self.x_length = self.data.shape[0]
            self.y_length = self.data.shape[1]
            self.Y_train, self.X_train, self.X_test, self.Y_test = self.__create_data_sets()
    
    ##########################################################
    ## SECRET METHODS
    
     # for custom variables
    def __customized_vars(self):
        pass
    
    def __limit_vars(self):
        """
        this selection was done since the VIF of the features were quite large
        (logically so), thus theyre combined so that as little information is lost
        """
        self.data["Diff age"] = self.data["Mean Age Female"]/self.data["Mean Age Male"]
        self.data["Lead over mean age"] = self.data["Age Lead"]/(self.data["Mean Age Female"] + self.data["Mean Age Male"])*2
        """
        logically, the amount of words features were going to be colinear, as seen by the
        VIF-factors, thus theyre combined into two different features
        (the fractions for lead and male have VIFs of ~<7, which is quite bad but
        since some sources say VIFs<10 are acceptable and since we dont want to discard too
        much data, theyre accepted as is)
        """
        self.data["Frac female words"] = self.data["Number words female"]/self.data["Total words"]
        self.data["Frac male words"] = self.data["Number words male"]/self.data["Total words"]
        
        # if this turns out to increase k-fold accuracy, it stays
        self.data["Difference in words frac"] = self.data["Difference in words lead and co-lead"]/self.data["Total words"]
        self.data["Frac female actors"] = self.data["Number of female actors"]/self.data["Number of male actors"]
        
        """
        the feature Year is omitted entirely partly due to it being multicolinear with 
        features and partyl since it seems to have no large impact on the classification
        """
        self.data = self.data.drop(["Age Lead", "Mean Age Male", "Mean Age Female", "Total words",
                                    "Difference in words lead and co-lead", "Year", "Gross",
                                    "Number words female", "Number words male"],axis=1)


    def __create_data_sets(self):
        if self.random:
            train = self.data.sample(frac= .7)
        else:
            train = self.data.sample(frac = .7, random_state=10)
        test = self.data.drop(train.index)

        Y_train = train["Lead"]
        X_train = train.drop("Lead", axis=1)
        
        if not self.gender:
            Y_train = Y_train.replace("Female", -1)
            Y_train = Y_train.replace("Male", 1)

        Y_test = test["Lead"]
        X_test = test.drop("Lead", axis=1)
        
        if not self.gender:
            Y_test = Y_test.replace("Female", -1)
            Y_test = Y_test.replace("Male", 1)
        
        if self.numpy_bool:
            return Y_train.to_numpy(), X_train.to_numpy(), \
                    X_test.to_numpy(), Y_test.to_numpy()
        else:
            return Y_train, X_train, X_test, Y_test
        
    def __create_pred_sets(self):
        return self.data

    ##########################################################
    ## ACCESSIBLE METHODS

    def get_sets(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def raw(self):
        X = self.data.drop(columns=['Lead'])
        Y = self.data['Lead']
        return X, Y
