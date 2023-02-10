### IMPORTS ###

import numpy as np
import pandas as pd
import sys
import os


### CHECKING FOLDERS ###

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/general_classes", "")

### GLOBALS ###

clear = lambda : os.system("cls")


class DataPreperation():
    def __init__(self, path_data, numpy_bool = False):
        self.numpy_bool = numpy_bool
        try:
            if sys.platform == "darwin": # for macOS
                self.data = pd.read_csv(os.path.join(dirname, path_data)) 
            else:
                self.data = pd.read_csv(path_data)
        except OSError as e:
            print("FileNotFoundError: [Errno 2] No such file or directory")

        self.x_length = self.data.shape[0]
        self.y_length = self.data.shape[1]
        self.Y_train, self.X_train, self.X_test, self.Y_test = self.__create_data_sets()
        

    def __create_data_sets(self):
        train = self.data.sample(frac = .7, random_state=200)
        test = self.data.drop(train.index)

        Y_train = train["Lead"]
        Y_train = Y_train.replace("Female", -1)
        Y_train = Y_train.replace("Male", 1)
        X_train = train.drop("Lead", axis=1)


        Y_test = test["Lead"]
        Y_test = Y_test.replace("Female", -1)
        Y_test = Y_test.replace("Male", 1)
        X_test = test.drop("Lead", axis=1)

        # add visualization methods

        # CLEAR COLUMNS AND PREPARE DATA

        if self.numpy_bool:
            return Y_train.to_numpy(), X_train.to_numpy(), \
                    X_test.to_numpy(), Y_test.to_numpy()
        else:
            return Y_train, X_train, X_test, Y_test

    def __gender_to_bin(self):
        pass
        
