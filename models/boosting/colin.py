##########################################################
## IMPORTS
from sklearn.feature_selection import f_regression
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(rc={'figure.figsize':(12,8)})
import os
import sys
from collinearity import SelectNonCollinear

##########################################################
## FIXING PATH
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting", "")
sys.path.insert(1,os.path.join(dirname, "general_classes"))

##########################################################
## LOCAL PACKAGES
from DataPreparation import DataPreparation

"""
Plot of colinearity

"""
path_data = dirname + "/data/train.csv"
drop_cols = []
DataPrep = DataPreparation(path_data, numpy_bool = False, drop_cols = drop_cols, gender=False)
X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

features = X_train.columns

df = pd.DataFrame(X_train,columns=features)
sns.heatmap(df.corr().abs(),annot=True)

import matplotlib.pyplot as plt
plt.show()
