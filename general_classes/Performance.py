##########################################################
## IMPORTS

import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score,\
                            recall_score, confusion_matrix, f1_score,\
                            cohen_kappa_score, roc_curve

##########################################################
## GLOBALS

bar = "##################################################"

##########################################################
## PERFORMANCE CLASS

class Performance():
    """This class is for evaluating the performance of the different models"""
    def __init__(self, Y_pred, Y_true, gender=False):
        self.Y_pred = Y_pred
        self.Y_true = Y_true
        self.gender = gender
    
    ##########################################################
    ## PERFORMANCE METHODS
    
    def accuracy(self):
        return np.sum(self.Y_true == self.Y_pred) / len(self.Y_true)
    
    def balanced_accuracy(self):
        return balanced_accuracy_score(self.Y_true, self.Y_pred)
    
    def precision(self):
        male_precision = precision_score(self.Y_true, self.Y_pred, pos_label="Male" if self.gender else 1)
        female_precision = precision_score(self.Y_true, self.Y_pred, pos_label="Female" if self.gender else -1, zero_division=0)
        
        return (male_precision, female_precision)
    
    def recall(self):
        male_recall = recall_score(self.Y_true, self.Y_pred, pos_label="Male" if self.gender else 1)
        female_recall = recall_score(self.Y_true, self.Y_pred, pos_label="Female" if self.gender else -1)
        
        return (male_recall, female_recall)
    
    def confusion(self):
        return confusion_matrix(self.Y_true, self.Y_pred)
    
    def f1(self):
        male_f1 = f1_score(self.Y_true, self.Y_pred, pos_label="Male" if self.gender else 1)
        female_f1 = f1_score(self.Y_true, self.Y_pred, pos_label="Female" if self.gender else -1)
    
        return (male_f1, female_f1)

    def cohen(self):
        return cohen_kappa_score(self.Y_true, self.Y_pred)
    
    def roc(self):
        m_fpr, m_tpr, m_thres = roc_curve(self.Y_true, self.Y_pred, pos_label="Male" if self.gender else 1)
        f_fpr, f_tpr, f_thres = roc_curve(self.Y_true, self.Y_pred, pos_label="Female" if self.gender else -1)
        
        return ((m_fpr, m_tpr, m_thres), (f_fpr, f_tpr, f_thres))
    
    ##########################################################
    ## COLLECTION
    
    def combination(self, data):
        funcs = [self.accuracy, self.balanced_accuracy, self.precision,
                 self.recall, self.f1, self.cohen]
        
        for func, key in zip(funcs, data.keys()):
            data[key].append(func())
            
    
##########################################################
## OUTSIDE FUNCTIONS

# these are used to easily format and print the dicts containing the metrics

# printing the data dict
def print_combination(data):
    print("mean performance of the folds")
    print(bar)
    for key in data.keys():
        try:
            print(key + ":", round(sum(data[key])/len(data[key]),2))
        except ZeroDivisionError:
            print(key, "undefined due to strange parameters")  
        except:
            l = len(data[key])
            m = sum(val[0] for val in data[key])
            f = sum(val[1] for val in data[key])
            
            print(key + ":", round(m/l,2), round(f/l,2))
    print(bar)

# fetching a data dict
def get_dict():
    return {"accuracy":[], "balanced accuracy":[], "precision":[], "recall":[],
            "f1-score":[], "cohen kappa":[]}