############################################################
## IMPORTS

import os
import sys
from time import sleep

############################################################
## LOCAL PACKAGES 

print("Importing local packages", end="\r")
from data.show import show_data
print("Importing local packages.", end="\r")
from stat_analysis.number_words import words_user
from stat_analysis.stats import lead_user
print("Importing local packages..",end="\r")
from models.KNN.main import main as knn
from models.randomforest.main import main as rfc
from models.QDALDA.QDA import main as qda
from models.QDALDA.LDA import main as lda
from models.log_reg.LogReg import main as logr
from models.boosting.mains.ada_lib import main as ada
print("Importing local packages...", end="\r")
from models.boosting.mains.ada_scratch import main as adascratch
from models.boosting.mains.GDB import main as gdb
from models.boosting.mains.XGBOOST import main as xgb
from models.naive.naive import main as naive
print("Import of local packages done!")

############################################################
## GLOBALS

# function that clears the terminal
try:
    if sys.platform == "darwin": # macOS
        clear = lambda : os.system('clear')
    else:
        clear = lambda : os.system('cls')
except OSError as e:
    print("Error identifying operating system")

bar = "************************************************************"

############################################################
## FUNCTIONS

def menu():
    
    def _exit():
        pass
    
    clear()
    
    choice = ""
    
    choices = {"show": show_data, "data":_analy_menu,\
        "model": _model_menu, "exit":_exit}
    
    while choice != "exit":
        print("Select Action || Main Menu")
        print(bar)
        print(f"{'show data: ':{'.'}<30}{' show':{'.'}>30}")
        print(f"{'data analysis: ':{'.'}<30}{' data':{'.'}>30}")
        print(f"{'choose model: ':{'.'}<30}{' model':{'.'}>30}")
        print(f"{'exit the program: ':{'.'}<30}{' exit':{'.'}>30}")
        print(bar)
        
        choice = input("input: ").lower().strip()
        
        if choice in choices.keys():
            choices[choice]()
        else:
            print("\nincorrect input")
            sleep(1.5)
        
        clear()
        
def _analy_menu():
    
    def _back(dummy):
        pass
    
    def _describe(dummy):
        clear()
        print(bar)
        print("the input is in the format: \"type flag1 flag2 ...\"")
        print("where type defines which one of the analyses will be")
        print("performed and the flags determine which plots will be")
        print("generated (superflous flags will be discarded), no flags")
        print("will generate all plots. the plots are saved in the")
        print("directory \"stat_analysis/graphs\"")
        print(bar)
        input("press enter to go back")
    
    clear()
    
    key = ""
    
    choices = {"words": words_user, "lead":lead_user,\
        "info": _describe, "back":_back}
    
    while key != "back":
        print("Select Action || Data Analysis")
        print(bar)
        print(f"{'words analysis: ':{'.'}<30}{' type: words':{'.'}>30}")
        print("\t\tflags: gender time gross", "\n")
        print(f"{'lead analysis: ':{'.'}<30}{' type: lead':{'.'}>30}")
        print("\t\tflags: gender time gross", "\n")
        print(f"{'describe the input: ':{'.'}<30}{' info':{'.'}>30}")
        print(f"{'go back: ':{'.'}<30}{' back':{'.'}>30}")
        print("input: type flag1 flag2...")
        print(bar)
        
        choice = list(input("input: ").lower().strip().split(" "))
        
        key = choice.pop(0)
        
        choice = set(choice)
        
        if key in choices.keys():
            choices[key](choice)
        else:
            print("\nincorrect input")
            sleep(1.5)
        
        clear()

def _model_menu():
    
    def _back():
        pass
    
    clear()
    
    choice = ""
    
    choices = {"log": logr, "rfc":rfc,\
        "knn": knn, "qda":qda, "lda": lda, "ada": ada, "adascratch":adascratch,\
        "gdb": gdb, "xgb": xgb, "naive": naive, "back": _back}
    
    while choice != "back":
        print("Select Action || Model Selection")
        print(bar)
        print(f"{'Logarithmic regression: ':{'.'}<30}{' log':{'.'}>30}")
        print(f"{'Random forest classifier: ':{'.'}<30}{' rfc':{'.'}>30}")
        print(f"{'K-NN: ':{'.'}<30}{' knn':{'.'}>30}")
        print(f"{'QDA: ':{'.'}<30}{' qda':{'.'}>30}")
        print(f"{'LDA: ':{'.'}<30}{' lda':{'.'}>30}")
        print(f"{'AdaBoost: ':{'.'}<30}{' ada':{'.'}>30}")
        print(f"{'AdaBoost (scratch): ':{'.'}<30}{' adascratch':{'.'}>30}")
        print(f"{'GDBoost: ':{'.'}<30}{' gdb':{'.'}>30}")
        print(f"{'XGBoost: ':{'.'}<30}{' xgb':{'.'}>30}")
        print(f"{'Naive: ':{'.'}<30}{' naive':{'.'}>30}")
        print(f"{'go back: ':{'.'}<30}{' back':{'.'}>30}")
        print(bar)
        
        choice = input("input: ").lower().strip()
        
        if choice in choices.keys():
            clear()
            choices[choice]()
            if choice != "back":
                input("press enter to continue")
        else:
            print("\nincorrect input")
            sleep(1.5)
        
        clear()


############################################################
## MAIN

def main():
    menu()

############################################################
## RUN CODE

if __name__ == "__main__":
    main()
