############################################################
## IMPORTS

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics as stat
import os

from sys import platform

############################################################
## FIXING PATH

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/stat_analysis", "")

############################################################
## LOCAL PACKAGES

import general_methods as gm

############################################################
## GLOBALS

# function that clears the terminal
clear = lambda : os.system('cls')

# converts the strings into binaries. 1 for "Female", 0 for everything else
gender_to_bin = lambda g : int(bool(g == "Female"))

bar = "************************************************************"

############################################################
## FUNCTIONS

def bin_roles(data):
    """This function evaluates the relationship between major speaking roles and actor gender"""
    
    numb_actors = data[["Number of male actors", "Number of female actors"]].values.tolist()
    
    actors_M = [entry[0] for entry in numb_actors]
    numb_M = sum(actors_M)
    actors_F = [entry[1] for entry in numb_actors]
    numb_F = sum(actors_F)
    
    roles = []
    
    for M, F in zip(actors_M, actors_F):
        roles += [-1]*M + [1]*F
    
    mu = stat.mean(roles)
    
    sigma = stat.stdev(roles)
    
    N = stat.NormalDist(mu, sigma)
    
    print(f"{'female roles: ':{'.'}<30}{f' {numb_F}':{'.'}>30}")
    print(f"{'male roles: ':{'.'}<30}{f' {numb_M}':{'.'}>30}")
    print(f"{'total roles: ':{'.'}<30}{f' {numb_M + numb_F}':{'.'}>30}")
    print(bar, "\n")
    print("APPROXIMATION TO NORMAL DISTRIBUTION")
    print(bar)
    
    meanstr = f"mean [\u03BC]: "
    stdstr = f"standard deviation [\u03C3]: "
    invar = "P(gender \u2208 [0,1]): " 
    
    print(f"{meanstr:{'.'}<30}{f' {round(mu, 3)}':{'.'}>30}")
    print(f"{stdstr:{'.'}<30}{f' {round(sigma, 3)}':{'.'}>30}")
    print(f"{invar:{'.'}<30}{f' {round(1-N.cdf(0),3)}':{'.'}>30}")
    print(bar)
    
    x = list(np.linspace(-1,1,200))
    y = [N.pdf(xt) for xt in x]
    
    print("saving graph...")
    
    # PLOTTING
    ########################################################
    plt.figure(1)
    plt.plot(x,y)
    plt.axvline(x = 0, color = 'r', linestyle="--")
    plt.grid()
    plt.xticks([-1 + 0.2*k for k in range(11)])
    plt.xlabel("Binary gender [Male,Female]=[-1,1]")
    plt.ylabel("Probability density")
    plt.title("Patriarchy            Equality            Matriarchy")
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/bin-roles.png"))
    ########################################################
    print("graph saved as bin-roles.png")
    print(bar)
    
def roles_years(data):
    """This function evaluates the relationship between speaking role gender and time"""
    
    role_data = data[["Number of male actors", "Number of female actors", "Year"]].sort_values(by="Year").values.tolist()
    years = [entry[2] for entry in role_data]
    
    dic = {}
    
    for year, dat in zip(years, role_data):
        try:
            dic[year][0] += dat[0]
            dic[year][1] += dat[1]
        except:
            dic[year] = [dat[0], dat[1]]

    frac = [dat[1]/(dat[0]+dat[1]) for dat in dic.values()]
    
    x_years = list(dic.keys())
    
    # linear regression for the fraction of female leads
    # using all of the data
    theta1 = gm.lin_reg(x_years, frac)[0]
    x1 = [x_years[0], x_years[-1]]
    y1 = [theta1[0] + x1[0]*theta1[1], theta1[0]+x1[1]*theta1[1]]
    
    # using all of the data after 1980 (due to the few samples from earlier dates)
    theta2 = gm.lin_reg(x_years[15:], frac[15:])[0]
    x2 = [x_years[15], x_years[-1]]
    y2 = [theta2[0] + x2[0]*theta2[1], theta2[0]+x2[1]*theta2[1]]
    
    print("saving graph...")
    
    # PLOTTING
    ########################################################
    # the fraction of female leads, and two linear regressions
    plt.figure(2)
    plt.plot(x_years, frac, "--k", x1, y1, "r", x2, y2, "b")
    
    plt.legend(["Data", "1939-2015", "1980-2015"])
    plt.grid()
    plt.title("Fraction of female speaking roles over time")
    plt.xlabel("Year [1939-2015]")
    plt.ylabel("Fraction of female roles [%]")
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/roles-years.png"))
    ########################################################
    
    print("graph saved as roles-years.png")
    print(bar)
    
    
def words_gross(data):
    """This function evaluates the relationship between the grossing of the movie and the amount of female words"""
    
    role_data = data[["Total words", "Number words female", "Gross"]].sort_values(by="Number words female").values.tolist()
    gross = [entry[2] for entry in role_data]
    words_F = [entry[1] for entry in role_data]
    tots = [entry[0] for entry in role_data]
    
    lims = list(range(0,18000,500))
    newvals = [0 for _ in range(len(lims))]
    
    for i in range(len(lims)-1):
        l = 0
        for t, g, w in zip(tots, gross, words_F):
            if lims[i] <= w < lims[i+1]:
                newvals[i] += g
                l += 1
        try:
            newvals[i] /= l
        except:
            newvals[i] = -1
    
    newnewvals = [val for val in newvals if val != -1]
    newlims = [lim for lim, val in zip(lims, newvals) if val != -1]
    
    theta1 = gm.lin_reg(newlims, newnewvals)[0]
    x = [newlims[0], newlims[-1]]
    y = [theta1[0] + x[0]*theta1[1], theta1[0]+x[1]*theta1[1]]
    
    print("saving graph 1...")
    
    # PLOTTING
    ########################################################
    plt.figure(3)
    plt.scatter(newlims, newnewvals)
    plt.plot(x,y, "--k")
    plt.grid()
    plt.xticks(list(range(0,18000,2000)))
    plt.title("Grossing over number words female")
    plt.xlabel("Number words female")
    plt.ylabel("Gross")
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/words-gross-sing.png"))
    ########################################################
    
    print("graph 1 saved as words-gross-sing.png")
    print(bar)
    
    words_gross_M(data)
    
def words_gross_M(data):
    """This function evaluates the relationship between the grossing of the movie and the amount of male words"""
    
    role_data = data[["Total words", "Number words male", "Gross"]].sort_values(by="Number words male").values.tolist()
    gross = [entry[2] for entry in role_data]
    words_F = [entry[1] for entry in role_data]
    tots = [entry[0] for entry in role_data]
    
    lims = list(range(0,18000,500))
    newvals = [0 for _ in range(len(lims))]
    
    for i in range(len(lims)-1):
        l = 0
        for t, g, w in zip(tots, gross, words_F):
            if lims[i] <= w < lims[i+1]:
                newvals[i] += g
                l += 1
        try:
            newvals[i] /= l
        except:
            newvals[i] = -1
    
    newnewvals = [val for val in newvals if val != -1]
    newlims = [lim for lim, val in zip(lims, newvals) if val != -1]
    
    theta1 = gm.lin_reg(newlims, newnewvals)[0]
    x = [newlims[0], newlims[-1]]
    y = [theta1[0] + x[0]*theta1[1], theta1[0]+x[1]*theta1[1]]
    
    print("saving graph 2...")
    
    # PLOTTING
    ########################################################
    plt.figure(4)
    plt.scatter(newlims, newnewvals)
    plt.plot(x,y, "--k")
    plt.grid()
    plt.xticks(list(range(0,18000,2000)))
    plt.title("Grossing over number words male")
    plt.xlabel("Number words male")
    plt.ylabel("Gross")
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/words-gross-male.png"))
    ########################################################
    
    print("graph 2 saved as words-gross-male.png")
    print(bar)
    
    words_frac(data)

def words_frac(data):
    """This function evaluates the relationship between the grossing of the movie and the fraction of female words"""
    
    role_data = data[["Total words", "Number words female", "Gross", "Lead", "Number of words lead"]].values.tolist()
    gross = [entry[2] for entry in role_data]
    frac = [entry[1]/entry[0]  if entry[3] == "Male" else (entry[1] + entry[4])/entry[0] for entry in role_data]
    
    lims = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    newvals = [0 for _ in range(len(lims))]
    
    for i in range(len(lims)-1):
        l = 0
        for g, f in zip(gross, frac):
            if lims[i] <= f < lims[i+1]:
                newvals[i] += g
                l += 1
        try:
            newvals[i] /= l
        except:
            newvals[i] = -1
    
    newnewvals = [val for val in newvals if val != -1][0:-1]
    newlims = [lim for lim, val in zip(lims, newvals) if val != -1][0:-1]
    
    theta1 = gm.lin_reg(newlims, newnewvals)[0]
    x = [newlims[0], newlims[-1]]
    y = [theta1[0] + x[0]*theta1[1], theta1[0]+x[1]*theta1[1]]
    
    print("saving graph 3...")
    
    # PLOTTING
    ########################################################
    plt.figure(5)
    plt.scatter(newlims, newnewvals)
    plt.plot(x,y, "--k")
    plt.grid()
    plt.xticks(lims)
    plt.title("Grossing over fraction of female words")
    plt.xlabel("Fraction of total words by females")
    plt.ylabel("Gross")
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/words-gross-frac.png"))
    ########################################################
    
    print("graph 3 saved as words-gross-frac.png")
    print(bar)
    
    words_frac_M(data)
    
def words_frac_M(data):
    """This function evaluates the relationship between the grossing of the movie and the fraction of male words"""
    
    role_data = data[["Total words", "Number words male", "Gross", "Lead", "Number of words lead"]].values.tolist()
    gross = [entry[2] for entry in role_data]
    frac = [entry[1]/entry[0]  if entry[3] == "Female" else (entry[1] + entry[4])/entry[0] for entry in role_data]
    
    lims = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    newvals = [0 for _ in range(len(lims))]
    
    for i in range(len(lims)-1):
        l = 0
        for g, f in zip(gross, frac):
            if lims[i] <= f < lims[i+1]:
                newvals[i] += g
                l += 1
        try:
            newvals[i] /= l
        except:
            newvals[i] = -1
    
    newnewvals = [val for val in newvals if val != -1][0:-1]
    newlims = [lim for lim, val in zip(lims, newvals) if val != -1][0:-1]
    
    theta1 = gm.lin_reg(newlims, newnewvals)[0]
    x = [newlims[0], newlims[-1]]
    y = [theta1[0] + x[0]*theta1[1], theta1[0]+x[1]*theta1[1]]
    
    print("saving graph 4...")
    
    # PLOTTING
    ########################################################
    plt.figure(6)
    plt.scatter(newlims, newnewvals)
    plt.plot(x,y, "--k")
    plt.grid()
    plt.xticks(lims)
    plt.title("Grossing over fraction of male words")
    plt.xlabel("Fraction of total words by males")
    plt.ylabel("Gross")
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/words-gross-frac-male.png"))
    ########################################################
    
    print("graph 4 saved as words-gross-frac-male.png")
    print(bar)

############################################################
## FOR USER INPUT

def words_user(choices):
    """This function is for the user input from the ../main.py"""
    
    def _gender(data):
        print("### MAJOR ROLE BY GENDER ###")
        print(bar)
        print(bar)
        bin_roles(data)
        print(bar, "\n")
    
    def _time(data):
        print("### ROLE BALANCE OVER TIME ###")
        print(bar)
        print(bar)
        roles_years(data)
        print(bar, "\n")
    
    def _gross(data):
        print("### WORDS SPOKEN AND GROSSING ###")
        print(bar)
        print(bar)
        words_gross(data)
        print(bar, "\n")
    
    clear()
    
    flags = {"gender": _gender, "time": _time, "gross": _gross}
    
    if platform == "darwin": # check for mac os
        training_data = pd.read_csv(os.path.join(dirname, "data/train.csv"))

    else:
        training_data = pd.read_csv("data/train.csv")

    print("Statistical Analysis of the Training Data")
    print("")
    
    if len(choices) == 0:
        _gender(training_data)
        _time(training_data)
        _gross(training_data)
    else:
        for flag in choices:
            if flag in flags.keys():
                flags[flag](training_data)
                
    input("press enter to continue")

############################################################
## MAIN

def main():
    
    if platform == "darwin": # check for mac os
        training_data = pd.read_csv(os.path.join(dirname, "data/train.csv"))

    else:
        training_data = pd.read_csv("data/train.csv") 

    print("STATISTICAL ANALYSIS OF THE TRAINING DATA")
    print("")
    print("### MAJOR ROLE BY GENDER ###")
    print(bar)
    print(bar)
    bin_roles(training_data)
    print(bar, "\n")
    print("### ROLE BALANCE OVER TIME ###")
    print(bar)
    print(bar)
    roles_years(training_data)
    print(bar, "\n")
    print("### WORDS SPOKEN AND GROSSING ###")
    print(bar)
    print(bar)
    words_gross(training_data)
    print(bar, "\n")
############################################################
## RUN CODE

if __name__ == "__main__":
    main()