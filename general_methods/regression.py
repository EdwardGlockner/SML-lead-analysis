import numpy as np
import math

def lin_reg(xdata, ydata):
    """
    assumes one set of xdata
    xdata = [data]
    ydata can be multidimensional:
    ydata = [[set1], [set2], etc...]
    """
    
    ones = np.ones(len(xdata))
    
    Xtemp = np.array([xdata])
    Xt = np.vstack((ones, Xtemp))   # X'
    
    X = Xt.T
    Y = np.array([ydata]).T
    Z = np.linalg.inv(Xt@X)
    
    return [Z@Xt@Y, X]       # (X'X)**(-1)X'Y


## deprecated function
def log_reg(xdata, ydata):
    
    def local_exponent(theta, x):
        exponent = math.exp(theta[0]+theta[1]*x)
        
        return exponent/(1+exponent)
    
    [theta, X] = lin_reg(xdata, ydata)    
    
    return [local_exponent(theta,x) for x in xdata]
    
if __name__ == "__main__":
    pass