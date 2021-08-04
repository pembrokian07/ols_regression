from collections.abc import Sequence
from numbers import Number
from typing import Tuple

def covar(x: Sequence, y: Sequence, mean_x: Number=None, mean_y: Number=None) -> Number:
    """
    Computes the sample covariance between two sequences of numerics. 
    The two sequences must be of same length.
    
    @param x: sequence of numeric type
    @param y: sequence of numeric type
    @param mean_x: optional, expectation of sequence x
    @param mean_y: optionql, expectation of sequence y
    @return: covariance of x and y
    @raise Exception: if x and y are not of equi-length
    """
    # check inputs are of equi-length
    n = len(x)
    if n != len(y):
        raise Exception("Inputs x and y should be of same length.")
    
    mean_x = sum(x)/n if mean_x is None else mean_x
    mean_y = sum(y)/n if mean_y is None else mean_y
    
    return sum([(x[i] - mean_x)*(y[i] - mean_y) for i in range(0,n)])/(n-1)


def ols_estimator(x: Sequence, y: Sequence) -> Tuple[float, float]:
    """
    Estimate univariate OLS linear regression coefficients for y = a + b*x.
    
    @param x: regressor, sequence of numeric type
    @param y: regressand, sequence of numeric type
    @return: a tuple of (slope, intercept)
    @raise Exception: if x and y are not of equi-length
    """
    # check inputs are of equi-length
    n = len(x)
    if n != len(y):
        raise Exception("Inputs x and y should be of same length.")
        
    mean_x = sum(x)/n
    mean_y = sum(y)/n
    
    beta = covar(x,y,mean_x,mean_y)/covar(x,x,mean_x,mean_x)
    alpha = mean_y - beta*mean_x
    
    return (beta, alpha)


# test case 1:
# a = range(0,10)
# b = range(1,11)
# ols_estimator(a,b)


# test case 2:
# a = [10,3,5,9]
# b = [7,5,6,7]
# ols_estimator(a,b)