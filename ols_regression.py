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
    @param mean_y: optional, expectation of sequence y
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
    @raise Exception: if input sequences are ill defined
    """
    # check inputs are of equi-length
    n = len(x)
    if n != len(y):
        raise Exception("Inputs x and y should be of same length.")
    
    if n < 2:
        raise Exception("Input sequence should have length greater than one.")
        
    mean_x = sum(x)/n
    mean_y = sum(y)/n
    var_x = covar(x,x,mean_x,mean_x)
    
    if var_x == 0:
        raise Exception("OLS Regressor cannot have zero variance.")
    
    beta = covar(x,y,mean_x,mean_y)/var_x
    alpha = mean_y - beta*mean_x
    
    return (beta, alpha)


import unittest, random
import numpy as np
from sklearn.linear_model import LinearRegression

class TestOLSRegression(unittest.TestCase):
    def test_ols_regression(self):
        x = [random.random() for _ in range(10)]
        seed = random.randrange(1, 50, 1)
        y = [random.random() + seed*v for v in x]
        
        fit = LinearRegression().fit(np.array(x).reshape(-1,1), np.array(y).reshape(-1,1))
        actual_beta = fit.coef_[0][0]
        actual_alpha = fit.intercept_[0]
        
        test_beta, test_alpha = ols_estimator(x,y)
        
        self.assertEqual(round(actual_alpha,6), round(test_alpha, 6), 'Intercept estimate incorrect.')
        self.assertEqual(round(actual_beta,6), round(test_beta, 6), 'Slope estimate incorrect.')

        
def main():
    print('Running OLS Regression test...')
    test = TestOLSRegression()
    test.test_ols_regression()
    print('Test complete.')

if __name__ == '__main__':
    main()