import numpy as np
from scipy.special import gamma, gammaincc

def misra_m(y, m):
    """
    Compute f_m(y) = ∫_1^∞ x^m exp(-x y) dx
    for y > 0 and real m.
    """
    y = np.asarray(y)
    return gamma(m + 1) * gammaincc(m + 1, y) / y**(m + 1)

# Functions below are only used for testing

def misra_m_quad(y, m):
    from scipy.integrate import quad
    integrand = lambda x: x**m * np.exp(-x*y)
    return quad(integrand, 1, np.inf)[0]

def misra_n_integer(y, n):
    from math import factorial
    y = np.asarray(y)
    s = sum(y**k / factorial(k) for k in range(n + 1))
    return factorial(n) * np.exp(-y) * s / y**(n + 1)
