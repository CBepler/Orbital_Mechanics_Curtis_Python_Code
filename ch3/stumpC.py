import math

def stumpC(z):
    """
    This function evaluates the Stumpff function C(z) according
    to Equation 3.50.
    
    Parameters:
    z - input argument
    c - value of C(z)
    
    User M-functions required: none
    """
    
    if z > 0:
        c = (1 - math.cos(math.sqrt(z))) / z
    elif z < 0:
        c = (math.cosh(math.sqrt(-z)) - 1) / (-z)
    else:
        c = 1/2
    
    return c