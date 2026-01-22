import math

def stumpS(z):
    """
    This function evaluates the Stumpff function S(z) according
    to Equation 3.49.
    
    Parameters:
    z - input argument
    s - value of S(z)
    
    User M-functions required: none
    """
    
    if z > 0:
        s = (math.sqrt(z) - math.sin(math.sqrt(z))) / (math.sqrt(z))**3
    elif z < 0:
        s = (math.sinh(math.sqrt(-z)) - math.sqrt(-z)) / (math.sqrt(-z))**3
    else:
        s = 1/6
    
    return s