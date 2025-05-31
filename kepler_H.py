#solution of Kepler’s equation for the hyperbola using Newton’s method

import math

def kepler_H(e, M):
    """
    This function uses Newton's method to solve Kepler's
    equation for the hyperbola  e*sinh(F) - F = M  for the
    hyperbolic eccentric anomaly, given the eccentricity and
    the hyperbolic mean anomaly.
    
    Parameters:
    F - hyperbolic eccentric anomaly (radians)
    e - eccentricity, passed from the calling program
    M - hyperbolic mean anomaly (radians), passed from the
        calling program
    
    User M-functions required: none
    """
    
    # Set an error tolerance:
    error = 1.0e-8
    
    # Starting value for F:
    F = M
    
    # Iterate on Equation 3.42 until F is determined to within
    # the error tolerance:
    ratio = 1
    while abs(ratio) > error:
        ratio = (e * math.sinh(F) - F - M) / (e * math.cosh(F) - 1)
        F = F - ratio
    
    return F

if __name__ == "__main__":
    e = 2.7696
    M = 40.690
    F = kepler_H(e, M)
    print(f"Hyperbolic Eccentric Anomaly F: {F:.4f} radians")