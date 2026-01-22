#solution of Kepler’s equation by Newton’s method

import math

def kepler_E(e, M):
    """
    This function uses Newton's method to solve Kepler's
    equation  E - e*sin(E) = M  for the eccentric anomaly,
    given the eccentricity and the mean anomaly.
    
    Parameters:
    e  - eccentric anomaly (radians)
    e  - eccentricity, passed from the calling program
    M  - mean anomaly (radians), passed from the calling program
    pi - 3.141592...
    
    User M-functions required: none
    """
    
    # Set an error tolerance:
    error = 1.0e-8
    
    # Select a starting value for E:
    if M < math.pi:
        E = M + e/2
    else:
        E = M - e/2
    
    # Iterate on Equation 3.14 until E is determined to within
    # the error tolerance:
    ratio = 1
    while abs(ratio) > error:
        ratio = (E - e*math.sin(E) - M) / (1 - e*math.cos(E))
        E = E - ratio
    
    return E

if __name__ == "__main__":
    M = 3.6029
    e = 0.37255
    E = kepler_E(e, M)
    print(f"Eccentric Anomaly E: {E:.4f} radians")
