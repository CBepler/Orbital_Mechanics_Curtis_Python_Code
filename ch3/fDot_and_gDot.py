import math

from stumpC import stumpC
from stumpS import stumpS

def fDot_and_gDot(x, r, ro, a, mu):
    """
    This function calculates the time derivatives of the
    Lagrange f and g coefficients.
    
    Parameters:
    mu   - the gravitational parameter (km^3/s^2)
    a    - reciprocal of the semimajor axis (1/km)
    ro   - the radial position at time t (km)
    t    - the time elapsed since initial state vector (s)
    r    - the radial position after time t (km)
    x    - the universal anomaly after time t (km^0.5)
    fDot - time derivative of the Lagrange f coefficient (1/s)
    gDot - time derivative of the Lagrange g coefficient
           (dimensionless)
    
    User M-functions required: stumpC, stumpS
    
    Returns:
    fDot, gDot - Time derivatives of Lagrange coefficients
    """
    
    z = a * x**2
    
    # Equation 3.66c:
    fDot = math.sqrt(mu) / r / ro * (z * stumpS(z) - 1) * x
    
    # Equation 3.66d:
    gDot = 1 - x**2 / r * stumpC(z)
    
    return fDot, gDot