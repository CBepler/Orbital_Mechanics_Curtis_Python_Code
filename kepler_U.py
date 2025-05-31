import math

from stumpC import stumpC
from stumpS import stumpS

def kepler_U(dt, ro, vro, a, mu):
    """
    This function uses Newton's method to solve the universal
    Kepler equation for the universal anomaly.
    
    Parameters:
    mu   - gravitational parameter (km^3/s^2)
    x    - the universal anomaly (km^0.5)
    dt   - time since x = 0 (s)
    ro   - radial position (km) when x = 0
    vro  - radial velocity (km/s) when x = 0
    a    - reciprocal of the semimajor axis (1/km)
    z    - auxiliary variable (z = a*x^2)
    C    - value of Stumpff function C(z)
    S    - value of Stumpff function S(z)
    n    - number of iterations for convergence
    nMax - maximum allowable number of iterations
    
    User M-functions required: stumpC, stumpS
    
    Returns:
    x    - universal anomaly
    n    - number of iterations used
    """
    
    # Set an error tolerance and a limit on the number of iterations:
    error = 1.0e-8
    nMax = 1000
    
    # Starting value for x:
    x = math.sqrt(mu) * abs(a) * dt
    
    # Iterate on Equation 3.62 until convergence occurs within
    # the error tolerance:
    n = 0
    ratio = 1
    
    while abs(ratio) > error and n <= nMax:
        n = n + 1
        z = a * x**2
        C = stumpC(z)
        S = stumpS(z)
        F = ro * vro / math.sqrt(mu) * x**2 * C + (1 - a * ro) * x**3 * S + ro * x - math.sqrt(mu) * dt
        dFdx = ro * vro / math.sqrt(mu) * x * (1 - a * x**2 * S) + (1 - a * ro) * x**2 * C + ro
        
        ratio = F / dFdx
        x = x - ratio
    
    # Deliver a value for x, but report that nMax was reached:
    if n > nMax:
        print(f'\n **No. iterations of Kepler\'s equation')
        print(f' = {n}')
        print(f'\n   F/dFdx = {ratio}\n')
    
    return x, n

if __name__ == "__main__":
    mu = 398600
    dt = 3600 
    ro = 10000
    vro = 3.0752
    a = 1 / -19655
    
    x, n = kepler_U(dt, ro, vro, a, mu)
    print(f"Universal anomaly x: {x:.3f} km^0.5")
    print(f"Number of iterations used: {n}")