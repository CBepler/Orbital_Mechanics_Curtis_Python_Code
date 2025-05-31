import math
import numpy as np

from kepler_U import kepler_U
from f_and_g import f_and_g
from fDot_and_gDot import fDot_and_gDot

def rv_from_r0v0(R0, V0, t, mu):
    """
    This function computes the state vector (R,V) from the
    initial state vector (R0,V0) and the elapsed time.
    
    Parameters:
    mu - gravitational parameter (km^3/s^2)
    R0 - initial position vector (km)
    V0 - initial velocity vector (km/s)
    t  - elapsed time (s)
    R  - final position vector (km)
    V  - final velocity vector (km/s)
    
    User M-functions required: kepler_U, f_and_g, fDot_and_gDot
    
    Returns:
    R, V - Final position and velocity vectors
    """
    
    # Convert to numpy arrays for vector operations
    R0 = np.array(R0)
    V0 = np.array(V0)
    
    # Magnitudes of R0 and V0:
    r0 = np.linalg.norm(R0)
    v0 = np.linalg.norm(V0)
    
    # Initial radial velocity:
    vr0 = np.dot(R0, V0) / r0
    
    # Reciprocal of the semimajor axis (from the energy equation):
    alpha = 2/r0 - v0**2/mu
    
    # Compute the universal anomaly:
    x, n = kepler_U(t, r0, vr0, alpha, mu)
    
    # Compute the f and g functions:
    f, g = f_and_g(x, t, r0, alpha, mu)
    
    # Compute the final position vector:
    R = f * R0 + g * V0
    
    # Compute the magnitude of R:
    r = np.linalg.norm(R)
    
    # Compute the derivatives of f and g:
    fDot, gDot = fDot_and_gDot(x, r, r0, alpha, mu)
    
    # Compute the final velocity:
    V = fDot * R0 + gDot * V0
    
    return R, V

if __name__ == "__main__":
    mu = 398600
    R0 = [7000, -12124, 0] 
    V0 = [2.6679, 4.6210, 0]
    t = 3600
    R, V = rv_from_r0v0(R0, V0, t, mu)
    print(f"Final Position R: {R}")
    print(f"Final Velocity V: {V}")