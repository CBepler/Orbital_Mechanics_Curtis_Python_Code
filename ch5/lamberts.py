import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ch3'))

from ch3.stumpC import stumpC as c2 
from ch3.stumpS import stumpS as c3 


MU = 398600

def lambert(r1, r2, delta_t, mu=MU):
    # Lambert's method assuming a prograde orbit

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    # Calculate the change in true anomaly
    quad_determiner = np.cross(r1, r2)[2]
    if quad_determiner >= 0:
        delta_theta = np.arccos(np.dot(r1, r2) / (r1_mag * r2_mag))
    else:
        delta_theta = 2 * np.pi - np.arccos(np.dot(r1, r2) / (r1_mag * r2_mag))

    # Calculate A
    A = np.sin(delta_theta) * np.sqrt((r1_mag * r2_mag) / (1 - np.cos(delta_theta)))

    # Iteratively solve for z using Newtons method
    z = 0

    y = lambda z: r1_mag + r2_mag + A * ((z * c3(z) - 1) / np.sqrt(c2(z)))
    f = lambda z: (y(z) / c2(z))**(3/2) * c3(z) + A * np.sqrt(y(z)) - np.sqrt(mu) * delta_t

    def f_prime(z):
        if z != 0:
            return (y(z) / c2(z)) ** (3/2) * ((1/(2*z)) * (c2(z) - 3/2 * (c3(z) / c2(z))) + 3/4 * (c3(z) ** 2 / c2(z))) + (A/8) * (3 * c3(z) / c2(z) * np.sqrt(y(z)) + A * np.sqrt(c2(z) / y(z)))
        else:
            return np.sqrt(2) / 40 * y(0) ** (3/2) + A/8 * (np.sqrt(y(0)) + A * np.sqrt(1 / (2 * y(0))))

    z = 0
    while abs(f(z) / f_prime(z)) > 1e-7:
        z = z - f(z) / f_prime(z)

    y_val = y(z)

    # Calculate f and g
    f_coef = 1 - y_val/r1_mag
    g_coef = A * np.sqrt(y_val/mu)
    gdot = 1 - y_val/r2_mag

    # Calculate the velocity vectors
    v1 = (1 / g_coef) * (r2 - f_coef * r1)
    v2 = (1 / g_coef) * (gdot * r2 - r1)
    
    return v1, v2


if __name__ == "__main__":
    r1 = np.array([-3600, 3600, 5100])
    r2 = np.array([-5500, -6240, -520])
    delta_t = 1800
    print(lambert(r1, r2, delta_t))
    