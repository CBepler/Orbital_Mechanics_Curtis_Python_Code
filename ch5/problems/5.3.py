import numpy as np
import sys
import os

# Add parent directories to path to import local modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ch3'))

from ch3.stumpC import stumpC as c2 
from ch3.stumpS import stumpS as c3 

MU = 398600

z_0 = 600
z_1 = 300
r_0 = z_0 + 6378
r_1 = z_1 + 6378
t_0 = 0
t_1 = 15 * 60
delta_theta = 60 * np.pi / 180

delta_t = t_1 - t_0

A = np.sin(delta_theta) * np.sqrt((r_0 * r_1) / (1 - np.cos(delta_theta)))

y = lambda z: r_0 + r_1 + A * ((z * c3(z) - 1) / np.sqrt(c2(z)))
f = lambda z: (y(z) / c2(z))**(3/2) * c3(z) + A * np.sqrt(y(z)) - np.sqrt(MU) * delta_t

def f_prime(z):
    if z != 0:
        return (y(z) / c2(z)) ** (3/2) * ((1/(2*z)) * (c2(z) - 3/2 * (c3(z) / c2(z))) + 3/4 * (c3(z) ** 2 / c2(z))) + (A/8) * (3 * c3(z) / c2(z) * np.sqrt(y(z)) + A * np.sqrt(c2(z) / y(z)))
    else:
        return np.sqrt(2) / 40 * y(0) ** (3/2) + A/8 * (np.sqrt(y(0)) + A * np.sqrt(1 / (2 * y(0))))

z = 0
while abs(f(z) / f_prime(z)) > 1e-7:
    z = z - f(z) / f_prime(z)

y = y(z)

# Calculate f and g
f = 1 - y/r_0
g = A * np.sqrt(y/MU)
gdot = 1 - y/r_1

# Calculate the velocity vectors
v1 = (1 / g) * (r_1 - f * r_0)
v2 = (1 / g) * (gdot * r_1 - r_0)

print(z)