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

y_val = y(z)

# ============================================================
# Lagrange Coefficients (Curtis Eqs. 5.46, 5.47, 5.48)
# ============================================================
# Eq. 5.46: f = 1 - y/r_1
f_coef = 1 - y_val/r_0

# Eq. 5.47: g = A * sqrt(y/mu)
g_coef = A * np.sqrt(y_val/MU)

# Eq. 5.48: g_dot = 1 - y/r_2
gdot = 1 - y_val/r_1

# ============================================================
# Set up position vectors in the orbital plane
# Place r1 along x-axis, r2 at angle delta_theta (prograde)
# ============================================================
r1_vec = np.array([r_0, 0, 0])
r2_vec = np.array([r_1 * np.cos(delta_theta), r_1 * np.sin(delta_theta), 0])

# ============================================================
# Velocity at position 1 (Curtis Eq. 5.28)
# v1 = (1/g) * (r2 - f*r1)
# ============================================================
v1_vec = (1/g_coef) * (r2_vec - f_coef * r1_vec)

# ============================================================
# Orbital Elements from State Vector (Curtis Chapter 4)
# ============================================================

# Eq. 4.2: Specific angular momentum h = r x v
h_vec = np.cross(r1_vec, v1_vec)
h = np.linalg.norm(h_vec)

# Velocity magnitude
v1 = np.linalg.norm(v1_vec)

# Eq. 4.10: e = (1/mu)*[(v^2 - mu/r)*r - r*v_r*v]
# where v_r = (r.v)/r is the radial velocity, so r*v_r = r.v
v_r = np.dot(r1_vec, v1_vec) / r_0  # radial velocity component
e_vec = (1/MU) * ((v1**2 - MU/r_0) * r1_vec - r_0 * v_r * v1_vec)
e = np.linalg.norm(e_vec)

# Eq. 2.71: Perigee radius r_p = h^2 / (mu * (1 + e))
r_perigee = h**2 / (MU * (1 + e))

# Eq. 2.73: Semi-major axis a = h^2 / (mu * (1 - e^2))
a = h**2 / (MU * (1 - e**2))

# Perigee altitude
z_perigee = r_perigee - 6378

# ============================================================
# Results
# ============================================================
print(f"Universal variable z = {z:.6f}")
print(f"Angular momentum h = {h:.2f} km^2/s")
print(f"Eccentricity e = {e:.6f}")
print(f"Semi-major axis a = {a:.2f} km")
print(f"Perigee radius r_p = {r_perigee:.2f} km")
print(f"Perigee altitude z_p = {z_perigee:.2f} km")