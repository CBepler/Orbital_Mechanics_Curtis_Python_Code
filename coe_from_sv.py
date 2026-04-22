import numpy as np

def coe_from_sv(R, V, mu=398600.4418):
    """
    This function computes the classical orbital elements (coe)
    from the state vector (R,V) using Algorithm 4.1.

    Args:
        R (np.array): Position vector in the geocentric equatorial frame (km)
        V (np.array): Velocity vector in the geocentric equatorial frame (km/s)

    Returns:
        list: Orbital elements [h, e, RA, incl, w, TA, a]
              h - the magnitude of the angular momentum vector (km^2/s)
              e - eccentricity (magnitude of E)
              RA - right ascension of the ascending node (rad)
              incl - inclination of the orbit (rad)
              w - argument of perigee (rad)
              TA - true anomaly (rad)
              a - semimajor axis (km)
    """

    eps = 1.0e-10  # a small number below which the eccentricity is considered to be zero
    pi = np.pi  # 3.1415926...

    r = np.linalg.norm(R)
    v = np.linalg.norm(V)

    vr = np.dot(R, V) / r

    H = np.cross(R, V)
    h = np.linalg.norm(H)

    # Equation 4.7:
    incl = np.arccos(H[2] / h)  # H(3) in MATLAB is H[2] in Python (0-indexed)

    # Equation 4.8:
    N = np.cross(np.array([0, 0, 1]), H) #Node line vector
    n = np.linalg.norm(N)

    # Equation 4.9:
    if n != 0:
        RA = np.arccos(N[0] / n)  # N(1) in MATLAB is N[0] in Python
        if N[1] < 0:  # N(2) in MATLAB is N[1] in Python
            RA = 2 * pi - RA
    else:
        RA = 0

    # Equation 4.10:
    # E = 1/mu * ((v^2 - mu/r)*R - r*vr*V)  -- Original MATLAB comment
    E = (1 / mu) * ((v**2 - mu / r) * R - r * vr * V)
    e = np.linalg.norm(E)

    # Equation 4.12 (incorporating the case e = 0):
    if e > eps:
        w = np.arccos(np.dot(N, E) / (n * e))
        if E[2] < 0:  # E(3) in MATLAB is E[2] in Python
            w = 2 * pi - w
    else:
        w = 0

    # Equation 4.13a (incorporating the case e = 0):
    if e > eps:
        TA = np.arccos(np.dot(E, R) / (e * r))
        if vr < 0:
            TA = 2 * pi - TA
    else:
        cp = np.cross(N, R)
        if cp[2] >= 0:  # cp(3) in MATLAB is cp[2] in Python
            TA = np.arccos(np.dot(N, R) / (n * r))
        else:
            TA = 2 * pi - np.arccos(np.dot(N, R) / (n * r))

    # Equation 2.61 (a < 0 for a hyperbola):
    a = h**2 / (mu * (1 - e**2))

    coe = [h, e, RA, incl, w, TA, a]

    return coe

def rads_to_degrees(radians):
    """
    Convert radians to degrees.
    """
    return radians * (180.0 / np.pi)

def print_coe(coe):
    """
    Print the classical orbital elements in a formatted way.
    """
    print(f"Angular momentum (h): {coe[0]:.4f} km^2/s")
    print(f"Eccentricity (e): {coe[1]:.4f}")
    print(f"Right Ascension of Ascending Node (RA): {coe[2]:.4f} rads ({rads_to_degrees(coe[2]):.4f} degrees)")
    print(f"Inclination (incl): {coe[3]:.4f} rads ({rads_to_degrees(coe[3]):.4f} degrees)") 
    print(f"Argument of Perigee (w): {coe[4]:.4f} rads ({rads_to_degrees(coe[4]):.4f} degrees)")
    print(f"True Anomaly (TA): {coe[5]:.4f} rads ({rads_to_degrees(coe[5]):.4f} degrees)")
    print(f"Semi-major axis (a): {coe[6]:.4f} km")
    if coe[1] < 1:
        mu = 398600.4418 
        T = 2 * np.pi / np.sqrt(mu) * (coe[6]**1.5)
        print(f"\nPeriod:")
        print(f"Seconds = {T:.4f}")
        print(f"Minutes = {T/60:.4f}")
        print(f"Hours = {T/3600:.4f}")
        print(f"Days = {T/86400:.4f}")

if __name__ == '__main__':
    R2 = np.array([0, 778.6 * (10 ** 6), 0])
    V2 = np.array([-17.304, 3.7148, 0])
    coe2 = coe_from_sv(R2, V2, mu=132712000000)
    print_coe(coe2)