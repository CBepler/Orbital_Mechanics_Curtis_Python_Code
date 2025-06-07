import numpy as np

def sv_from_coe(coe):
    """
    This function computes the state vector (r,v) from the
    classical orbital elements (coe).

    Args:
        coe (list): Orbital elements [h, e, RA, incl, w, TA]
              h - angular momentum (km^2/s)
              e - eccentricity
              RA - right ascension of the ascending node (rad)
              incl - inclination of the orbit (rad)
              w - argument of perigee (rad)
              TA - true anomaly (rad)

    Returns:
        tuple: (r, v)
              r - position vector in the geocentric equatorial frame (km)
              v - velocity vector in the geocentric equatorial frame (km/s)
    """

    # global mu  # In Python, we pass constants or define them at a module level
    mu = 398600.4418  # gravitational parameter (km^3/s^2)

    h = coe[0]
    e = coe[1]
    RA = coe[2]
    incl = coe[3]
    w = coe[4]
    TA = coe[5]

    # Equations 4.37 and 4.38 (rp and vp are column vectors):
    # MATLAB: rp = (h^2/mu) * (1/(1 + e*cos(TA))) * [cos(TA); sin(TA); 0]
    # MATLAB: vp = (mu/h) * (-sin(TA); (e + cos(TA)); 0)
    
    # In Python, we'll construct these as 1D arrays or row vectors and then
    # ensure consistent multiplication.
    
    # rp calculation
    factor_rp = (h**2 / mu) * (1 / (1 + e * np.cos(TA)))
    rp = factor_rp * np.array([np.cos(TA), np.sin(TA), 0])

    # vp calculation
    factor_vp = (mu / h)
    vp = factor_vp * np.array([-np.sin(TA), (e + np.cos(TA)), 0])

    # Equation 4.39: R3_W - Rotation matrix about the z-axis through the angle RA
    R3_W = np.array([
        [np.cos(RA), np.sin(RA), 0],
        [-np.sin(RA), np.cos(RA), 0],
        [0, 0, 1]
    ])

    # Equation 4.40: R1_i - Rotation matrix about the x-axis through the angle incl
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(incl), np.sin(incl)],
        [0, -np.sin(incl), np.cos(incl)]
    ])

    # Equation 4.41: R3_w - Rotation matrix about the z-axis through the angle w
    R3_w = np.array([
        [np.cos(w), np.sin(w), 0],
        [-np.sin(w), np.cos(w), 0],
        [0, 0, 1]
    ])

    # Equation 4.44: Q_PX - Matrix of the transformation from perifocal to
    # geocentric equatorial frame. In MATLAB, matrix multiplication is `*`
    # For NumPy, use `@` for matrix multiplication or `np.dot()`.
    Q_PX = R3_W.T @ R1_i @ R3_w.T

    # Equations 4.46 (r and v are column vectors):
    # In MATLAB, `Q_PX*rp` and `Q_PX*vp` perform matrix-vector multiplication.
    # In NumPy, direct multiplication `Q_PX @ rp` or `np.dot(Q_PX, rp)` works.
    r = Q_PX @ rp
    v = Q_PX @ vp

    # Convert r and v into row vectors:
    # MATLAB: r = r' and v = v' means transposing.
    # In NumPy, rp and vp were already created as 1D arrays, which are
    # effectively row vectors for these operations. If they were 2D column
    # vectors, transposing would be needed. Since the result of Q_PX @ rp
    # will be a 1D array, no explicit transpose is needed for row vector representation
    # unless a specific 2D (1x3) shape is required. We'll return them as 1D arrays.

    r[2] = -r[2]
    v[2] = -v[2]

    return r, v

if __name__ == '__main__':
    # Example usage (replace with your actual coe list)
    # This is just a placeholder example for demonstration.
    # You'll need to define 'coe' based on your specific problem.

    # Example 1: Circular orbit (coe from a common orbital mechanics textbook example)
    # h = 53344.0, e = 0, RA = 0, incl = 0, w = 0, TA = 0
    coe1 = [53344.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    r1, v1 = sv_from_coe(coe1)
    print(f"Example 1 Position Vector (r): {r1}")
    print(f"Example 1 Velocity Vector (v): {v1}")
    print("-" * 30)

    # Example 2: Elliptical orbit (coe from another common example)
    # This example assumes arbitrary valid orbital elements for demonstration.
    # h = 80000, e = 1.4, RA = np.deg2rad(40), incl = np.deg2rad(30), w = np.deg2rad(60), TA = np.deg2rad(30)
    coe2 = [80000.0, 1.4, np.deg2rad(40), np.deg2rad(30), np.deg2rad(60), np.deg2rad(30)]
    r2, v2 = sv_from_coe(coe2)
    print(f"Example 2 Position Vector (r): {r2}")
    print(f"Example 2 Velocity Vector (v): {v2}")
    print("-" * 30)

    # Note: The output values will be in the units specified in the comments
    # (distances in km, velocities in km/s).