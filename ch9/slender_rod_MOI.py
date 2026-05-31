"""
Moment of inertia tensor of a uniform slender rod about its center of mass,
given symbolic parameterizations x(s), y(s), z(s) where s is arclength.

Uses sympy so the parameterization may contain unknown constants
(length, radius, angles, etc.).
"""

import sympy as sp


def _center_of_mass(x, y, z, s, bounds, L):
    s0, s1 = bounds
    x_cm = sp.integrate(x, (s, s0, s1)) / L
    y_cm = sp.integrate(y, (s, s0, s1)) / L
    z_cm = sp.integrate(z, (s, s0, s1)) / L
    return x_cm, y_cm, z_cm


def point_mass_MOI(x_cm, y_cm, z_cm, m):
    """Inertia matrix of a point mass m located at (x_cm, y_cm, z_cm)
    about the origin. Add this to a COM-frame inertia matrix to apply
    the parallel-axis theorem (shift from COM to the origin).
    """
    return sp.Matrix([
        [ m * (y_cm**2 + z_cm**2), -m * x_cm * y_cm,         -m * x_cm * z_cm        ],
        [-m * x_cm * y_cm,          m * (x_cm**2 + z_cm**2), -m * y_cm * z_cm        ],
        [-m * x_cm * z_cm,         -m * y_cm * z_cm,          m * (x_cm**2 + y_cm**2)],
    ])


def get_distance_vector(start, end):
    """Component-wise displacement end - start for two 3-element vectors
    (tuples/lists of sympy expressions).
    """
    return [e - s for s, e in zip(start, end)]


def slender_rod_COM(x, y, z, s, bounds):
    """COM coordinates (x_cm, y_cm, z_cm) of the rod parameterized by
    arclength s over the given bounds, in the input frame.
    """
    s0, s1 = bounds
    L = s1 - s0
    return _center_of_mass(x, y, z, s, bounds, L)


def slender_rod_MOI_COM(x, y, z, s, bounds, m):
    """Inertia matrix of a uniform slender rod about its COM, expressed in
    axes parallel to the frame in which x(s), y(s), z(s) are written.

    Assumes s is arclength, so L = bounds[1] - bounds[0] and ds is a length
    element directly (matches the textbook formulation).
    """
    s0, s1 = bounds
    L = s1 - s0
    rho = m / L

    x_cm, y_cm, z_cm = _center_of_mass(x, y, z, s, bounds, L)
    xs = x - x_cm
    ys = y - y_cm
    zs = z - z_cm

    def I(expr):
        return sp.integrate(expr * rho, (s, s0, s1))

    Ixx =  I(ys**2 + zs**2)
    Iyy =  I(xs**2 + zs**2)
    Izz =  I(xs**2 + ys**2)
    Ixy = -I(xs * ys)
    Ixz = -I(xs * zs)
    Iyz = -I(ys * zs)

    return sp.simplify(sp.Matrix([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz],
    ]))


if __name__ == "__main__":
    s = sp.symbols('s', real=True)
    m, L, Theta, d, omega = sp.symbols('m L Theta d omega', positive=True)

    x1 = (s - 2*L/3)
    y1 = 0
    z1 = 0

    x2 = (s - 2*L/3) * sp.cos(Theta)
    y2 = -(s - 2*L/3) * sp.sin(Theta)
    z2 = 0

    com1 = slender_rod_COM(x1, y1, z1, s, (0, L))
    com2 = slender_rod_COM(x2, y2, z2, s, (0, L))
    com_tot = [(a + b) / 2 for a, b in zip(com1, com2)]


    I_C1 = slender_rod_MOI_COM(x1, y1, z1, s, (0, L), m)
    [dx, dy, dz] = get_distance_vector(com1, com_tot)
    I_mp1 = point_mass_MOI(dx, dy, dz, m)
    I_P1 = I_C1 + I_mp1
    print("I_P1:")
    sp.pprint(I_P1)

    I_C2 = slender_rod_MOI_COM(x2, y2, z2, s, (0, L), m)
    [dx, dy, dz] = get_distance_vector(com2, com_tot)
    I_mp2 = point_mass_MOI(dx, dy, dz, m)
    I_P2 = I_C2 + I_mp2
    print("I_P2:")
    sp.pprint(I_P2)


    I_tot = I_P1 + I_P2
    print("I_tot:")
    sp.pprint(I_tot)

    # Shaft spins at constant rate about x. In the body-fixed frame I_tot is
    # constant and H_rel-dot = 0, so M = dH/dt = omega x H = omega x (I omega).
    omega_vec = sp.Matrix([omega, 0, 0])

    H = I_tot * omega_vec
    print("H = I_tot @ omega:")
    sp.pprint(H)

    M = omega_vec.cross(H)
    print("M = omega x H:")
    sp.pprint(sp.simplify(M))

    # Bearing positions on the x-axis (relative to C / the param-frame origin),
    # and their position vectors from the system COM G.
    G = sp.Matrix(com_tot)
    p_a = sp.Matrix([-2 * L / 3, 0, 0])
    p_b = sp.Matrix([L / 3, 0, 0])
    r_a = p_a - G
    r_b = p_b - G

    # G is off the spin axis, so it has a centripetal acceleration that the
    # bearings must supply as a net force. The full rigid-body acceleration of a
    # body-fixed point G is
    #     a_G = a_O + alpha x r_G + omega x (omega x r_G),
    # where O is the origin C. Two terms drop out here:
    #   - a_O = 0: C lies on the x-axis (the fixed spin axis held by the
    #     bearings), so the origin does not translate.
    #   - alpha x r_G = 0: the shaft spins at constant omega, so alpha = 0.
    # (The relative-motion and Coriolis terms are also zero since G is fixed in
    # the body.) What remains is the pure centripetal term:
    a_G = omega_vec.cross(omega_vec.cross(G))
    m_tot = 2 * m

    # Unknown bearing reactions; the problem says they are normal to AB (no x).
    FAy, FAz, FBy, FBz = sp.symbols('FAy FAz FBy FBz', real=True)
    F_a = sp.Matrix([0, FAy, FAz])
    F_b = sp.Matrix([0, FBy, FBz])

    # Newton-Euler about the COM:  sum F = m a_G,  sum M_G = omega x H = M.
    eqs = list(F_a + F_b - m_tot * a_G) + list(r_a.cross(F_a) + r_b.cross(F_b) - M)
    sol = sp.solve(eqs, [FAy, FAz, FBy, FBz], dict=True)[0]
    F_a = F_a.subs(sol)
    F_b = F_b.subs(sol)

    print("F_a:")
    sp.pprint(F_a.T)
    print("F_b:")
    sp.pprint(F_b.T)
    print("|F_a|:", sp.simplify(F_a.norm()))
    print("|F_b|:", sp.simplify(F_b.norm()))



