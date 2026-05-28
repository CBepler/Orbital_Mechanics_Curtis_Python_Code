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
    m, L, Theta, d = sp.symbols('m L Theta d', positive=True)

    # Straight rod along the x-axis, s from 0 to L.
    # Expected: diag(0, m L^2 / 12, m L^2 / 12) about the COM.
    I_rod = slender_rod_MOI_COM(s, 0, 0, s, (0, L), m)
    print("Straight rod along x-axis, MOI about COM:")
    sp.pprint(I_rod)
    print()

    # Propeller (Problem 9.9): uniform rod of mass m, length L, tilted by
    # Theta in the yz-plane, whose center C is offset from P by d along x.
    x = d
    y = (s - L/2) * sp.cos(Theta)
    z = (s - L/2) * sp.sin(Theta)
    I_C = slender_rod_MOI_COM(x, y, z, s, (0, L), m)
    print("Propeller MOI about its COM (C):")
    sp.pprint(I_C)
    print()

    x_cm, y_cm, z_cm = slender_rod_COM(x, y, z, s, (0, L))
    I_mp = point_mass_MOI(x_cm, y_cm, z_cm, m)
    print("I_mp:")
    sp.pprint(I_mp)

    # Parallel-axis shift from C to P: add the point-mass MOI of the COM.
    I_P = sp.simplify(I_C + I_mp)
    print("Propeller MOI about P (parallel-axis shift):")
    sp.pprint(I_P)
