import numpy as np

MU = 398600


def gauss(direction_cosines, observor_positions, times):
    assert len(direction_cosines) == len(observor_positions) == len(times) == 3
    # Calculate time intervals
    tau_1 = times[0] - times[1]
    tau_3 = times[2] - times[1]
    tau = tau_3 - tau_1
    # Calculate the direction cross products
    p1 = np.cross(direction_cosines[1], direction_cosines[2])
    p2 = np.cross(direction_cosines[0], direction_cosines[2])
    p3 = np.cross(direction_cosines[0], direction_cosines[1])
    # Calculate D0
    D0 = np.dot(direction_cosines[0], p1)
    # Computes Ds
    D11 = np.dot(observor_positions[0], p1)
    D12 = np.dot(observor_positions[0], p2)
    D13 = np.dot(observor_positions[0], p3)
    D21 = np.dot(observor_positions[1], p1)
    D22 = np.dot(observor_positions[1], p2)
    D23 = np.dot(observor_positions[1], p3)
    D31 = np.dot(observor_positions[2], p1)
    D32 = np.dot(observor_positions[2], p2)
    D33 = np.dot(observor_positions[2], p3)
    # Calculate A and B
    A = (1 / D0) * (-D12 * (tau_3 / tau) + D22 + D32 * (tau_1 / tau))
    B = (1 / (6 * D0)) * (
        D12 * (tau_3**2 - tau**2) * (tau_3 / tau)
        + D32 * (tau**2 - tau_1**2) * (tau_1 / tau)
    )
    # Calculate E
    E = np.dot(observor_positions[1], direction_cosines[1])
    R_2_squared = np.dot(observor_positions[1], observor_positions[1])
    # Calulate a,b,c
    a = -(A**2 + 2 * A * E + R_2_squared)
    b = -2 * MU * B * (A + E)
    c = -(MU**2) * B**2
    # Find the roots of x**8 + a * x**6 + b * x**3 + c = 0
    roots = np.roots([1, 0, a, 0, 0, b, 0, 0, c])
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]
    # Calculate ranges
    rho_2 = A + (MU * B / (roots**3))
    rho_1 = (1 / D0) * (
        (
            6 * (D31 * (tau_1 / tau_3) + D21 * (tau / tau_3))(roots**3)
            + MU * D31 * (tau**2 - tau_1**2) * (tau_1 / tau_3)
        )
        / (6 * roots**3 + MU * (tau**2 - tau_3**2))
        - D11
    )
    rho_3 = (1 / D0) * (
        (
            6 * (D13 * (tau_3 / tau_1) - D23 * (tau / tau_1))(roots**3)
            + MU * D13 * (tau**2 - tau_3**2) * (tau_3 / tau_1)
        )
        / (6 * roots**3 + MU * (tau**2 - tau_3**2))
        - D33
    )
    # Calculate position vectors
    r1 = observor_positions[0] + rho_1 * direction_cosines[0]
    r2 = observor_positions[1] + rho_2 * direction_cosines[1]
    r3 = observor_positions[2] + rho_3 * direction_cosines[2]
    # Calculate the lagrange coefficients
    f1 = 1 - (1 / 2) * (MU / roots**3) * (tau_1**2)
    f3 = 1 - (1 / 2) * (MU / roots**3) * (tau_3**2)
    g1 = tau_1 - (1 / 6) * (MU / roots**3) * (tau_1**3)
    g3 = tau_3 - (1 / 6) * (MU / roots**3) * (tau_3**3)
    # Calculate v2
    v2 = (1 / (f1 * g3 - f3 * g1)) * (-f3 * r1 + f1 * r3)
    return r2, v2
