import numpy as np

from ch5.gauss import gauss
from ch3.kepler_U import kepler_U
from ch3.stumpC import stumpC
from ch3.stumpS import stumpS

MU = 398600


def gauss_iterative(direction_cosines, observor_positions, times, max_iterations=10):
    assert len(direction_cosines) == len(observor_positions) == len(times) == 3
    # Use Gauss' method to find r2 and v2
    r2, v2 = gauss(direction_cosines, observor_positions, times)
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
    # Iterate to find r2 and v2
    for i in range(max_iterations):
        # Calculate magnitudes
        r2_mag = np.linalg.norm(r2)
        v2_mag = np.linalg.norm(v2)
        # Calculate alpha (reciprocal of semi-major axis)
        alpha = (2 / r2_mag) - (v2_mag**2 / MU)
        # Calculate the radial component of v2
        vr2_mag = np.dot(v2, r2) / r2_mag
        # Solve universal Kepler's equation for chi1 and chi3 at times t1 and t3
        tau1 = times[0] - times[1]
        tau3 = times[2] - times[1]
        chi1, _ = kepler_U(tau1, r2_mag, vr2_mag, 1 / alpha, MU)
        chi3, _ = kepler_U(tau3, r2_mag, vr2_mag, 1 / alpha, MU)
        # Calculate lagrange coefficients
        f1 = 1 - (chi1**2 / r2_mag) * stumpC(alpha * chi1**2)
        f3 = 1 - (chi3**2 / r2_mag) * stumpC(alpha * chi3**2)
        g1 = (times[0] - times[1]) - (1 / np.sqrt(MU)) * chi1**3 * stumpS(
            alpha * chi1**2
        )
        g3 = (times[2] - times[1]) - (1 / np.sqrt(MU)) * chi3**3 * stumpS(
            alpha * chi3**2
        )
        # Calculate c1 and c3
        c1 = g3 / (f1 * g3 - f3 * g1)
        c3 = -g1 / (f1 * g3 - f3 * g1)
        # Calculate ranges
        rho1 = (1 / D0) * (-D11 + (D21 / c1) - (c3 / c1) * D31)
        rho2 = (1 / D0) * (-c1 * D12 + D22 - c3 * D32)
        rho3 = (1 / D0) * (-(c1 / c3) * D13 + (D23 / c3) - D33)
        # Calculate position vectors
        r1 = observor_positions[0] + rho1 * direction_cosines[0]
        r2_new = observor_positions[1] + rho2 * direction_cosines[1]
        r3 = observor_positions[2] + rho3 * direction_cosines[2]

        # Damping to prevent divergence
        r2 = r2 + 0.5 * (r2_new - r2)

        # Calculate velocity vector
        v2 = (1 / (f1 * g3 - f3 * g1)) * (-f3 * r1 + f1 * r3)
    return r2, v2
