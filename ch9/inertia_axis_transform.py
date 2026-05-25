"""
Transform an inertia tensor (or any second-order tensor) between two
orthogonal coordinate systems sharing a common origin.

Given the direction-cosine matrix [Q] whose rows are the unit vectors of
the primed frame x'y'z' expressed in the unprimed frame xyz, the inertia
tensor transforms as

    [I'] = [Q] [I] [Q]^T          (Curtis Eq. 9.49a)

Equivalently, individual components are

    I_x'    = (row 1 of Q) [I] (row 1 of Q)^T          (Eq. 9.50)
    I_y'z'  = (row 2 of Q) [I] (row 3 of Q)^T
    ...
"""

import numpy as np


def transform_inertia_tensor(I, Q):
    # [I'] = [Q] [I] [Q]^T
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    return Q @ I @ Q.T


def transformed_component(I, Q, i, j):
    # Single component I'_{ij} = (row i of Q) [I] (row j of Q)^T
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    return Q[i] @ I @ Q[j]


def moment_of_inertia_about_axis(I, v):
    # Scalar moment of inertia about an axis through the origin
    # in the direction of vector v: I_v = u_hat . [I] . u_hat
    I = np.asarray(I, dtype=float)
    v = np.asarray(v, dtype=float)
    u_hat = v / np.linalg.norm(v)
    return float(u_hat @ I @ u_hat)


def rotation_matrix(axis, angle_rad):
    # Direction-cosine matrix for a rotation of the frame about 'x', 'y', or 'z'.
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    if axis == "y":
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    if axis == "z":
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")


if __name__ == "__main__":
    # Demo using the inertia tensor from Problem 9.5 (about the CoM).
    I = np.array([
        [792, 356, 36],
        [356, 792, -76],
        [36, -76, 792],
    ])

    # 30-degree rotation of the frame about the z-axis.
    Q = rotation_matrix("z", np.deg2rad(30))

    I_prime = transform_inertia_tensor(I, Q)
    print("Q =")
    print(Q)
    print("\n[I'] = Q I Q^T =")
    print(I_prime)

    # Verify a single component matches the full-matrix result.
    print(f"\nI'_xy from component formula: {transformed_component(I, Q, 0, 1):.6f}")
    print(f"I'_xy from full transform:    {I_prime[0, 1]:.6f}")

    v = [1, 2, 2]
    I_v = moment_of_inertia_about_axis(I, v)
    print(f"\nI about axis through {tuple(v)} m = {I_v:.4f} kg-m^2")
