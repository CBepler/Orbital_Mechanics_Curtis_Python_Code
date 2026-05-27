"""
Principal moments of inertia and principal directions from the components
of a symmetric inertia tensor.
"""

import numpy as np


def principle_directions(I_x, I_y, I_z, I_xy, I_xz, I_yz):
    I = np.array([
        [ I_x, I_xy, I_xz],
        [I_xy,  I_y, I_yz],
        [I_xz, I_yz,  I_z],
    ])
    eigvals, eigvecs = np.linalg.eigh(I)
    return eigvals, eigvecs.T


if __name__ == "__main__":
    # Example 9.2 (Curtis)
    moments, directions = principle_directions(
        I_x=1666.7, I_y=3333.3, I_z=4333.3,
        I_xy=-1500, I_xz=-750, I_yz=-500,
    )
    for lam, v in zip(moments, directions):
        print(f"lambda = {lam:.4f}  direction = {v}")
