"""
Moment of inertia tensor for a point mass and for a collection of point masses
about the origin of the given coordinate system.

Each point is specified as [x, y, z, m].
"""

import numpy as np


def calculate_moment_of_inertia(point):
    # Moment of inertia tensor of a single point mass about the origin
    x, y, z, m = point
    return np.array([
        [ m * (y**2 + z**2), -m * x * y,        -m * x * z       ],
        [-m * x * y,          m * (x**2 + z**2), -m * y * z      ],
        [-m * x * z,         -m * y * z,         m * (x**2 + y**2)],
    ])

def calculate_COM(points):
    pts = np.asarray(points, dtype=float)
    m = pts[:, 3]
    total_mass = m.sum()
    x_com = (pts[:, 0] * m).sum() / total_mass
    y_com = (pts[:, 1] * m).sum() / total_mass
    z_com = (pts[:, 2] * m).sum() / total_mass
    return [x_com, y_com, z_com, total_mass]

def calculate_point_COM_MOI(points):
    return calculate_moment_of_inertia(calculate_COM(points))


def calculate_point_collection_MOI_origin(points):
    # Sum the inertia tensors of every point mass in the collection
    return np.sum([calculate_moment_of_inertia(p) for p in points], axis=0)

def calculate_point_collection_MOI_COM(points):
    MOI_origin = calculate_point_collection_MOI_origin(points)
    MOI_point_COM = calculate_point_COM_MOI(points)
    return MOI_origin - MOI_point_COM



if __name__ == "__main__":
    # Example 9.1 (Curtis): four point masses
    points = [
        [1, 1, 1, 10],
        [-1, -1 , -1, 10],
        [4, -4, 4, 8],
        [-2, 2, -2, 8],
        [3, -3, -3, 12],
        [-3, 3, 3, 12],
    ]
    I = calculate_point_collection_MOI_COM(points)
    print("Moment of inertia tensor about the center of mass (kg-m^2):")
    print(I)
