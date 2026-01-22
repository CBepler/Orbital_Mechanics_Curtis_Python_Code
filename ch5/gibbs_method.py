import numpy as np

MU = 398600

def gibbs_method(r1, r2, r3):
    # Calculate the magnitude of the position vectors
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    r3_mag = np.linalg.norm(r3)
    
    # Calculate C_12, C_23, C_31
    C_12 = np.cross(r1, r2)
    C_23 = np.cross(r2, r3)
    C_31 = np.cross(r3, r1)
    
    # # Verify that the position vectors are coplanar
    # if np.dot(C_23, r1 / r1_mag) != 0:
    #     raise ValueError("The position vectors are not coplanar")

    # Calculate N, D, S
    N = r1_mag * C_23 + r2_mag * C_31 + r3_mag * C_12
    D = C_12 + C_23 + C_31
    S = r1_mag * (r2 - r3) + r2_mag * (r3 - r1) + r3_mag * (r1 - r2)

    # Calculate the velocity vector
    v = np.sqrt(MU / (np.linalg.norm(N) * np.linalg.norm(D))) * (np.cross(D, r2) / r2_mag + S)
    return v


if __name__ == "__main__":
    r1 = np.array([5887, -3520, -1204])
    r2 = np.array([5572, -3457, -2376])
    r3 = np.array([5088, -3289, -3480])
    print(gibbs_method(r1, r2, r3))
    print(np.linalg.norm(gibbs_method(r1, r2, r3)))
    