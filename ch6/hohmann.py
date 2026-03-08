import numpy as np

MU = 132712000000


def hohmann_transfer(r1, r2):
    # Calculates the time (s) and the delta-v (m/s) for a hohmann transfer
    # orbit from a circular orbit to another circular orbit.
    v1 = np.sqrt(MU / r1)
    v2 = np.sqrt(MU / r2)

    # transfer orbit apoapsis and perapsis radii
    rta = r1 if r1 > r2 else r2
    rtp = r2 if r1 > r2 else r1
    et = (rta - rtp) / (rta + rtp)
    ht = np.sqrt(rta * MU * (1 - et))

    v1t = ht / r1
    delta_v1 = np.abs(v1t - v1)

    v2t = ht / r2
    delta_v2 = np.abs(v2t - v2)

    delta_v = delta_v1 + delta_v2

    at = (rta + rtp) / 2
    T = ((2 * np.pi) / np.sqrt(MU)) * (at ** (3 / 2))

    return delta_v, T / 2


if __name__ == "__main__":
    r1 = 149600000
    r2 = 227900000
    delta_v, T = hohmann_transfer(r1, r2)
    print(f"Delta v: {delta_v:.2f} km/s")
    print(f"Transfer time: {T:.2f} s")
    print(f"Transfer time: {T / 3600:.2f} h")
    print(f"Transfer time: {T / 86400:.2f} d")
