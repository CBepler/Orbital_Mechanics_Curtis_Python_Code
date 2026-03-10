import numpy as np

MU = 398600


def bi_elliptic_transfer(r1, r2, r3):
    # r1, r2, r3 are in km
    # Calculates the time (s) and the delta-v (m/s) for a bi-elliptic transfer
    # orbit from a circular orbit to another circular orbit.
    # r1: radius of the initial circular orbit
    # r2: radius of the final circular orbit
    # r3: radius of the apogee of the first ellipse and the perigee of the second ellipse

    v1 = circular_velocity(r1)
    v2 = circular_velocity(r2)

    a1 = (r1 + r3) / 2
    a2 = (r3 + r2) / 2

    vt1p = ellipitcal_velocity(r1, a1)
    vt1a = ellipitcal_velocity(r3, a1)
    vt2p = ellipitcal_velocity(r2, a2)
    vt2a = ellipitcal_velocity(r3, a2)

    delta_v1 = np.abs(vt1p - v1)
    delta_v2 = np.abs(vt2a - vt1a)
    delta_v3 = np.abs(v2 - vt2p)

    delta_v = delta_v1 + delta_v2 + delta_v3

    T1 = period_ellipse(a1)
    T2 = period_ellipse(a2)

    T = (T1 + T2) / 2

    return delta_v, T


def ellipitcal_velocity(r, a):
    # r is the distance from the central body
    # a is the semi-major axis of the ellipse
    return np.sqrt(2 * MU * (1 / r - 1 / (2 * a)))


def circular_velocity(r):
    # r is the distance from the central body
    return np.sqrt(MU / r)


def period_ellipse(a):
    # a is the semi-major axis of the ellipse
    return (2 * np.pi / np.sqrt(MU)) * (a ** (3 / 2))


if __name__ == "__main__":
    r1 = 6671
    r2 = 9371
    r3 = 12389
    delta_v, T = bi_elliptic_transfer(r1, r2, r3)
    print(f"Delta v: {delta_v:.2f} km/s")
    print(f"Transfer time: {T:.2f} s")
    print(f"Transfer time: {T / 3600:.2f} h")
    print(f"Transfer time: {T / 86400:.2f} d")
