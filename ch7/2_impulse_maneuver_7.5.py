import numpy as np
from helpers import *

MU = 398600

def delta_v(r_a, delta_r_o, delta_v_o_minus, t):
    # Two-impulse rendezvous via the Clohessy-Wiltshire equations.
    # Computes the chaser's required impulses to drive its relative
    # position to zero (rendezvous with the target) over time t.
    #
    # Inputs:
    #   r_a             - radius of the target's circular orbit (km)
    #   delta_r_o       - initial relative position of chaser in
    #                     target LVLH frame [x, y, z] (km)
    #   delta_v_o_minus - chaser's relative velocity just before the
    #                     first impulse [x, y, z] (km/s)
    #   t               - transfer time from first to second impulse (s)
    #
    # Returns:
    #   (delta_v_0, delta_v_f) - impulse vectors at start and end (km/s)
    #     delta_v_0 = (v_0+) - (v_0-)  applied at t = 0
    #     delta_v_f = (v_f+) - (v_f-)  applied at t = t, with v_f+ = 0
    T = ((2 * np.pi) / np.sqrt(MU)) * (r_a ** (3/2))
    n = (2 * np.pi) / T
    rr = get_rr(n, t)
    rv = get_rv(n, t)
    vr = get_vr(n, t)
    vv = get_vv(n, t)
    delta_v_o_plus = -np.linalg.inv(rv) @ rr @ delta_r_o
    delta_v_f_minus = (vr - vv @ np.linalg.inv(rv) @ rr) @ delta_r_o
    return (delta_v_o_plus - delta_v_o_minus, -delta_v_f_minus)

if __name__ == "__main__":
    dv0, dvf = delta_v(6600, np.array([1, 1, 1]), np.array([0, 0, 0.005]), 1778.7)
    mag0 = np.linalg.norm(dv0)
    magf = np.linalg.norm(dvf)
    print(f"|dv0| = {mag0:.6f} km/s")
    print(f"|dvf| = {magf:.6f} km/s")
    print(f"total = {mag0 + magf:.6f} km/s")