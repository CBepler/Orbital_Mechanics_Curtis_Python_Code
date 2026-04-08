import numpy as np

MU = 398600


def circular_relative(r_target, r_chaser, delta_theta):
    # Computes relative position, velocity, and acceleration of chaser
    # w.r.t. target in the target's LVLH frame.
    # Both spacecraft are in circular Earth orbits.
    #
    # Inputs:
    #   r_target    - radius of target circular orbit (km)
    #   r_chaser    - radius of chaser circular orbit (km)
    #   delta_theta - true anomaly of chaser minus target (rad)
    #
    # Outputs:
    #   r_rel - relative position  [x, y, z] (km)
    #   v_rel - relative velocity  [x, y, z] (km/s)
    #   a_rel - relative acceleration [x, y, z] (km/s^2)
    #
    # LVLH frame: +x radial out, +y along-track, +z orbit normal

    n_T = np.sqrt(MU / r_target**3)
    n_C = np.sqrt(MU / r_chaser**3)

    # Absolute quantities in LVLH frame (target at theta_T = 0)
    R_T = np.array([r_target, 0, 0])
    R_C = np.array([r_chaser * np.cos(delta_theta),
                     r_chaser * np.sin(delta_theta), 0])

    V_T = np.array([0, n_T * r_target, 0])
    V_C = np.array([-n_C * r_chaser * np.sin(delta_theta),
                      n_C * r_chaser * np.cos(delta_theta), 0])

    a_T = np.array([-n_T**2 * r_target, 0, 0])
    a_C = np.array([-n_C**2 * r_chaser * np.cos(delta_theta),
                     -n_C**2 * r_chaser * np.sin(delta_theta), 0])

    Omega = np.array([0, 0, n_T])

    # Equation 1.38: v = v_O + Omega x r_rel + v_rel
    r_rel = R_C - R_T
    v_rel = V_C - V_T - np.cross(Omega, r_rel)

    # Equation 1.42: a = a_O + Omega_dot x r_rel + Omega x (Omega x r_rel)
    #                    + 2*Omega x v_rel + a_rel
    # (Omega_dot = 0 for circular orbit)
    a_rel = (a_C - a_T - np.cross(Omega, np.cross(Omega, r_rel))
             - 2 * np.cross(Omega, v_rel))

    return r_rel, v_rel, a_rel


if __name__ == "__main__":
    r_rel, v_rel, a_rel = circular_relative(8000, 7000, 0)
    print("Same orbit, same position:")
    print(f"  r_rel = {r_rel}")
    print(f"  v_rel = {v_rel}")
    print(f"  a_rel = {a_rel}")
