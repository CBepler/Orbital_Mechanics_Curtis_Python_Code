import numpy as np

def get_rr(n, t):
    return np.array([[4 - 3 * np.cos(n * t), 0, 0], [6 * (np.sin(n * t) - n * t), 1, 0], [0, 0, np.cos(n * t)]])

def get_rv(n, t):
    return np.array([[(1/n) * np.sin(n * t), (2/n) * (1 - np.cos(n * t)), 0], [(2/n) * (np.cos(n * t) - 1), (1/n) * (4 * np.sin(n * t) - 3 * n * t), 0], [0, 0, (1/n) * np.sin(n * t)]])

def get_vr(n, t):
    return np.array([[3 * n * np.sin(n * t), 0, 0], [6 * n *(np.cos(n * t) - 1), 0, 0], [0, 0, -n * np.sin(n * t)]])

def get_vv(n, t):
    return np.array([[np.cos(n * t), 2 * np.sin(n * t), 0], [-2 * np.sin(n * t), 4 * np.cos(n * t) - 3, 0], [0, 0, np.cos(n * t)]])

def get_v_o_plus(n, t, r_o):
    return -np.linalg.inv(get_rv(n, t)) @ get_rr(n, t) @ r_o

def get_v_f_minus(n, t, r_o):
    return (get_vr(n, t) - get_vv(n, t) @ np.linalg.inv(get_rv(n, t)) @ get_rr(n, t)) @ r_o

def get_v(n, t, r_o, v_o):
    return get_vr(n, t) @ r_o + get_vv(n, t) @ v_o

def get_r(n, t, r_o, v_o):
    return get_rr(n, t) @ r_o + get_rv(n, t) @ v_o

if __name__ == "__main__":
    n = 7.2921 * (10 ** (-5))
    t = 21600
    r_o = [-10, 10, 0]
    print(get_v_o_plus(n, t, r_o))
    print(get_v_f_minus(n, t, r_o))
    print(get_v(7.2921 * (10 ** (-5)), 7200, [-10, 10, 0], [0, 0, 0]))
    print(get_v(n, 7200, [0, 0, 0], [-0.00177, 0.000587, 0]))