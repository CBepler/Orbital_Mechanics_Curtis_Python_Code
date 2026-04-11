import numpy as np

MU = 398600

def get_rr(n, t):
    return np.matrix([[4 - 3 * np.cos(n * t), 0, 0], [6 * (np.sin(n * t) - n * t), 1, 0], [0, 0, np.cos(n * t)]])

def get_rv(n, t):
    return np.matrix([[(1/n) * np.sin(n * t), (2/n) * (1 - np.cos(n * t)), 0], [(2/n) * (np.cos(n * t) - 1), (1/n) * (4 * np.sin(n * t) - 3 * n * t), 0], [0, 0, (1/n) * np.sin(n * t)]])

def get_vr(n, t):
    return np.matrix([[3 * n * np.sin(n * t), 0, 0], [6 * n *(np.cos(n * t) - 1), 0, 0], [0, 0, -n * np.sin(n * t)]])

def get_vv(n, t):
    return np.matrix([[np.cos(n * t), 2 * np.sin(n * t), 0], [-2 * np.sin(n * t), 4 * np.cos(n * t) - 3, 4 * np.cos(n * t) - 3, 0], [0, 0, np.cos(n * t)]])

def delta_v(r_a, delta_r_o, delta_v_o_minus, t):
    T = ((2 * np.pi) / np.sqrt(MU)) * (r_a ** (3/2))
    n = (2 * np.pi) / T
    rr = get_rr(n, t)
    rv = get_rv(n, t)
    vr = get_vr(n, t)
    vv = get_vv(n, t)
    delta_v_o_plus = -np.cross(np.linalg.inv(rv), np.cross(rr, delta_r_o))
    delta_v_f_minus = np.cross((vr - np.matmul(vv, np.linalg.inv(rv))), delta_r_o)
    return (delta_v_o_plus - delta_v_o_minus, -delta_v_f_minus)

if __name__ == "__main__":
    print(delta_v(6600, np.array([1, 1, 1]), np.array([0, 0, 0.005]), 1778.7))