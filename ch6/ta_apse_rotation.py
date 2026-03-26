import numpy as np
import sympy as sym


def _is_symbolic(*args):
    return any(isinstance(a, sym.Basic) for a in args)


def get_a(e1, e2, h1, h2, eta):
    cos = sym.cos if _is_symbolic(e1, e2, h1, h2, eta) else np.cos
    return (e1 * h2**2) - (e2 * h1**2 * cos(eta))

def get_b(e2, h1, eta):
    sin = sym.sin if _is_symbolic(e2, h1, eta) else np.sin
    return -e2 * h1**2 * sin(eta)

def get_c(h1, h2):
    return h1**2 - h2**2

def get_phi(e1, e2, h1, h2, eta):
    atan = sym.atan if _is_symbolic(e1, e2, h1, h2, eta) else np.arctan
    return atan(get_b(e2, h1, eta) / get_a(e1, e2, h1, h2, eta))

def ta_of_apse_rotation(e1, e2, h1, h2, eta):
    symbolic = _is_symbolic(e1, e2, h1, h2, eta)
    acos = sym.acos if symbolic else np.arccos
    cos = sym.cos if symbolic else np.cos
    a = get_a(e1, e2, h1, h2, eta)
    b = get_b(e2, h1, eta)
    c = get_c(h1, h2)
    phi = get_phi(e1, e2, h1, h2, eta)
    term2 = acos((c/a) * cos(phi))
    return [phi + term2, phi - term2]


if __name__ == "__main__":
    e1 = sym.Symbol('e')
    e2 = e1
    h1 = sym.Symbol('h')
    h2 = h1
    eta = sym.Symbol('eta')
    thetas = ta_of_apse_rotation(e1, e2, h1, h2, eta)
    print(f"Thetas Rad: {thetas}")
    for t in thetas:
        print(f"  simplified: {sym.simplify(t)}")