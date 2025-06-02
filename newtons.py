import numpy as np

def newtons_approx(fx, dfx, x1, tolerance=1e-7, max_iterations=100):
    """
    Approximate the root of a function using Newton's method.

    Parameters:
    fx (function): The function for which we want to find the root.
    dfx (function): The derivative of the function fx.
    x1 (float): Initial guess for the root.
    tolerance (float): The tolerance for convergence.
    max_iterations (int): Maximum number of iterations to perform.

    Returns:
    float: The approximated root of the function.
    """
    for i in range(max_iterations):
        f_x1 = fx(x1)
        d_f_x1 = dfx(x1)

        if d_f_x1 == 0:
            raise ValueError("Derivative is zero. No solution found.")

        x2 = x1 - f_x1 / d_f_x1

        if abs(x2 - x1) < tolerance:
            return x2

        x1 = x2

    raise ValueError("Maximum iterations reached. No solution found.")

if __name__ == "__main__":
    # Problem 3.1
    # def fx(x):
    #     return x**2 - 5*x + 4 - 10 * np.exp(np.sin(x))

    # def dfx(x):
    #     return 2*x - 5 - 10 * np.exp(np.sin(x)) * np.cos(x)

    # Problem 3.2
    def fx(x):
        return np.tanh(x) - np.tan(x)
    
    def dfx(x):
        return 1 / np.cosh(x)**2 - 1 / np.cos(x)**2

    initial_guess = 8
    root = newtons_approx(fx, dfx, initial_guess)
    print(f"Approximated root: {root}")
