import numpy as np

# Metoda Newtona-Raphsona
def newton_raphson(f, df, x0, tol=1e-6, max_iter=3):
    x = x0
    for _ in range(max_iter):
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Metoda bisekcji
def bisection(f, a, b, tol=1e-6, max_iter=3):
    if f(a) * f(b) > 0:
        raise ValueError("Funkcja musi mieć różne znaki na krańcach przedziału [a, b].")
    for _ in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

# Metoda podziału na trzy części (poszukiwanie minimum)
def trisection_min(f, a, b, tol=1e-6, max_iter=3):
    for _ in range(max_iter):
        c1 = a + (b - a) / 3
        c2 = b - (b - a) / 3
        if abs(b - a) < tol:
            return (a + b) / 2
        if f(c1) < f(c2):
            b = c2
        else:
            a = c1
    return (a + b) / 2

# Metoda reguła falsi
def regula_falsi(f, a, b, tol=1e-6, max_iter=3):
    if f(a) * f(b) > 0:
        raise ValueError("Funkcja musi mieć różne znaki na krańcach przedziału [a, b].")
    for _ in range(max_iter):
        c = a - f(a) * (b - a) / (f(b) - f(a))
        if abs(f(c)) < tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c

# Metoda siecznych
def secant(f, x0, x1, tol=1e-6, max_iter=3):
    for _ in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    return x1

# Metoda złotego podziału (poszukiwanie minimum)
def golden_section_min(f, a, b, tol=1e-6, max_iter=3):
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    c = b - resphi * (b - a)
    d = a + resphi * (b - a)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            return (a + b) / 2
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - resphi * (b - a)
        d = a + resphi * (b - a)
    return (a + b) / 2



if __name__ == "__main__":
    f = lambda x: x**3 - x - 2
    df = lambda x: 3*x**2 - 1
    F = lambda x: np.array([x[0]**2 + x[1]**2 + x[2]**2 - 1, 
                            x[0] + x[1] + x[2] - 1, 
                            x[0]*x[1]*x[2] - 0.1])
    J = lambda x: np.array([[2*x[0], 2*x[1], 2*x[2]], 
                            [1, 1, 1], 
                            [x[1]*x[2], x[0]*x[2], x[0]*x[1]]])

    iterations = 5
    # Przykłady dla metod jednej zmiennej
    root_newton = newton_raphson(f, df, x0=1.5, max_iter=iterations)
    root_bisection = bisection(f, a=1, b=2, max_iter=iterations)
    min_trisection = trisection_min(f, a=0, b=2, max_iter=iterations)
    root_regula_falsi = regula_falsi(f, a=1, b=2, max_iter=iterations)
    root_secant = secant(f, x0=1, x1=2, max_iter=iterations)
    min_golden_section = golden_section_min(f, a=1, b=2, max_iter=iterations)


    print("Metoda Newtona-Raphsona:\t", root_newton)
    print("Metoda bisekcji:\t\t", root_bisection)
    print("Metoda podziału na trzy części:\t", min_trisection)
    print("Metoda reguła falsi:\t\t", root_regula_falsi)
    print("Metoda siecznych:\t\t", root_secant)
    print("Metoda złotego podziału:\t", min_golden_section) 