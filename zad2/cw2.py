import numpy as np
import sympy as sp

# Funkcja do obliczeń
def f(x):
    return np.exp(x)

# Symboliczna wersja funkcji do obliczeń analitycznych
x = sp.symbols('x')
f_sym = sp.exp(x)

# Całkowanie
def rectangle_rule(f, a, b, n):
    dx = (b - a) / n
    result = 0
    for i in range(n):
        result += f(a + i * dx) * dx
    return result

def trapezoidal_rule(f, a, b, n):
    result = 0
    dx  = (b-a)/n
    for i in range(n):
        fa = a + dx * i
        fb = a + dx * (i + 1)
        result += (f(fa) + f(fb)) / 2 * dx
    return result

def simpson_rule(f, a, b, n):
    if n % 2:
        n += 1
    dx = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n, 2):
        result += 4 * f(a + i * dx)
    for i in range(2, n-1, 2):
        result += 2 * f(a + i * dx)
    result *= dx / 3
    return result

def simpson_3_8_rule(f, a, b, n):
    if n % 3 != 0:
        n += 3 - (n % 3)  # Upewnij się, że n jest wielokrotnością 3
    dx = (b - a) / n
    result = 0

    for i in range(0, n, 3):
        x0 = a + i * dx
        x1 = x0 + dx
        x2 = x0 + 2 * dx
        x3 = x0 + 3 * dx
        result += (3 * dx / 8) * (f(x0) + 3 * f(x1) + 3 * f(x2) + f(x3))
    
    return result

def nc2_rule(f, a, b, n):
    if n % 4 != 0:
        n += 4 - (n % 4)  # Upewnij się, że n jest wielokrotnością 4
    dx = (b - a) / n
    result = 0

    for i in range(0, n, 4):
        x0 = a + i * dx
        x1 = x0 + dx
        x2 = x0 + 2 * dx
        x3 = x0 + 3 * dx
        result += (4 * dx / 3) * (2 * f(x1) - f(x2) + 2 * f(x3))
    
    return result

def nc4_rule(f, a, b, n):
    if n % 6 != 0:
        n += 6 - (n % 6)  # Upewnij się, że n jest wielokrotnością 6
    dx = (b - a) / n
    result = 0

    for i in range(0, n, 6):
        x0 = a + i * dx
        x1 = x0 + dx
        x2 = x0 + 2 * dx
        x3 = x0 + 3 * dx
        x4 = x0 + 4 * dx
        x5 = x0 + 5 * dx
        result += (6 * dx / 20) * (11 * f(x1) - 14 * f(x2) + 26 * f(x3) - 14 * f(x4) + 11 * f(x5))
    
    return result

def gauss_legendre_quadrature(f, a, b, N):
    if N == 1:
        x = 0
        w = 2
        result = w * f((b - a) / 2 * x + (b + a) / 2)
    elif N == 2:
        x = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        w = np.array([1, 1])
        result = 0
        for i in range(N):
            result += w[i] * f((b - a) / 2 * x[i] + (b + a) / 2)
        result *= (b - a) / 2
    else:
        raise ValueError("Currently, only N=1 and N=2 are supported.")
    return result

# Pochodne
def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h):
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def forward_difference2(f, x, h):
    return (-f(x + 2*h) + 4*f(x + h) - 3*f(x)) / (2*h)

def backward_difference2(f, x, h):
    return (3*f(x) - 4*f(x - h) + f(x - 2*h)) / (2*h)

def central_difference2(f, x, h):
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12*h)

def forward_difference_second(f, x, h):
    return (f(x + 2*h) - 2*f(x + h) + f(x)) / (h**2)

def backward_difference_second(f, x, h):
    return (f(x) - 2*f(x - h) + f(x - 2*h)) / (h**2)

def central_difference_second(f, x, h):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

def central_difference_fourth(f, x, h):
    return (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h)) / (12*h**2)

# Testowanie funkcji
a = 0
b = 2*np.pi
n = 3

h = 0.01
x_val = np.pi / 4

# Analityczne obliczenia
integral_exact = sp.integrate(f_sym, (x, a, b)).evalf()
first_derivative_exact = sp.diff(f_sym, x).subs(x, x_val).evalf()
second_derivative_exact = sp.diff(f_sym, x, 2).subs(x, x_val).evalf()

# Numeryczne obliczenia
integral_rectangle = rectangle_rule(f, a, b, n)
integral_trapezoidal = trapezoidal_rule(f, a, b, n)
integral_simpson = simpson_rule(f, a, b, n)
integral_simpson_3_8 = simpson_3_8_rule(f, a, b, n)
integral_nc2 = nc2_rule(f, a, b, n)
integral_nc4 = nc4_rule(f, a, b, n)
integral_gauss_1 = gauss_legendre_quadrature(f, a, b, 1)
integral_gauss_2 = gauss_legendre_quadrature(f, a, b, 2)

first_derivative_forward = forward_difference(f, x_val, h)
first_derivative_backward = backward_difference(f, x_val, h)
first_derivative_central = central_difference(f, x_val, h)
second_derivative_forward= forward_difference_second(f, x_val, h)
second_derivative_backward = backward_difference_second(f, x_val, h)
second_derivative_central = central_difference_second(f, x_val, h)
second_derivative_fourth = central_difference_fourth(f, x_val, h)

print("Całkowanie metodą prostokątów: \t\t", integral_rectangle)
print("Całkowanie metodą trapezów:\t\t", integral_trapezoidal)
print("Całkowanie metodą Simpsona:\t\t", integral_simpson)
print("Całkowanie metodą Simpsona 3/8:\t\t", integral_simpson_3_8)
print("Całkowanie metodą Newtona-Cotesa 2:\t", integral_nc2)
print("Całkowanie metodą Newtona-Cotesa 4:\t", integral_nc4)
print("Całkowanie kwadraturą Gaussa dla N=1:\t", integral_gauss_1)
print("Całkowanie kwadraturą Gaussa dla N=2:\t", integral_gauss_2)
print("Całka analityczna:\t\t\t", integral_exact)

print()

print("Pierwsza pochodna (prawa):\t\t", first_derivative_forward)
print("Pierwsza pochodna (lewa):\t\t", first_derivative_backward)
print("Pierwsza pochodna (centralna):\t\t", first_derivative_central)
print("Pierwsza pochodna analityczna:\t\t", first_derivative_exact)

print("Druga pochodna (prawa):\t\t\t", second_derivative_forward)
print("Druga pochodna (lewa):\t\t\t", second_derivative_backward)
print("Druga pochodna (centralna):\t\t", second_derivative_central)
print("Druga pochodna (centralna czwarta):\t", second_derivative_fourth)
print("Druga pochodna analityczna:\t\t", second_derivative_exact)

