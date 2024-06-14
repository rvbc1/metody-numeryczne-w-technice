import numpy as np
import json
import scipy.linalg

def read_matrices(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        return data

def is_diagonally_dominant(A):
    D = np.diag(np.abs(A))
    S = np.sum(np.abs(A), axis=1) - D
    return np.all(D > S)

def gauss_elimination(A, B):
    n = len(A)
    C = np.zeros((n, n + 1))
    X = np.zeros(n)

    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j]
        C[i][n] = B[i]

    for p in range(n - 1):
        for i in range(p + 1, n):
            factor = C[i][p] / C[p][p]
            for j in range(p + 1, n + 1):
                C[i][j] -= factor * C[p][j]

    X[n - 1] = C[n - 1][n] / C[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += C[i][j] * X[j]
        X[i] = (C[i][n] - suma) / C[i][i]

    return X

def simple_iteration(A, b, tolerance=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x) < tolerance:
            return x_new
        x = x_new
    raise ValueError("Simple Iteration did not converge")

def lu_decomposition(A):
    A = np.array(A)
    n = A.shape[0]
    
    lower = np.zeros((n, n))
    upper = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            sum = np.dot(lower[i, :i], upper[:i, j])
            upper[i, j] = A[i, j] - sum
 
        lower[i, i] = 1
        
        for j in range(i+1, n):
            sum = np.dot(lower[j, :i], upper[:i, i])
            lower[j, i] = (A[j, i] - sum) / upper[i, i]
 
    return lower, upper


def gauss_seidel(A, b, tolerance=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x_new - x) < tolerance:
            return x_new
        x = x_new
    raise ValueError("Gauss-Seidel did not converge")

def thomas_algorithm(A, b):
    n = len(b)
    a = np.zeros(n-1, float)
    b_diag = np.zeros(n, float)
    c = np.zeros(n-1, float)
    
    for i in range(n):
        b_diag[i] = A[i][i]
        if i < n - 1:
            c[i] = A[i][i+1]
            a[i] = A[i+1][i]

    w = np.zeros(n-1, float)
    g = np.zeros(n, float)
    p = np.zeros(n, float)
    
    w[0] = c[0] / b_diag[0]
    g[0] = b[0] / b_diag[0]

    for i in range(1, n-1):
        w[i] = c[i] / (b_diag[i] - a[i-1] * w[i-1])
    
    for i in range(1, n):
        g[i] = (b[i] - a[i-1] * g[i-1]) / (b_diag[i] - a[i-1] * w[i-1])
    
    p[n-1] = g[n-1]
    for i in range(n-2, -1, -1):
        p[i] = g[i] - w[i] * p[i+1]
    
    return p

def sor(A, b, omega=1.25, tolerance=1e-10, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x_new - x) < tolerance:
            return x_new
        x = x_new
    raise ValueError("SOR did not converge")

def format_solution(solution):
    return [f"{x:.2f}" for x in solution]

def compare_with_scipy(results, scipy_result):
    for method, result in results.items():
        if result is not None:
            norm_diff = np.linalg.norm(np.array(result) - np.array(scipy_result))
            print(f"Difference between {method} and SciPy: {norm_diff:.2e}")

def main():
    filename = 'matrix.json'
    data = read_matrices(filename)
    
    for i, dataset in enumerate(data):
        A = np.array(dataset['A'])
        b = np.array(dataset['b'])
        
        print(f"\nDataset {i+1}:")
        for row, bi in zip(A, b):
            print(' '.join(f"{elem:6.2f}" for elem in row), f" | {bi:6.2f}")
        
        if not is_diagonally_dominant(A):
            print("Warning: The matrix is not diagonally dominant. Iterative methods may not converge.")
        
        results = {}
        
        try:
            gauss_result = gauss_elimination(A.copy(), b.copy())
            results["Gauss Elimination"] = gauss_result
            print("Gauss Elimination:\t", format_solution(gauss_result))
        except Exception as e:
            print(f"Gauss Elimination failed: {e}")
        
        try:
            simple_iter_result = simple_iteration(A.copy(), b.copy())
            results["Simple Iteration"] = simple_iter_result
            print("Simple Iteration:\t", format_solution(simple_iter_result))
        except Exception as e:
            print(f"Simple Iteration failed: {e}")
        
        try:
            L, U = lu_decomposition(A.copy())
            # print(L)
            # print(U)
            # print(np.dot(L, U))
            y = gauss_elimination(L.copy(), b.copy())
            x = gauss_elimination(U.copy(), y.copy())

            results["LU Decomposition"] = x
            print("LU Decomposition:\t", format_solution(x))
        except Exception as e:
            print(f"LU Decomposition failed: {e}")
        
        try:
            gauss_seidel_result = gauss_seidel(A.copy(), b.copy())
            results["Gauss-Seidel"] = gauss_seidel_result
            print("Gauss-Seidel:\t\t", format_solution(gauss_seidel_result))
        except Exception as e:
            print(f"Gauss-Seidel failed: {e}")
        
        try:
            thomas_result = thomas_algorithm(A.copy(), b.copy())
            results["Thomas Algorithm"] = thomas_result
            print("Thomas Algorithm:\t", format_solution(thomas_result))
        except Exception as e:
            print(f"Thomas Algorithm failed: {e}")
        
        try:
            sor_result = sor(A.copy(), b.copy())
            results["SOR"] = sor_result
            print("SOR:\t\t\t", format_solution(sor_result))
        except Exception as e:
            print(f"SOR failed: {e}")
        
        try:
            scipy_result = scipy.linalg.solve(A, b)
            print("SciPy Solve:\t\t", format_solution(scipy_result))
        except Exception as e:
            print(f"SciPy Solve failed: {e}")
        
        print("\nComparing results with SciPy:")
        compare_with_scipy(results, scipy_result)

if __name__ == "__main__":
    main()
