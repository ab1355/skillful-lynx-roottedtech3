import sympy as sp

def test_calculus():
    x = sp.Symbol('x')
    f = x**3 - 6*x**2 + 11*x - 6
    
    # Find critical points
    critical_points = sp.solve(sp.diff(f, x))
    
    # Find inflection points
    inflection_points = sp.solve(sp.diff(f, x, 2))
    
    # Calculate definite integral
    integral = sp.integrate(f, (x, 0, 2))
    
    return critical_points, inflection_points, integral

def test_linear_algebra():
    A = sp.Matrix([[1, 2], [3, 4]])
    B = sp.Matrix([[5, 6], [7, 8]])
    
    # Matrix multiplication
    C = A * B
    
    # Eigenvalues and eigenvectors
    eigenvalues = A.eigenvals()
    eigenvectors = A.eigenvects()
    
    # Determinant and inverse
    det_A = A.det()
    inv_A = A.inv()
    
    return C, eigenvalues, eigenvectors, det_A, inv_A

print("Calculus Test Results:", test_calculus())
print("Linear Algebra Test Results:", test_linear_algebra())