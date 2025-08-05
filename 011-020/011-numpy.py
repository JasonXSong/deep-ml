"""
Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b. The function should iterate n times, rounding each intermediate solution to four decimal places, and return the approximate solution x.

Example:
Input:
A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b = [-1, 2, 3], n=2
Output:
[0.146, 0.2032, -0.5175]
Reasoning:
The Jacobi method iteratively solves each equation for x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)), where a_ii is the diagonal element of A and a_ij are the off-diagonal elements.
"""


import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros(b.shape[0])
    A_diag = np.diag(np.diag(A))
    A_diag_inv = np.linalg.inv(A_diag)
    for i in range(n):
        xn = A_diag_inv @ (b - (A - A_diag) @ x)
        x = xn
    return x


if __name__ == "__main__":
    A = np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]])
    b = np.array([-1, 2, 3])
    n = 2
    print(solve_jacobi(A, b, n))
