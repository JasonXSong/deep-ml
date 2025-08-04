"""
Write a Python function that calculates the eigenvalues of a 2x2 matrix. The function should return a list containing the eigenvalues, sort values from highest to lowest.

Example:
Input:
matrix = [[2, 1], [1, 2]]
Output:
[3.0, 1.0]
Reasoning:
The eigenvalues of the matrix are calculated using the characteristic equation of the matrix, which for a 2x2 matrix is $ λ^2 - trace(A)λ+det(A) = 0 $, where λ are the eigenvalues.
"""


import numpy as np


def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    matrix_n = np.array(matrix)
    eig, _ = np.linalg.eig(matrix_n)
    return eig.tolist()


if __name__ == "__main__":
    matrix = [[2, 1], [1, 2]]
    print(calculate_eigenvalues(matrix))