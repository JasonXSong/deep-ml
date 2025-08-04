"""
Write a Python function that multiplies a matrix by a scalar and returns the result.

Example:
Input:
matrix = [[1, 2], [3, 4]], scalar = 2
Output:
[[2, 4], [6, 8]]
Reasoning:
Each element of the matrix is multiplied by the scalar.
"""


import numpy as np


def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    mat = np.array(matrix)
    return (mat * scalar).tolist()


if __name__ == "__main__":
    matrix = [[1, 2], [3, 4]]
    scalar = 2
    print(scalar_multiply(matrix, scalar))