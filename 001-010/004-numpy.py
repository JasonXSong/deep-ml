"""
Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.

Example:
Input:
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'
Output:
[4.0, 5.0, 6.0]
Reasoning:
Calculating the mean of each column results in [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3].
"""


import numpy as np


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == 'row':
        # (3, 3) -> (3, 1)
        return np.mean(matrix, axis=1).tolist()
    else:
        # (3, 3) -> (1, 3)
        return np.mean(matrix, axis=0).tolist()


if __name__ == '__main__':
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(calculate_matrix_mean(matrix, 'row'))
    print(calculate_matrix_mean(matrix, 'column'))