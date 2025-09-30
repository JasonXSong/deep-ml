"""
Implement a function that performs Min-Max Normalization on a list of integers, scaling all values to the range [0, 1]. Min-Max normalization helps ensure that all features contribute equally to a model by scaling them to a common range.
Example:
Input:
min_max([1, 2, 3, 4, 5])
Output:
[0.0, 0.25, 0.5, 0.75, 1.0]
Reasoning:
The minimum value is 1 and the maximum is 5. Each value is scaled using the formula (x - min) / (max - min).
"""


def min_max(x: list[int]) -> list[float]:
    # Your code here
    min_val = min(x)
    max_val = max(x)
    if min_val == max_val:
        return [0] * len(x)
    return [(xx - min_val) / (max_val - min_val) for xx in x]


if __name__ == "__main__":
    print(min_max([1, 2, 3, 4, 5]))
