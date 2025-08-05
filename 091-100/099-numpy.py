"""
Implement the Softplus activation function, a smooth approximation of the ReLU function. Your task is to compute the Softplus value for a given input, handling edge cases to prevent numerical overflow or underflow.

Example:
Input:
softplus(2)
Output:
2.1269
Reasoning:
For x = 2, the Softplus activation is calculated as log(1+e^x).
"""


import math

def softplus(x: float) -> float:
    """
    Compute the softplus activation function.

    Args:
        x: Input value

    Returns:
        The softplus value: log(1 + e^x)
    """
    # Your code here
    val = math.log(1 + math.exp(x))
    return round(val,4)


if __name__ == "__main__":
    print(softplus(2))