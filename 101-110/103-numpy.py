"""
Implement the SELU (Scaled Exponential Linear Unit) activation function, a self-normalizing variant of ELU. Your task is to compute the SELU value for a given input while ensuring numerical stability.

Example:
Input:
selu(-1.0)
Output:
-1.1113
Reasoning:
For x = -1.0, the SELU activation is calculated using the formula SELU(x)=λα(e^x -1). Substituting the values of 
λ and α, we get SELU(-1.0)=1.0507 * 1.6733 * (e^{-1.0} -1)=-1.1113.
"""


import numpy as np

def selu(x: float) -> float:
    """
    Implements the SELU (Scaled Exponential Linear Unit) activation function.

    Args:
        x: Input value

    Returns:
        SELU activation value
    """
    alpha = 1.6732632423543772
    scale = 1.0507009873554804
    # Your code here
    if x >= 0:
        val = scale * x
    else:
        val = scale * alpha * (np.exp(x) - 1)
    return val


if __name__ == "__main__":
    print(round(selu(-1.0), 4))
    print(round(selu(1.0), 4))
