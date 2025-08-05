"""
Implement the ELU (Exponential Linear Unit) activation function, which helps mitigate the limitations of ReLU by providing negative outputs for negative inputs. The function should compute the ELU activation value for a given input.

Example:
Input:
elu(-1)
Output:
-0.6321
Reasoning:
For x = -1 and alpha = 1.0, the ELU activation is computed as 
α(e^x−1).
"""


import math

def elu(x: float, alpha: float = 1.0) -> float:
    """
    Compute the ELU activation function.

    Args:
        x (float): Input value
        alpha (float): ELU parameter for negative values (default: 1.0)

    Returns:
        float: ELU activation value
    """
    # Your code here
    if x >= 0:
        val = float(x)
    else:
        val = alpha * (math.exp(x) - 1)
    return round(val,4)


if __name__ == "__main__":
    print(elu(1))