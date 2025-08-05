"""
In machine learning and statistics, the softmax function is a generalization of the logistic function that converts a vector of scores into probabilities. The log-softmax function is the logarithm of the softmax function, and it is often used for numerical stability when computing the softmax of large numbers.

Given a 1D numpy array of scores, implement a Python function to compute the log-softmax of the array.

Example:
Input:
A = np.array([1, 2, 3])
print(log_softmax(A))
Output:
array([-2.4076, -1.4076, -0.4076])
Reasoning:
The log-softmax function is applied to the input array [1, 2, 3]. The output array contains the log-softmax values for each element.
"""


import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    # Your code here
    scores_n = np.array(scores)
    scores_exp = np.exp(scores_n)
    return np.round(np.log(scores_exp / np.sum(scores_exp)), 4)

if __name__ == "__main__":
    A = np.array([1, 2, 3])
    print(log_softmax(A))