"""
Write a Python function that computes the softmax activation for a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.

Example:
Input:
scores = [1, 2, 3]
Output:
[0.0900, 0.2447, 0.6652]
Reasoning:
The softmax function converts a list of values into a probability distribution. The probabilities are proportional to the exponential of each element divided by the sum of the exponentials of all elements in the list.
"""


import math

import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores_n = np.array(scores)
    scores_exp = np.exp(scores_n)
    scores_softmax = np.round(scores_exp / np.sum(scores_exp), 4)
    return scores_softmax.tolist()


if __name__ == "__main__":
    scores = [1, 2, 3]
    print(softmax(scores))