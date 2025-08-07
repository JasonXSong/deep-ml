"""
Implement a function that computes the average cross-entropy loss for a batch of predictions in a multi-class classification task. Your function should take in a batch of predicted probabilities and one-hot encoded true labels, then return the average cross-entropy loss. Ensure that you handle numerical stability by clipping probabilities by epsilon.

Example:
Input:
predicted_probs = [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]]
true_labels = [[1, 0, 0], [0, 1, 0]]
Output:
0.4338
Reasoning:
The predicted probabilities for the correct classes are 0.7 and 0.6. The cross-entropy is computed as -mean(log(0.7), log(0.6)), resulting in approximately 0.4463.
"""


import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    clipped_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    ce = - np.sum(true_labels * np.log(clipped_probs)) / predicted_probs.shape[0]
    return ce


if __name__ == "__main__":
    predicted_probs = np.array([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]])
    true_labels = np.array([[1, 0, 0], [0, 1, 0]])
    print(compute_cross_entropy_loss(predicted_probs, true_labels))

    pred = np.array([[1, 0, 0], [0, 1, 0]])
    true = np.array([[1, 0, 0], [0, 1, 0]])
    print(round(compute_cross_entropy_loss(pred, true), 4))