"""
Implement a function to compute the Pointwise Mutual Information (PMI) given the joint occurrence count of two events, their individual counts, and the total number of samples. PMI measures how much the actual joint occurrence of events differs from what we would expect by chance.

Example:
Input:
compute_pmi(50, 200, 300, 1000)
Output:
-0.263
Reasoning:
The PMI calculation compares the actual joint probability (50/1000 = 0.05) to the product of the individual probabilities (200/1000 * 300/1000 = 0.06). Thus, PMI = log₂(0.05 / (0.2 * 0.3)) ≈ -0.263, indicating the events co-occur slightly less than expected by chance.
"""


import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
    # Implement PMI calculation here
    joint_probability = joint_counts / total_samples
    individual_probability = total_counts_x /total_samples * total_counts_y / total_samples
    pmi = np.log2(joint_probability / individual_probability)
    return round(pmi, 3)


if __name__ == "__main__":
    print(compute_pmi(50, 200, 300, 1000))