"""
Implement a function to generate train and test splits for K-Fold Cross-Validation. Your task is to divide the dataset into k folds and return a list of train-test indices for each fold.
Example:
Input:
k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=5, shuffle=False)
Output:
[([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]), ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]), ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]), ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]
Reasoning:
The function splits the dataset into 5 folds without shuffling and returns train-test splits for each iteration.
"""


import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Implement k-fold cross-validation by returning train-test indices.
    """
    # Your code here
    length = len(X)
    indices = np.arange(length)
    if shuffle:
        np.random.shuffle(indices)
    part = length // k
    ret = list()
    for i in range(k):
        cur_train = list()
        cur_test = list()
        for j in range(length):
            if j >= i * part and j < (i+1) * part:
                cur_test.append(indices[j])
            else:
                cur_train.append(indices[j])
        ret.append((cur_train, cur_test))
    return ret


if __name__ == "__main__":
    X = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([0,1,2,3,4,5,6,7,8,9])
    print(k_fold_cross_validation(X, y, k=5, shuffle=False))
