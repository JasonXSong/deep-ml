"""
Write a Python function to implement the Positional Encoding layer for Transformers. The function should calculate positional encodings for a sequence length (position) and model dimensionality (d_model) using sine and cosine functions as specified in the Transformer architecture. The function should return -1 if position is 0, or if d_model is less than or equal to 0. The output should be a numpy array of type float16.

Example:
Input:
position = 2, d_model = 8
Output:
[[[ 0.,0.,0.,0.,1.,1.,1.,1.,]
  [ 0.8413,0.0998,0.01,0.001,0.5405,0.995,1.,1.]]]
Reasoning:
The function computes the positional encoding by calculating sine values for even indices and cosine values for odd indices, ensuring that the encoding provides the required positional information.
"""


import numpy as np

def pos_encoding(position: int, d_model: int):
    # Your code here 
    
    if position == 0 or d_model <= 0:
        return -1
    pos_encoding = np.zeros((position, d_model), dtype=np.float16)

    for pos in range(position):
        for i in range(0, d_model, 2):
            denominator = np.power(10000, i / d_model)
            pos_encoding[pos, i] = np.sin(pos / denominator)
            pos_encoding[pos, i+1] = np.cos(pos / denominator)
    return np.round(pos_encoding, 4)


if __name__ == '__main__':
	print(pos_encoding(2, 8))
