"""
Implement the Softsign activation function, a smooth activation function used in neural networks. Your task is to compute the Softsign value for a given input, ensuring the output is bounded between -1 and 1.

Example:
Input:
softsign(1)
Output:
0.5
Reasoning:
For x = 1, the Softsign activation is calculated as x / (1+|x|).
"""


def softsign(x: float) -> float:
	"""
	Implements the Softsign activation function.

	Args:
		x (float): Input value

	Returns:
		float: The Softsign of the input	"""
	# Your code here
	val = x / (1 + abs(x))
	return round(val,4)
