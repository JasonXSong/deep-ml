"""
Implement the Adam optimizer update step function. Your function should take the current parameter value, gradient, and moving averages as inputs, and return the updated parameter value and new moving averages. The function should also handle scalar and array inputs and include bias correction for the moving averages.

Example:
Input:
parameter = 1.0, grad = 0.1, m = 0.0, v = 0.0, t = 1
Output:
(0.999, 0.01, 0.0001)
Reasoning:
The Adam optimizer computes updated values for the parameter, first moment (m), and second moment (v) using bias-corrected estimates of gradients. With input values parameter=1.0, grad=0.1, m=0.0, v=0.0, and t=1, the updated parameter becomes 0.999.
"""


import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adam optimizer.
    Adjusts the learning rate based on the moving averages of the gradient and squared gradient.
    :param parameter: Current parameter value
    :param grad: Current gradient
    :param m: First moment estimate
    :param v: Second moment estimate
    :param t: Current timestep
    :param learning_rate: Learning rate (default=0.001)
    :param beta1: First moment decay rate (default=0.9)
    :param beta2: Second moment decay rate (default=0.999)
    :param epsilon: Small constant for numerical stability (default=1e-8)
    :return: tuple: (updated_parameter, updated_m, updated_v)
    """
    # Your code here
    mk = beta1 * m + (1 - beta1) * grad
    vk = beta2 * v + (1 - beta2) * np.square(grad)
    mk_corr = mk / (1 - beta1 ** t)
    vk_corr = vk / (1 - beta2 ** t)
    parameter = parameter - learning_rate * mk_corr / (np.sqrt(vk_corr) + epsilon)
    return np.round(parameter,5), np.round(mk,5), np.round(vk,5)


if __name__ == "__main__":
	parameter = 1.0
	grad = 0.1
	m = 0.0
	v = 0.0
	t = 1
	print(adam_optimizer(parameter, grad, m, v, t))
