"""
Calculate the computational cost savings of an MoE layer compared to a dense layer, as discussed in the paper 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.' Given the number of experts, sparsity (number of active experts), and input/output dimensions, compute the floating-point operations (FLOPs) for both and determine the savings percentage.
Example:
Input:
compute_efficiency(1000, 2, 512, 512)
Output:
99.8
Reasoning:
Dense layer FLOPs: 1000 * 512 * 512 = 262,144,000. MoE FLOPs: 2 * 512 * 512 = 524,288. Savings: ((262,144,000 - 524,288) / 262,144,000) x 100 ≈ 99.8%.
"""


def compute_efficiency(n_experts, k_active, d_in, d_out):
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs
    """
    return (1 - k_active / n_experts) * 100
