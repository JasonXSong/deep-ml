"""
Task: Implement Long Short-Term Memory (LSTM) Network
Your task is to implement an LSTM network that processes a sequence of inputs and produces the final hidden state and cell state after processing all inputs.

Write a class LSTM with the following methods:

__init__(self, input_size, hidden_size): Initializes the LSTM with random weights and zero biases.
forward(self, x, initial_hidden_state, initial_cell_state): Processes a sequence of inputs and returns the hidden states at each time step, as well as the final hidden state and cell state.
The LSTM should compute the forget gate, input gate, candidate cell state, and output gate at each time step to update the hidden state and cell state.

Example:
Input:
input_sequence = np.array([[1.0], [2.0], [3.0]])
initial_hidden_state = np.zeros((1, 1))
initial_cell_state = np.zeros((1, 1))

lstm = LSTM(input_size=1, hidden_size=1)
outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

print(final_h)
Output:
[[0.73698596]] (approximate)
Reasoning:
The LSTM processes the input sequence [1.0, 2.0, 3.0] and produces the final hidden state [0.73698596].
"""


import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
        """
        h_t = initial_hidden_state
        c_t = initial_cell_state
        sequence_length = x.shape[0]
        outputs = np.zeros((sequence_length, self.hidden_size))
        for t in range(sequence_length):
            x_t = x[t][:, np.newaxis]
            concatenated_input = np.concatenate((h_t, x_t))
            f_t = sigmoid(np.matmul(self.Wf, concatenated_input) + self.bf)
            i_t = sigmoid(np.matmul(self.Wi, concatenated_input) + self.bi)
            o_t = sigmoid(np.matmul(self.Wo, concatenated_input) + self.bo)
            c_t_hat = np.tanh(np.matmul(self.Wc, concatenated_input) + self.bc)
            c_t = f_t * c_t + i_t * c_t_hat
            h_t = o_t * np.tanh(c_t)

            outputs[t] = h_t.reshape(-1)

        return outputs, h_t, c_t


if __name__ == "__main__":
    input_sequence = np.array([[0.1, 0.2], [0.3, 0.4]])
    initial_hidden_state = np.zeros((2, 1))
    initial_cell_state = np.zeros((2, 1))
    lstm = LSTM(input_size=2, hidden_size=2)
    # Set weights and biases for reproducibility
    lstm.Wf = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wi = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wc = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wo = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.bf = np.array([[0.1], [0.2]])
    lstm.bi = np.array([[0.1], [0.2]])
    lstm.bc = np.array([[0.1], [0.2]])
    lstm.bo = np.array([[0.1], [0.2]])
    outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)
    print(final_h)