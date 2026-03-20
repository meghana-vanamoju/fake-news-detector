"""
Pure NumPy inference engine for the Fake News Detector LSTM model.
No TensorFlow/Keras dependency required.
"""
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def lstm_step(x_t, h_prev, c_prev, kernel, recurrent_kernel, bias):
    units = h_prev.shape[-1]
    z = x_t @ kernel + h_prev @ recurrent_kernel + bias
    i = sigmoid(z[:, :units])
    f = sigmoid(z[:, units:2*units])
    c_candidate = np.tanh(z[:, 2*units:3*units])
    o = sigmoid(z[:, 3*units:])
    c = f * c_prev + i * c_candidate
    h = o * np.tanh(c)
    return h, c

def lstm_forward(x_seq, kernel, recurrent_kernel, bias, return_sequences=False):
    batch_size = x_seq.shape[0]
    timesteps = x_seq.shape[1]
    units = bias.shape[0] // 4
    
    h = np.zeros((batch_size, units))
    c = np.zeros((batch_size, units))
    
    if return_sequences:
        outputs = []
    
    for t in range(timesteps):
        h, c = lstm_step(x_seq[:, t, :], h, c, kernel, recurrent_kernel, bias)
        if return_sequences:
            outputs.append(h)
    
    if return_sequences:
        return np.stack(outputs, axis=1)
    return h

class NumpyModel:
    def __init__(self, weights_path="model_weights.npz"):
        w = np.load(weights_path)
        self.embedding = w["embedding_embeddings_0"]
        self.lstm1_kernel = w["lstm_lstm_cell_kernel_0"]
        self.lstm1_recurrent = w["lstm_lstm_cell_recurrent_kernel_0"]
        self.lstm1_bias = w["lstm_lstm_cell_bias_0"]
        self.lstm2_kernel = w["lstm_1_lstm_cell_kernel_0"]
        self.lstm2_recurrent = w["lstm_1_lstm_cell_recurrent_kernel_0"]
        self.lstm2_bias = w["lstm_1_lstm_cell_bias_0"]
        self.dense_kernel = w["dense_kernel_0"]
        self.dense_bias = w["dense_bias_0"]
    
    def predict(self, padded_sequence):
        # padded_sequence shape: (batch, 500) of integer token ids
        x = self.embedding[padded_sequence]  # (batch, 500, 128)
        x = lstm_forward(x, self.lstm1_kernel, self.lstm1_recurrent, self.lstm1_bias, return_sequences=True)
        x = lstm_forward(x, self.lstm2_kernel, self.lstm2_recurrent, self.lstm2_bias, return_sequences=False)
        # Dense + sigmoid
        out = x @ self.dense_kernel + self.dense_bias
        return sigmoid(out)
