import numpy as np
from graphviz import Digraph


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.biases_input_hidden = np.random.randn(1, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.biases_hidden_output = np.random.randn(1, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, inputs):
        # Input to hidden layer
        hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.biases_input_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)

        # Hidden to output layer
        output_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.biases_hidden_output
        output_outputs = self.sigmoid(output_inputs)

        return output_outputs

    def train(self, inputs, targets, learning_rate):
        # Feedforward
        hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.biases_input_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)
        output_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.biases_hidden_output
        output_outputs = self.sigmoid(output_inputs)

        # Backpropagation
        output_error = targets - output_outputs
        output_delta = output_error * self.sigmoid_derivative(output_outputs)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_outputs)

        # Update weights and biases
        self.weights_hidden_output += np.dot(hidden_outputs.T, output_delta) * learning_rate
        self.biases_hidden_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * learning_rate
        self.biases_input_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def plot_neural_network(self):
        # Create Graphviz object
        dot = Digraph()

        # Set the direction of the graph to be from left to right
        dot.attr(rankdir='LR')

        # Add nodes for input layer
        for i in range(self.input_size):
            dot.node(f'I{i}', label=f'Input {i}')

        # Add nodes for hidden layer
        for i in range(self.hidden_size):
            dot.node(f'H{i}', label=f'Hidden {i}')

        # Add node for output layer
        dot.node('O', label='Output')

        # Add edges from input layer to hidden layer
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                dot.edge(f'I{i}', f'H{j}', label=f'Weight: {self.weights_input_hidden[i][j]:.2f}')

        # Add edges from hidden layer to output layer
        for i in range(self.hidden_size):
            dot.edge(f'H{i}', 'O', label=f'Weight: {self.weights_hidden_output[i][0]:.2f}')

        # Render and display the graph
        dot.render('neural_network', format='png', cleanup=True)
        dot.view()


# Example usage
input_size = 2
hidden_size = 10
output_size = 1

# Create a neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Example training data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Train the neural network
for i in range(100000):
    nn.train(inputs, targets, learning_rate=0.1)

# Plot the neural network
nn.plot_neural_network()

# Test the neural network
for i in range(len(inputs)):
    print(f"Input: {inputs[i]} | Predicted Output: {nn.feedforward(inputs[i])}")
