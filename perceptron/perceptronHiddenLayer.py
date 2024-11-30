import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100, hidden_layer_size=2):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_layer_size = hidden_layer_size
        self.hidden_weights = np.random.rand(3, hidden_layer_size)
        self.output_weights = np.random.rand(hidden_layer_size + 1)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def predict(self, inputs):
        inputs = np.append(inputs, 1)
        hidden_input = np.dot(inputs, self.hidden_weights)
        hidden_output = self.relu(hidden_input)
        hidden_output = np.append(hidden_output, 1)
        net_input = np.dot(hidden_output, self.output_weights)
        return 1 if net_input >= 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                inputs = np.append(inputs, 1)
                hidden_input = np.dot(inputs, self.hidden_weights)
                hidden_output = self.relu(hidden_input)
                hidden_output = np.append(hidden_output, 1)
                self.output_weights += self.learning_rate * error * hidden_output
                hidden_gradient = error * self.output_weights[:-1] * (hidden_input > 0)
                self.hidden_weights += self.learning_rate * np.outer(inputs, hidden_gradient)


training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 1])

perceptron = Perceptron(learning_rate=0.1, epochs=10)
perceptron.train(training_inputs, labels)

print("Testowanie perceptrona:")
for inputs in training_inputs:
    print(f"Dla wejścia {inputs}, wynik to: {perceptron.predict(inputs)}")
