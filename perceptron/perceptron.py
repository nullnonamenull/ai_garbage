import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(3)

    def predict(self, inputs):
        inputs = np.append(inputs, 1)
        net_input = np.dot(inputs, self.weights)
        return 1 if net_input >= 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                inputs = np.append(inputs, 1)
                self.weights += self.learning_rate * error * inputs


training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 1])

perceptron = Perceptron(learning_rate=0.1, epochs=10)
perceptron.train(training_inputs, labels)

print("Testowanie perceptrona:")
for inputs in training_inputs:
    print(f"Dla wejścia {inputs}, wynik to: {perceptron.predict(inputs)}")
