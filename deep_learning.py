!pip install scikit-learn matplotlib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

# Problem 1: Load MNIST dataset
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)

    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    return train_test_split(X, y_onehot, test_size=0.2, random_state=42), X, y

# Problem 4 & 5: Activation Functions
class ReLU:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def backward(x):
        return (x > 0).astype(float)

class Sigmoid:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        sigmoid_x = Sigmoid.forward(x)
        return sigmoid_x * (1 - sigmoid_x)

class Softmax:
    @staticmethod
    def forward(x):
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def backward(predictions, labels):
        return predictions - labels

# Initializer and Optimizer classes
class XavierInitializer:
    def W(self, output_size, input_size):
        return np.random.randn(output_size, input_size) * np.sqrt(2 / (input_size + output_size))

    def B(self, output_size):
        return np.zeros((output_size, 1))

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, weights, biases, dW, dB):
        weights = weights - self.learning_rate * dW
        biases = biases - self.learning_rate * dB
        return weights, biases

# Problem 2 & 3: Fully Connected Layer
class FullyConnectedLayer:
    def __init__(self, input_size, output_size, initializer, optimizer, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.optimizer = optimizer
        self.weights = initializer.W(output_size, input_size)
        self.biases = initializer.B(output_size)

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights.T) + self.biases.T
        self.a = self.activation.forward(self.z)
        return self.a

    def backward(self, dA, labels=None):
        if isinstance(self.activation, Softmax):
            dZ = self.activation.backward(dA, labels)
        else:
            dZ = dA * self.activation.backward(self.z)

        dW = np.dot(self.input.T, dZ) / self.input.shape[0]
        dB = np.sum(dZ, axis=0, keepdims=True).T / self.input.shape[0]
        dA_prev = np.dot(dZ, self.weights)

        self.weights, self.biases = self.optimizer.update(self.weights, self.biases, dW.T, dB)
        return dA_prev

# Problem 8: Deep Neural Network Classifier
class ScratchDeepNeuralNetworkClassifier:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_gradient, y_batch):
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                loss_gradient = layer.backward(loss_gradient, y_batch)
            else:
                loss_gradient = layer.backward(loss_gradient)

    def train(self, x_train, y_train, x_val, y_val, epochs=100, batch_size=32):
        num_batches = x_train.shape[0] // batch_size
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]

                predictions = self.forward(x_batch)
                loss_gradient = Softmax.backward(predictions, y_batch)
                self.backward(loss_gradient, y_batch)

            # Compute losses after epoch
            train_pred = self.forward(x_train)
            val_pred = self.forward(x_val)

            train_loss = -np.sum(y_train * np.log(train_pred + 1e-7)) / y_train.shape[0]
            val_loss = -np.sum(y_val * np.log(val_pred + 1e-7)) / y_val.shape[0]

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Plot learning curve
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

# Problem 9: Learning and Estimation
(X_train, X_test, Y_train, Y_test), X, y = load_mnist()

# Show MNIST examples
plt.figure(figsize=(8, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

layers = [
    FullyConnectedLayer(784, 128, XavierInitializer(), SGD(0.01), ReLU()),
    FullyConnectedLayer(128, 64, XavierInitializer(), SGD(0.01), ReLU()),
    FullyConnectedLayer(64, 10, XavierInitializer(), SGD(0.01), Softmax())
]

model = ScratchDeepNeuralNetworkClassifier(layers)
model.train(X_train, Y_train, X_test, Y_test, epochs=10, batch_size=32)

y_pred = model.predict(X_test)
y_true = np.argmax(Y_test, axis=1)
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
