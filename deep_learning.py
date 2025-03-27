import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml

# Problem 1: Load MNIST dataset
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)
    
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))
    
    return train_test_split(X, y_onehot, test_size=0.2, random_state=42)

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
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    @staticmethod
    def backward(predictions, labels):
        return predictions - labels

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
        self.z = np.dot(self.weights, x) + self.biases
        self.a = self.activation.forward(self.z)
        return self.a
    
    def backward(self, dA):
        dZ = dA * self.activation.backward(self.z)
        dW = np.dot(dZ, self.input.T)
        dB = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.weights.T, dZ)
        
        self.weights, self.biases = self.optimizer.update(self.weights, self.biases, dW, dB)
        return dA_prev

# Problem 6: Initializers
class XavierInitializer:
    def W(self, n_nodes1, n_nodes2):
        return np.random.randn(n_nodes1, n_nodes2) * np.sqrt(1.0 / n_nodes1)
    
    def B(self, n_nodes2):
        return np.zeros((n_nodes2, 1))

# Problem 7: Optimizers
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, weights, biases, dW, dB):
        weights -= self.learning_rate * dW
        biases -= self.learning_rate * dB
        return weights, biases

# Problem 8: Deep Neural Network Classifier
class ScratchDeepNeuralNetworkClassifier:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, loss_gradient):
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)
    
    def train(self, x_train, y_train, epochs=100):
        for epoch in range(epochs):
            predictions = self.forward(x_train)
            loss_gradient = Softmax.backward(predictions, y_train)
            self.backward(loss_gradient)
            if epoch % 10 == 0:
                loss = -np.sum(y_train * np.log(predictions + 1e-7)) / y_train.shape[1]
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, x):
        return np.argmax(self.forward(x), axis=0)

# Problem 9: Learning and Estimation
# Load MNIST data
X_train, X_test, Y_train, Y_test = load_mnist()

# Define network architecture
layers = [
    FullyConnectedLayer(784, 128, XavierInitializer(), SGD(0.01), ReLU()),
    FullyConnectedLayer(128, 64, XavierInitializer(), SGD(0.01), ReLU()),
    FullyConnectedLayer(64, 10, XavierInitializer(), SGD(0.01), Softmax())
]

# Train and evaluate model
model = ScratchDeepNeuralNetworkClassifier(layers)
model.train(X_train, Y_train, epochs=100)

# Predict and compute accuracy
y_pred = model.predict(X_test)
y_true = np.argmax(Y_test, axis=0)
accuracy = np.mean(y_pred == y_true)
print(f"Test Accuracy: {accuracy:.4f}")
