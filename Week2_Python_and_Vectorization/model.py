
import numpy as np
import matplotlib.pyplot as plt

# 1. Create training data
X = np.array([
    [3, 8, 10, 1, 0, 1],
    [2, 6, 5, 0, 1, 1],
    [5, 7, 15, 1, 0, 1],
    [1, 5, 2, 0, 1, 0],
    [4, 9, 12, 1, 0, 1],
    [2, 6, 4, 0, 1, 1],
    [6, 8, 18, 1, 0, 1],
    [1, 4, 1, 0, 1, 0],
    [4, 7, 10, 1, 0, 1],
    [3, 6, 7, 1, 1, 1]
])

# 2. Create labels
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1]).reshape(-1, 1)

# 3. Standardize the features (mean = 0, std = 1)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 4. Initialize weights and bias
m, n = X.shape
w = np.zeros((n, 1))
b = 0.0

# 5. Define sigmoid activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 6. Training settings
learning_rate = 0.001
epochs = 1000
losses = []

# 7. Training loop
print(f"Loop started with {epochs} epochs")

for i in range(epochs):
    # Forward propagation
    z = np.dot(X, w) + b
    prediction = sigmoid(z)

    # Compute binary cross-entropy loss (correct version)
    loss = -np.mean(y * np.log(prediction + 1e-8) + (1 - y) * np.log(1 - prediction + 1e-8))
    losses.append(loss)

    # Compute gradients
    dw = np.dot(X.T, (prediction - y)) / m
    db = np.sum(prediction - y) / m

    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db

    # Optional: print every 200 steps
    if i % 200 == 0:
        print(f"Epochs: {i}, Loss: {loss:.4f}")

# 8. Make final predictions
pred_labels = (sigmoid(np.dot(X, w) + b) > 0.5).astype(int)

# 9. Accuracy check
accuracy = np.mean(pred_labels == y)
print(f"Final Accuracy: {accuracy:.2f}")

# 10. Plot loss curve
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.show()
