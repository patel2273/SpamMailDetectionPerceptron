import numpy as np

# 🔹 Step 1: Dataset (Email Spam Example)
# x1 = "Free", x2 = "Win", x3 = "Links"
X = np.array([
    [1, 1, 1],   # spam
    [0, 0, 0],   # not spam
    [1, 0, 1],   # spam
    [0, 1, 0]    # spam
])

y = np.array([1, 0, 1, 1])  # Labels

# 🔹 Step 2: Initialize weights and bias
weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.1
epochs = 10

# 🔹 Activation function (Step Function)
def activation(z):
    return 1 if z >= 0 else 0

# 🔹 Step 3: Training Loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    
    for i in range(len(X)):
        # Prediction
        z = np.dot(X[i], weights) + bias
        y_pred = activation(z)

        # Error
        error = y[i] - y_pred

        # Update weights & bias
        weights = weights + learning_rate * error * X[i]
        bias = bias + learning_rate * error

        # Print step-by-step
        print(f"Input: {X[i]}, Predicted: {y_pred}, Actual: {y[i]}, Error: {error}")
        print(f"Updated Weights: {weights}, Bias: {bias}")

# 🔹 Final Output
print("\nFinal Weights:", weights)
print("Final Bias:", bias)
