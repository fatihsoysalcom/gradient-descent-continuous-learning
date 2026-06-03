import random

def predict(x, m, b):
    """Simple linear model: y = mx + b"""
    return m * x + b

def mean_squared_error(y_true, y_pred):
    """Calculates Mean Squared Error"""
    return sum([(yt - yp)**2 for yt, yp in zip(y_true, y_pred)]) / len(y_true)

def train_linear_regression(X, y_true, learning_rate=0.01, epochs=1000):
    """
    Trains a simple linear regression model using gradient descent.
    Demonstrates adaptation and continuous learning.
    """
    m = random.uniform(-1, 1) # Initialize slope randomly
    b = random.uniform(-1, 1) # Initialize intercept randomly

    print(f"Initial parameters: m={m:.4f}, b={b:.4f}")
    initial_predictions = [predict(x, m, b) for x in X]
    initial_loss = mean_squared_error(y_true, initial_predictions)
    print(f"Initial loss: {initial_loss:.4f}\n")

    for epoch in range(epochs):
        # Make predictions with current parameters
        y_pred = [predict(x, m, b) for x in X]

        # Calculate gradients (how much m and b need to change)
        # This is the 'learning' step, adapting to the data
        dm = 0
        db = 0
        for i in range(len(X)):
            error = y_pred[i] - y_true[i]
            dm += error * X[i]
            db += error
        dm /= len(X)
        db /= len(X)

        # Update parameters using gradient descent
        # This is where the model 'adapts' based on the calculated gradients
        m -= learning_rate * dm
        b -= learning_rate * db

        if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
            current_predictions = [predict(x, m, b) for x in X]
            current_loss = mean_squared_error(y_true, current_predictions)
            print(f"Epoch {epoch+1}/{epochs}: m={m:.4f}, b={b:.4f}, Loss={current_loss:.4f}")

    print(f"\nFinal parameters after {epochs} epochs: m={m:.4f}, b={b:.4f}")
    final_predictions = [predict(x, m, b) for x in X]
    final_loss = mean_squared_error(y_true, final_predictions)
    print(f"Final loss: {final_loss:.4f}")
    return m, b

if __name__ == "__main__":
    # --- Simulate some data for demonstration ---
    # True relationship: y = 2 * x + 5
    true_m = 2
    true_b = 5
    X_data = [i for i in range(1, 11)] # x values from 1 to 10
    # Add some noise to y_true to make it more realistic
    y_true_data = [true_m * x + true_b + random.uniform(-1.5, 1.5) for x in X_data]

    print("--- Starting Linear Regression Training (Adaptation Process) ---")
    print(f"Target relationship (approximate): y = {true_m}x + {true_b}")
    print(f"Input X: {X_data}")
    print(f"True y (with noise): {[f'{y:.2f}' for y in y_true_data]}\n")

    # Train the model
    final_m, final_b = train_linear_regression(X_data, y_true_data, learning_rate=0.01, epochs=2000)

    print("\n--- Demonstration of Adaptation/Learning Complete ---")
    print(f"The model has adapted its parameters from random initial values to approximate the underlying data pattern.")
    print(f"You can see the loss decreasing over epochs, indicating the model is continuously learning and improving.")