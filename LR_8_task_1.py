import numpy as np
import tensorflow as tf

# Define hyperparameters
n_samples, batch_size, num_steps = 1000, 100, 20000
X_data = np.random.uniform(1, 10, (n_samples, 1)).astype(np.float32)  # Ensure float32 type
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))).astype(np.float32)  # Ensure float32 type

# Create the model's parameters
k = tf.Variable(tf.random.normal((1, 1), dtype=tf.float32), name='slope')
b = tf.Variable(tf.zeros((1,), dtype=tf.float32), name='bias')

# Define the model
def model(X):
    return tf.matmul(X, k) + b

# Define the loss function
def loss_fn(X, y):
    y_pred = model(X)
    return tf.reduce_sum((y - y_pred) ** 2)

# Define the optimizer (lower the learning rate)
optimizer = tf.optimizers.SGD(learning_rate=0.001)  # Reduced learning rate

# Training loop
display_step = 100
for i in range(num_steps):
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]

    with tf.GradientTape() as tape:
        loss_val = loss_fn(X_batch, y_batch)

    # Check for NaN in loss value
    if tf.reduce_any(tf.math.is_nan(loss_val)):
        print(f"NaN in loss at iteration {i+1}")
        break

    gradients = tape.gradient(loss_val, [k, b])

    # Check for NaN in gradients
    if any(tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients):
        print(f"NaN in gradients at iteration {i+1}")
        break

    # Clip gradients to prevent explosion
    gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]

    optimizer.apply_gradients(zip(gradients, [k, b]))

    if (i+1) % display_step == 0:
        print(f"Iteration {i+1}: Loss={loss_val.numpy():.8f}, k={k.numpy()[0][0]:.4f}, b={b.numpy()[0]:.4f}")
