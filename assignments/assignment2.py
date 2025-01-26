import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # Disable TensorFlow 2 behaviors
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
tf.set_random_seed(42)

# Generate normally distributed data for input features and target output
num_samples = 100
input_features = np.random.normal(0, 1, (num_samples, 1))
target_output = 3 * input_features + np.random.normal(0, 0.5, (num_samples, 1))

# Define placeholders for input features and target output
X = tf.placeholder(tf.float32, shape=[None, 1], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

# Define variables for weights and bias
W = tf.Variable(tf.random_normal([1, 1]), name='weights')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Define the model
predictions = tf.add(tf.matmul(X, W), b)

# Define the Mean Squared Error (MSE) loss function
loss = tf.reduce_mean(tf.square(predictions - Y))

# Define the Gradient Descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

# Train the model
num_epochs = 1000
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(num_epochs):
        _, current_loss = sess.run([optimizer, loss], feed_dict={X: input_features, Y: target_output})
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {current_loss}')
    
    # Print the trained weights and bias
    trained_weights, trained_bias = sess.run([W, b])
    print(f'Trained weights: {trained_weights}')
    print(f'Trained bias: {trained_bias}')