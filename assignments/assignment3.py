""" 
AI Use Casses - Lecture 6 - Assignment 3
Implemented by Juan Pablo Méndez Nogales - 3121979
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorboard import program
tf.disable_v2_behavior()

# 1. Define placeholder for input array
with tf.name_scope('Input_placeholder'):
    input_placeholder = tf.placeholder(tf.float32, shape=[None], name='input_placeholder')

# 2. Set up middle section nodes
with tf.name_scope('Middle_section'):
    b = tf.reduce_prod(input_placeholder, name='prod')
    d = tf.reduce_sum(input_placeholder, name='sum')
    c = tf.reduce_mean(input_placeholder, name='mean')
    e = tf.add(b, c, name='add')

# 3.  Defining final node in separate scope
with tf.name_scope('Final_node'):
    f = tf.multiply(e, d, name='mul')


# 4. Generating random input data with mean = 1 and std = 2
A = np.random.normal(1, 2, 100)

# Running the session
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # 5. Compute final_node by feeding the array A into the placeholder
    result = sess.run(f, feed_dict={input_placeholder: A})

    # 6. Saving the graph as event files
    writer = tf.summary.FileWriter('./graph', sess.graph)
    writer.close()


# 7. Launch a TensorBoard instance to access it through the browser at http://localhost:6006/
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', './graph'])
url = tb.launch()
print(f"TensorBoard at {url}")


# 8. Creating the scatter plot for A (Index, Value)
plt.scatter(range(len(A)), A)
plt.title('Input Array A')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
