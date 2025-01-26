import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # Disable TensorFlow 2 behaviors


a = tf.constant(20, name="input_a")
b = tf.constant(30, name="input_b")

c = tf.add(a, b, name="add_c")
           
with tf.Session() as sess:
    print('Addition of a and b:', sess.run(c))