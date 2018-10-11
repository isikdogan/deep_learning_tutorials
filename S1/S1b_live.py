import tensorflow as tf

# define the inputs
x = tf.placeholder(tf.float32)

with tf.variable_scope("linear_model", reuse=tf.AUTO_REUSE):
    w = tf.get_variable("weight", dtype=tf.float32, initializer=tf.constant(0.1))
    c = tf.get_variable("bias", dtype=tf.float32, initializer=tf.constant(0.0))
    model = x * w + c

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(model, feed_dict={x: 2.0}))