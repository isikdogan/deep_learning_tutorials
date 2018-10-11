import tensorflow as tf

# define the inputs
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# define the graph
g_mean = tf.sqrt(x * y)

# run the graph
with tf.Session() as sess:
    res = sess.run(g_mean, feed_dict={x: 2, y: 8})
    print(res)