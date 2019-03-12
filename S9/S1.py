import tensorflow as tf

x = 2.0
y = 8.0

@tf.function
def geometric_mean(x, y):
    g_mean = tf.sqrt(x * y)
    return g_mean

g_mean = geometric_mean(x, y)
tf.print(g_mean)