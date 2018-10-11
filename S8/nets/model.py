""" A simple encoder-decoder network with skip connections.
"""

import tensorflow as tf

def model(input_layer, training, weight_decay=0.00001):

    reg = tf.contrib.layers.l2_regularizer(weight_decay)

    def conv_block(inputs, num_filters, name):
        with tf.variable_scope(name):
            net = tf.layers.separable_conv2d(
                        inputs=inputs,
                        filters=num_filters,
                        kernel_size=(3,3),
                        padding='SAME',
                        use_bias=False,
                        activation=None,
                        pointwise_regularizer=reg,
                        depthwise_regularizer=reg)
            net = tf.layers.batch_normalization(
                        inputs=net,
                        training=training)
            net = tf.nn.relu(net)
        return net

    def pointwise_block(inputs, num_filters, name):
        with tf.variable_scope(name):
            net = tf.layers.conv2d(
                        inputs=inputs,
                        filters=num_filters,
                        kernel_size=1,
                        use_bias=False,
                        activation=None,
                        kernel_regularizer=reg)
            net = tf.layers.batch_normalization(
                        inputs=net,
                        training=training)
            net = tf.nn.relu(net)
        return net

    def pooling(inputs, name):
        with tf.variable_scope(name):
            net = tf.layers.max_pooling2d(inputs,
                                    pool_size=(2,2),
                                    strides=(2,2))
        return net

    def downsampling(inputs, name):
        with tf.variable_scope(name):
            net = tf.layers.average_pooling2d(inputs,
                                    pool_size=(2,2),
                                    strides=(2,2))
        return net

    def upsampling(inputs, name):
        with tf.variable_scope(name):
            dims = tf.shape(inputs)
            new_size = [dims[1]*2, dims[2]*2]
            net = tf.image.resize_bilinear(inputs, new_size)
        return net

    def output_block(inputs, name):
        with tf.variable_scope(name):
            net = tf.layers.conv2d(
                        inputs=inputs,
                        filters=3,
                        kernel_size=(1,1),
                        activation=None)
        return net

    def subnet_module(inputs, name, num_filters, num_layers=3):
        with tf.variable_scope(name):
            for i in range(num_layers-1):
                net = conv_block(inputs, num_filters=num_filters, name='{}_conv{}'.format(name, i))
                inputs = tf.concat([net, inputs], axis=3)
            net = conv_block(inputs, num_filters=num_filters, name='{}_conv3'.format(name))
        return net

    num_filters = 16
    net = input_layer
    skip_connections = []
    # encoder
    with tf.variable_scope('encoder'):
        for i in range(4):
            net = subnet_module(net, num_filters=num_filters, name='conv_e{}'.format(i))
            skip_connections.append(net)
            net = pooling(net, name='pool{}'.format(i))
            num_filters *= 2

    # bottleneck
    net = subnet_module(net, num_filters=num_filters, name='conv_bottleneck'.format(i))

    # decoder
    with tf.variable_scope('decoder'):
        for i in range(4):
            num_filters /= 2
            net = upsampling(net, name='upsample{}'.format(i))
            net = tf.concat([net, skip_connections.pop()], axis=3)
            net = subnet_module(net, num_filters=num_filters, name='subnet_d{}'.format(i))

    # exit flow
    with tf.variable_scope('exit_flow'):
        logits = output_block(net, name='output_block')

    return logits
