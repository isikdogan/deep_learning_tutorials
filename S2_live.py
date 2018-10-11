""" Deep Learning with TensorFlow
Coding session 2: Training a Multilayer Perceptron

Let's train a simple neural network that classifies handwritten digits using the MNIST dataset.
Video will be uploaded later.
"""

import tensorflow as tf

def preprocess_data(im, label):
    im = tf.cast(im, tf.float32)
    im = im / 127.5
    im = im - 1
    im = tf.reshape(im, [-1])
    return im, label

def data_layer(data_tensor, num_threads=8, prefetch_buffer=100, batch_size=32):
    with tf.variable_scope("data"):
        dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
        dataset = dataset.shuffle(buffer_size=60000).repeat()
        dataset = dataset.map(preprocess_data, num_parallel_calls=num_threads)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_buffer)
        iterator = dataset.make_one_shot_iterator()
    return iterator

def model(input_layer, num_classes=10):
    with tf.variable_scope("model"):
        net = tf.layers.dense(input_layer, 512)
        net = tf.nn.relu(net)
        net = tf.layers.dense(net, num_classes)
    return net

def loss_functions(logits, labels, num_classes=10):
    with tf.variable_scope("loss"):
        target_prob = tf.one_hot(labels, num_classes)
        total_loss = tf.losses.softmax_cross_entropy(target_prob, logits)
    return total_loss

def optimizer_func(total_loss, global_step, learning_rate=0.1):
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = optimizer.minimize(total_loss, global_step=global_step)
    return optimizer

def performance_metric(logits, labels):
    with tf.variable_scope("performance_metric"):
        preds = tf.argmax(logits, axis=1)
        labels = tf.cast(labels, tf.int64)
        corrects = tf.equal(preds, labels)
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy

def train(data_tensor):
    global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name="iter_number")

    # training graph
    images, labels = data_layer(data_tensor).get_next()
    logits = model(images)
    loss = loss_functions(logits, labels)
    optimizer = optimizer_func(loss, global_step)
    accuracy = performance_metric(logits, labels)

    # start training
    num_iter = 10000
    log_iter = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        streaming_loss = 0
        streaming_accuracy = 0

        for i in range(1, num_iter + 1):
            _, loss_batch, acc_batch = sess.run([optimizer, loss, accuracy])
            streaming_loss += loss_batch
            streaming_accuracy += acc_batch
            if i % log_iter == 0:
                print("Iteration: {}, Streaming loss: {:.2f}, Streaming accuracy: {:.2f}"
                        .format(i, streaming_loss/log_iter, streaming_accuracy/log_iter))
                streaming_loss = 0
                streaming_accuracy = 0

if __name__ == "__main__":
    # It's very easy to load the MNIST dataset through the Keras module.
    # Keras is a high-level neural network API that has become a part of TensorFlow since version 1.2.
    # Therefore, we don't need to install Keras separately.
    # In the upcoming lectures we will also see how to load and preprocess custom data.
    data_train, data_val = tf.keras.datasets.mnist.load_data()

    # The training set has 60,000 samples where each sample is a 28x28 grayscale image.
    # Each one of these samples have a single label Similarly the validation set has 10,000 images and corresponding labels.
    # We can verify this by printing the shapes of the loaded tensors
    print(data_train[0].shape, data_train[1].shape, data_val[0].shape, data_val[1].shape)

    # Let the training begin!
    train(data_tensor=data_train)

    # Even after very few epochs, we got a model that can classify the handwritten digits in the training set
    # with 98% accuracy. So far we haven't used the validation set at all.
    # You might wonder why we need a separate validation set in the first place.
    # The answer is to make sure that the model generalizes well to unseen data to have an idea of the actual performance of the model.
    # We will talk about that in the next session.