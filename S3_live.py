""" Deep Learning with TensorFlow
Coding session 3: Setting up the training and validation pipeline

In the previous session we trained a model without keeping track of how it's
doing on a validation set. Let's pick up where we left off and modify our code
from the previous session to keep track of validation accuracy while training.
"""

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def preprocess_data(im, label):
    im = tf.cast(im, tf.float32)
    im = im / 127.5
    im = im - 1
    im = tf.reshape(im, [-1])
    return im, label

# We will be using the same data pipeline for both training and validation sets
# So let's create a helper function for that
def create_dataset_pipeline(data_tensor, is_train=True, num_threads=8, prefetch_buffer=100, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
    if is_train:
        dataset = dataset.shuffle(buffer_size=60000).repeat()
    dataset = dataset.map(preprocess_data, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset

def data_layer():
    with tf.variable_scope("data"):
        data_train, data_val = tf.keras.datasets.mnist.load_data()
        dataset_train = create_dataset_pipeline(data_train, is_train=True)
        dataset_val = create_dataset_pipeline(data_val, is_train=False, batch_size=1)
        iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
        init_op_train = iterator.make_initializer(dataset_train)
        init_op_val = iterator.make_initializer(dataset_val)
    return iterator, init_op_train, init_op_val

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

def train():
    global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name="iter_number")

    # define the training graph
    iterator, init_op_train, init_op_val = data_layer()
    images, labels = iterator.get_next()
    logits = model(images)
    loss = loss_functions(logits, labels)
    optimizer = optimizer_func(loss, global_step)
    accuracy = performance_metric(logits, labels)

    # start training
    num_iter = 18750 # 10 epochs
    log_iter = 1875
    val_iter = 1875
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init_op_train)

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

            if i % val_iter == 0:
                sess.run(init_op_val)
                validation_accuracy = 0
                num_iter = 0
                while True:
                    try:
                        acc_batch = sess.run(accuracy)
                        validation_accuracy += acc_batch
                        num_iter += 1
                    except tf.errors.OutOfRangeError:
                        validation_accuracy /= num_iter
                        print("Iteration: {}, Validation accuracy: {:.2f}".format(i, validation_accuracy))
                        sess.run(init_op_train) # switch back to training set
                        break

if __name__ == "__main__":
    train()
