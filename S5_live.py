""" Deep Learning with TensorFlow
Live coding session 5: convolutional neural networks, batchnorm, learning rate schedules, optimizers
"""

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def preprocess_data(im, label):
    im = tf.cast(im, tf.float32)
    im = im / 127.5
    im = im - 1
    # im = tf.reshape(im, [-1])
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

########################################################################
def model(input_layer, training, num_classes=10):
    with tf.variable_scope("model"):
        net = tf.expand_dims(input_layer, axis=3)

        net = tf.layers.conv2d(net, 20, (5, 5))
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2))

        net = tf.layers.conv2d(net, 50, (5, 5))
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2))

        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 500)
        net = tf.nn.relu(net) # I forgot to add this ReLU in the video
        net = tf.layers.dropout(net, rate=0.2, training=training) # I forgot the training argument in the video
        net = tf.layers.dense(net, num_classes)
    return net

def loss_functions(logits, labels, num_classes=10):
    with tf.variable_scope("loss"):
        target_prob = tf.one_hot(labels, num_classes)
        tf.losses.softmax_cross_entropy(target_prob, logits)
        total_loss = tf.losses.get_total_loss() # include regularization loss
    return total_loss

def optimizer_func_momentum(total_loss, global_step, learning_rate=0.01):
    with tf.variable_scope("optimizer"):
        lr_schedule = tf.train.exponential_decay(learning_rate=learning_rate,
                                                 global_step=global_step,
                                                 decay_steps=1875,
                                                 decay_rate=0.9,
                                                 staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr_schedule, momentum=0.9)
            optimizer = optimizer.minimize(total_loss, global_step=global_step)
    return optimizer

def optimizer_func_adam(total_loss, global_step, learning_rate=0.01):
    with tf.variable_scope("optimizer"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1)
            optimizer = optimizer.minimize(total_loss, global_step=global_step)
    return optimizer
########################################################################

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
    training = tf.placeholder(tf.bool)
    logits = model(images, training) ##############################
    loss = loss_functions(logits, labels)
    optimizer = optimizer_func_adam(loss, global_step) ##############################
    accuracy = performance_metric(logits, labels)

    # summary placeholders
    streaming_loss_p = tf.placeholder(tf.float32)
    streaming_acc_p = tf.placeholder(tf.float32)
    val_acc_p = tf.placeholder(tf.float32)
    val_summ_ops = tf.summary.scalar('validation_acc', val_acc_p)
    train_summ_ops = tf.summary.merge([
        tf.summary.scalar('streaming_loss', streaming_loss_p),
        tf.summary.scalar('streaming_accuracy', streaming_acc_p)
    ])

    # start training
    num_iter = 18750 # 10 epochs
    log_iter = 1875
    val_iter = 1875
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init_op_train)

        # logs for TensorBoard
        logdir = 'logs'
        writer = tf.summary.FileWriter(logdir, sess.graph) # visualize the graph

        # load / save checkpoints
        checkpoint_path = 'checkpoints'
        saver = tf.train.Saver(max_to_keep=None)
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)

        # resume training if a checkpoint exists
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Loaded parameters from {}".format(ckpt.model_checkpoint_path))

        initial_step = global_step.eval()

        streaming_loss = 0
        streaming_accuracy = 0

        for i in range(initial_step, num_iter + 1):
            _, loss_batch, acc_batch = sess.run([optimizer, loss, accuracy], feed_dict={training: True}) ##############################
            streaming_loss += loss_batch
            streaming_accuracy += acc_batch
            if i % log_iter == 0:
                print("Iteration: {}, Streaming loss: {:.2f}, Streaming accuracy: {:.2f}"
                        .format(i, streaming_loss/log_iter, streaming_accuracy/log_iter))

                # save to log file for TensorBoard
                summary_train = sess.run(train_summ_ops, feed_dict={streaming_loss_p: streaming_loss,
                                                                    streaming_acc_p: streaming_accuracy})
                writer.add_summary(summary_train, global_step=i)

                streaming_loss = 0
                streaming_accuracy = 0

            if i % val_iter == 0:
                saver.save(sess, os.path.join(checkpoint_path, 'checkpoint'), global_step=global_step)
                print("Model saved!")

                sess.run(init_op_val)
                validation_accuracy = 0
                num_iter = 0
                while True:
                    try:
                        acc_batch = sess.run(accuracy, feed_dict={training: False}) ##############################
                        validation_accuracy += acc_batch
                        num_iter += 1
                    except tf.errors.OutOfRangeError:
                        validation_accuracy /= num_iter
                        print("Iteration: {}, Validation accuracy: {:.2f}".format(i, validation_accuracy))

                        # save log file to TensorBoard
                        summary_val = sess.run(val_summ_ops, feed_dict={val_acc_p: validation_accuracy})
                        writer.add_summary(summary_val, global_step=i)

                        sess.run(init_op_train) # switch back to training set
                        break
        writer.close()

if __name__ == "__main__":
    train()
