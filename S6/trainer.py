""" Trains a TensorFlow model
Example:
$ python trainer.py --checkpoint_path ./checkpoints --data_path ./tfrecords
"""

import tensorflow as tf
import numpy as np
import os, glob
import argparse
from nets import mobilenet_v1 #####################################################

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

class TFModelTrainer:

    def __init__(self, checkpoint_path, data_path):
        self.checkpoint_path = checkpoint_path

        # set training parameters #################################################
        self.learning_rate = 0.01
        self.num_iter = 100000
        self.save_iter = 5000
        self.val_iter = 5000
        self.log_iter = 100
        self.batch_size = 32

        # set up data layer
        self.training_filenames = glob.glob(os.path.join(data_path, 'train_*.tfrecord'))
        self.validation_filenames = glob.glob(os.path.join(data_path, 'test_*.tfrecord'))
        self.iterator, self.filenames = self._data_layer()
        self.num_val_samples = 10000
        self.num_classes = 2
        self.image_size = 224

    def preprocess_image(self, image_string):
        image = tf.image.decode_jpeg(image_string, channels=3)

        # flip for data augmentation
        image = tf.image.random_flip_left_right(image) ############################

        # normalize image to [-1, +1]
        image = tf.cast(image, tf.float32)
        image = image / 127.5
        image = image - 1
        return image

    def _parse_tfrecord(self, example_proto): #####################################
        keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                            'label': tf.FixedLenFeature([], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        image = parsed_features['image']
        label = parsed_features['label']
        image = self.preprocess_image(image)
        return image, label

    def _data_layer(self, num_threads=8, prefetch_buffer=100):
        with tf.variable_scope('data'):
            filenames = tf.placeholder(tf.string, shape=[None])
            dataset = tf.data.TFRecordDataset(filenames) ##########################
            dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=num_threads)
            dataset = dataset.repeat()
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(prefetch_buffer)
            iterator = dataset.make_initializable_iterator()
        return iterator, filenames

    def _loss_functions(self, logits, labels):
        with tf.variable_scope('loss'):
            target_prob = tf.one_hot(labels, self.num_classes)
            tf.losses.softmax_cross_entropy(target_prob, logits)
            total_loss = tf.losses.get_total_loss() #include regularization loss
        return total_loss

    def _optimizer(self, total_loss, global_step):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1)
            optimizer = optimizer.minimize(total_loss, global_step=global_step)
        return optimizer

    def _performance_metric(self, logits, labels):
        with tf.variable_scope("performance_metric"):
            preds = tf.argmax(logits, axis=1)
            labels = tf.cast(labels, tf.int64)
            corrects = tf.equal(preds, labels)
            accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
        return accuracy

    def train(self):
        # iteration number
        global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name='iter_number')

        # training graph
        images, labels = self.iterator.get_next()
        images = tf.image.resize_bilinear(images, (self.image_size, self.image_size))
        training = tf.placeholder(tf.bool, name='is_training')
        logits, _ = mobilenet_v1.mobilenet_v1(images,
                     num_classes=self.num_classes,
                     is_training=training,
                     scope='MobilenetV1',
                     global_pool=True) ############################################
        loss = self._loss_functions(logits, labels)
        optimizer = self._optimizer(loss, global_step)
        accuracy = self._performance_metric(logits, labels)

        # summary placeholders
        streaming_loss_p = tf.placeholder(tf.float32)
        accuracy_p = tf.placeholder(tf.float32)
        summ_op_train = tf.summary.scalar('streaming_loss', streaming_loss_p)
        summ_op_test = tf.summary.scalar('accuracy', accuracy_p)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.iterator.initializer, feed_dict={self.filenames: self.training_filenames})

            writer = tf.summary.FileWriter(self.checkpoint_path, sess.graph)

            saver = tf.train.Saver(max_to_keep=None) # keep all checkpoints
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)

            # resume training if a checkpoint exists
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loaded parameters from {}'.format(ckpt.model_checkpoint_path))

            initial_step = global_step.eval()

            # train the model
            streaming_loss = 0
            for i in range(initial_step, self.num_iter + 1):
                _, loss_batch = sess.run([optimizer, loss], feed_dict={training: True})

                if not np.isfinite(loss_batch):
                    print('loss diverged, stopping')
                    exit()

                # log summary
                streaming_loss += loss_batch
                if i % self.log_iter == self.log_iter - 1:
                    streaming_loss /= self.log_iter
                    print(i + 1, streaming_loss)
                    summary_train = sess.run(summ_op_train, feed_dict={streaming_loss_p: streaming_loss})
                    writer.add_summary(summary_train, global_step=i)
                    streaming_loss = 0

                # save model
                if i % self.save_iter == self.save_iter - 1:
                    saver.save(sess, os.path.join(self.checkpoint_path, 'checkpoint'), global_step=global_step)
                    print("Model saved!")

                # run validation
                if i % self.val_iter == self.val_iter - 1:
                    print("Running validation.")
                    sess.run(self.iterator.initializer, feed_dict={self.filenames: self.validation_filenames})

                    validation_accuracy = 0
                    for j in range(self.num_val_samples // self.batch_size): ###################################
                        acc_batch = sess.run(accuracy, feed_dict={training: False})
                        validation_accuracy += acc_batch
                    validation_accuracy /= j

                    print("Accuracy: {}".format(validation_accuracy))

                    summary_test = sess.run(summ_op_test, feed_dict={accuracy_p: validation_accuracy})
                    writer.add_summary(summary_test, global_step=i)

                    sess.run(self.iterator.initializer, feed_dict={self.filenames: self.training_filenames})

            writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/',
                        help="Path to the dir where the checkpoints are saved")
    parser.add_argument('--data_path', type=str, default='./tfrecords/', help="Path to the TFRecords")
    args = parser.parse_args()
    trainer = TFModelTrainer(args.checkpoint_path, args.data_path)
    trainer.train()

if __name__ == '__main__':
    main()
