""" Coding Session 8: using a python iterator as a data generator and training a denoising autoencoder
"""

import tensorflow as tf
import numpy as np
import os, glob
import argparse
from nets.model import model
from datagenerator import DataGenerator #########################

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

class TFModelTrainer:

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

        # set training parameters
        self.learning_rate = 0.01
        self.num_iter = 100000
        self.save_iter = 1000
        self.val_iter = 1000
        self.log_iter = 100
        self.batch_size = 16

        # set up data layer
        self.image_size = (224, 224)
        self.data_generator = DataGenerator(self.image_size)

    def preprocess_image(self, image):
        # normalize image to [-1, +1]
        image = tf.cast(image, tf.float32)
        image = image / 127.5
        image = image - 1
        return image

    def _preprocess_images(self, image_orig, image_noisy):
        image_orig = self.preprocess_image(image_orig)
        image_noisy = self.preprocess_image(image_noisy)
        return image_orig, image_noisy

    def _data_layer(self, num_threads=8, prefetch_buffer=100):
        with tf.variable_scope('data'):
            data_shape = self.data_generator.get_tensor_shape() #########################
            dataset = tf.data.Dataset.from_generator(lambda: self.data_generator,
                                                     (tf.float32, tf.float32),
                                                     (tf.TensorShape(data_shape),
                                                      tf.TensorShape(data_shape)))
            dataset = dataset.map(self._preprocess_images, num_parallel_calls=num_threads)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(prefetch_buffer)
            iterator = dataset.make_initializable_iterator()
        return iterator

    def _loss_functions(self, preds, ground_truth):
        with tf.name_scope('loss'):
            tf.losses.mean_squared_error(ground_truth, preds) #########################
            total_loss = tf.losses.get_total_loss()
        return total_loss

    def _optimizer(self, loss, global_step):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1)
            optimizer = optimizer.minimize(loss, global_step=global_step)
        return optimizer

    def train(self):
        # iteration number
        global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name='iter_number')

        # training graph
        iterator = self._data_layer()
        image_orig, image_noisy = iterator.get_next()
        training = tf.placeholder(tf.bool, name='is_training')
        logits = model(image_noisy, training=training)
        loss = self._loss_functions(logits, image_orig)
        optimizer = self._optimizer(loss, global_step)

        # summary placeholders
        streaming_loss_p = tf.placeholder(tf.float32)
        validation_loss_p = tf.placeholder(tf.float32)
        summ_op_train = tf.summary.scalar('streaming_loss', streaming_loss_p)
        summ_op_test = tf.summary.scalar('validation_loss', validation_loss_p)

        # don't allocate entire gpu memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)

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
                    self.data_generator.set_mode(is_training=False)
                    sess.run(iterator.initializer)

                    validation_loss = 0
                    for j in range(self.data_generator.num_val // self.batch_size):
                        loss_batch = sess.run(loss, feed_dict={training: False})
                        validation_loss += loss_batch
                    validation_loss /= j

                    print("Validation loss: {}".format(validation_loss))

                    summary_test = sess.run(summ_op_test, feed_dict={validation_loss_p: validation_loss})
                    writer.add_summary(summary_test, global_step=i)

                    self.data_generator.set_mode(is_training=True)
                    sess.run(iterator.initializer)

            writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/',
                        help="Path to the dir where the checkpoints are saved")
    args = parser.parse_args()
    trainer = TFModelTrainer(args.checkpoint_path)
    trainer.train()

if __name__ == '__main__':
    main()
