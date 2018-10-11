""" Freezes a checkpoint, outputs a single pbfile that encapsulates both the graph and weights
Example:
$ python freeze_model.py --checkpoint_path ./checkpoints
"""

import tensorflow as tf
import argparse
from nets import mobilenet_v1
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def freeze_graph(checkpoint_path, output_node_name, outfile):
    input_layer = tf.placeholder(tf.uint8, shape=[None, None, 3], name='input')
    with tf.variable_scope('input_scaling'):
        image = tf.expand_dims(input_layer, axis=0)
        image = tf.image.resize_bilinear(image, [224, 224])
        image = tf.cast(image, tf.float32)
        image = image / 127.5
        image = image - 1

    logits, _ = mobilenet_v1.mobilenet_v1(image, num_classes=2, is_training=False)
    preds = tf.squeeze(tf.nn.softmax(logits), name='preds')

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), [output_node_name])

        with tf.gfile.GFile(outfile, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

        # print a list of ops
        for op in output_graph_def.node:
            print(op.name)

        print('Saved frozen model to {}'.format(outfile))
        print('{:d} ops in the final graph.'.format(len(output_graph_def.node)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./', help="Path to the dir where the checkpoints are saved")
    parser.add_argument('--output_node_name', type=str, default='preds', help="Name of the output node")
    parser.add_argument('--outfile', type=str, default='frozen_model.pb', help="Frozen model path")
    args = parser.parse_args()
    freeze_graph(args.checkpoint_path, args.output_node_name, args.outfile)