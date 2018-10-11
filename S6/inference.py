""" Runs inference given a frozen model and a set of images
Example:
$ python inference.py --frozen_model frozen_model.pb --input_path ./test_images
"""

import argparse
import tensorflow as tf
import os, glob
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

class InferenceEngine:
    def __init__(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="Pretrained")

        self.graph = graph

    def run_inference(self, input_path):
        if os.path.isdir(input_path):
            filenames = glob.glob(os.path.join(input_path, '*.jpg'))
            filenames.extend(glob.glob(os.path.join(input_path, '*.jpeg')))
            filenames.extend(glob.glob(os.path.join(input_path, '*.png')))
            filenames.extend(glob.glob(os.path.join(input_path, '*.bmp')))
        else:
            filenames = [input_path]

        input_layer = self.graph.get_tensor_by_name('Pretrained/input:0')
        preds = self.graph.get_tensor_by_name('Pretrained/preds:0')
        pred_idx = tf.argmax(preds)

        with tf.Session(graph=self.graph) as sess:
            for filename in filenames:
                image = cv2.imread(filename)
                class_label, probs = sess.run([pred_idx, preds], feed_dict={input_layer: image})
                print("Label: {:d}, Probability: {:.2f} \t File: {}".format(class_label, probs[class_label], filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model", default="frozen_model.pb", type=str, help="Path to the frozen model file to import")
    parser.add_argument("--input_path", type=str, help="Path to the input file(s). If this is a dir all files will be processed.")
    args = parser.parse_args()

    ie = InferenceEngine(args.frozen_model)
    ie.run_inference(args.input_path)

