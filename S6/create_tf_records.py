""" Converts an image dataset into TFRecords. The dataset should be organized as:

base_dir:
-- class_name1
---- image_name.jpg
...
-- class_name2
---- image_name.jpg
...
-- class_name3
---- image_name.jpg
...

Example:
$ python create_tf_records.py --input_dir ./dataset --output_dir ./tfrecords --num_shards 10 --split_ratio 0.2
"""

import tensorflow as tf
import os, glob
import argparse
import random

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _create_tfexample(image_data, label):
    example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image_data),
            'label': _int64_feature(label)
            }))
    return example

def enumerate_classes(class_list, sort=True):
    class_ids = {}
    class_id_counter = 0

    if sort:
        class_list.sort()

    for class_name in class_list:
        if class_name not in class_ids:
            class_ids[class_name] = class_id_counter
            class_id_counter += 1

    return class_ids

def create_tfrecords(save_dir, dataset_name, filenames, class_ids, num_shards):

    im_per_shard = int( len(filenames) / num_shards ) + 1

    for shard in range(num_shards):
        output_filename = os.path.join(save_dir, '{}_{:03d}-of-{:03d}.tfrecord'
                                       .format(dataset_name, shard, num_shards))
        print('Writing into {}'.format(output_filename))
        filenames_shard = filenames[shard*im_per_shard:(shard+1)*im_per_shard]

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:

            for filename in filenames_shard:
                image = tf.gfile.FastGFile(filename, 'rb').read()
                class_name = os.path.basename(os.path.dirname(filename))
                label = class_ids[class_name]

                example = _create_tfexample(image, label)
                tfrecord_writer.write(example.SerializeToString())

    print('Finished writing {} images into TFRecords'.format(len(filenames)))

def main(args):

    supported_formats = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    filenames = []
    for extension in supported_formats:
        pattern = os.path.join(args.input_dir, '**', extension)
        filenames.extend(glob.glob(pattern, recursive=False))

    random.seed(args.seed)
    random.shuffle(filenames)

    num_test = int(args.split_ratio * len(filenames))
    num_shards_test = int(args.split_ratio * args.num_shards)
    num_shards_train = args.num_shards - num_shards_test

    # write the list of classes and their corresponding ids to a file
    class_list = [name for name in os.listdir(args.input_dir)
                    if os.path.isdir(os.path.join(args.input_dir, name))]
    class_ids = enumerate_classes(class_list)
    with open(os.path.join(args.output_dir, 'classes.txt'), 'w') as f:
        for cid in class_ids:
            print('{}:{}'.format(class_ids[cid], cid), file=f)

    # create TFRecords for the training and test sets
    create_tfrecords(save_dir=args.output_dir,
                     dataset_name='train',
                     filenames=filenames[num_test:],
                     class_ids=class_ids,
                     num_shards=num_shards_train)
    create_tfrecords(save_dir=args.output_dir,
                     dataset_name='test',
                     filenames=filenames[:num_test],
                     class_ids=class_ids,
                     num_shards=num_shards_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='path to the directory where the images will be read from')
    parser.add_argument('--output_dir', type=str,
                        help='path to the directory where the TFRecords will be saved to')
    parser.add_argument('--num_shards', type=int,
                        help='total number of shards')
    parser.add_argument('--split_ratio', type=float, default=0.2,
                        help='ratio of number of images in the test set to the total number of images')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for repeatable train/test splits')
    args = parser.parse_args()
    main(args)
