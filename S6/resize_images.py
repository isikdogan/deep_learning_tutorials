""" A utility script that resizes all images in a given directory to a specified size
WARNING: the original images will be overwritten!
"""

import cv2
import os, glob
import argparse

def main(args):
    supported_formats = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    filenames = []
    for extension in supported_formats:
        pattern = os.path.join(args.input_dir, '**', extension)
        filenames.extend(glob.glob(pattern, recursive=True))

    num_images = len(filenames)
    for i in range(num_images):
        if i % 100 == 0:
            print("{} of {} \t Resizing: {}".format(i, num_images, filenames[i]))
        image = cv2.imread(filenames[i])
        image = cv2.resize(image, (args.resize, args.resize), interpolation=cv2.INTER_AREA)
        cv2.imwrite(filenames[i], image)

if __name__ == '__main__':
    # resizes images in-place
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='path to the directory where the images will be read from')
    parser.add_argument('--resize', type=int,
                        help='the images will be resized to NxN')

    args = parser.parse_args()
    main(args)
