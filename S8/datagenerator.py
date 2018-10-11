""" A Python iterator that loads and processes images.
This iterator will be called through TensorFlow Dataset API to feed pairs of
clean and noisy images into the model.
"""

import cv2
import numpy as np
import random
import os, glob

class DataGenerator:

    def __init__(self, image_size, base_dir = '../S6/dataset'):
		# all images will be center cropped and resized to image_size
        self.image_size = image_size

        # number of validation samples
        self.num_val = 320

        filenames = glob.glob(os.path.join(base_dir, '**', '*.jpg'), recursive=True)
        self.filenames_train = filenames[self.num_val:]
        self.filenames_val = filenames[:self.num_val]

        # dataset mode
        self.is_training = True
        self.val_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.fetch_sample()
        except IndexError:
            raise StopIteration()

    def get_tensor_shape(self):
        return (self.image_size[1], self.image_size[0], 3)

    def set_mode(self, is_training):
        self.is_training = is_training
        self.val_idx = 0

    def fetch_sample(self):
        if self.is_training:
            # pick a random image
            impath = random.choice(self.filenames_train)
        else:
            # pick the next validation sample
            impath = self.filenames_val[self.val_idx]
            self.val_idx += 1
        image_in = cv2.imread(impath)

        # resize to image_size
        image_in = self.center_crop_and_resize(image_in)

        # inject noise
        image_out = self.add_random_noise(image_in)

        return image_in, image_out

    def center_crop_and_resize(self, image):
        R, C, _ = image.shape
        if R > C:
            pad = (R - C) // 2
            image = image[pad:-pad, :]
        elif C > R:
            pad = (C - R) // 2
            image = image[:, pad:-pad]
        image = cv2.resize(image, self.image_size)
        return image

    def add_random_noise(self, image):
        noise_var = random.randrange(3, 15)
        h, w, c = image.shape
        image_out = image.copy()
        image_out = image_out.astype(np.float32)
        noise = np.random.randn(h, w, c) * noise_var
        image_out += noise
        image_out = np.minimum(np.maximum(image_out, 0), 255)
        image_out = image_out.astype(np.uint8)
        return image_out

if __name__ == '__main__':
    dg = DataGenerator(image_size=(256, 256))
    image_in, image_out = next(dg)
    cv2.imwrite('image_in.jpg', image_in)
    cv2.imwrite('image_out.jpg', image_out)
