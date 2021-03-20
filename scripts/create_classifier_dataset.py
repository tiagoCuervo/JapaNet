import pandas
import os
import numpy as np
import pathlib
import IPython.display as display
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import pickle 
import time

data_directory = pathlib.Path("./data/train_char")

# ALL_CLASSES = np.array([item.name for item in data_directory.glob(
#     '*') )

# np.save("ALL_CLASSES.npy", ALL_CLASSES, allow_pickle = True)

# with open('./data/CLASSES', 'rb') as fp:
#     CLASSES = pickle.load(fp)

#tf.data.Dataset object 
list_dataset = tf.data.Dataset.list_files(str(data_directory/'*/*'))


class DataSetCreator(object):
    def __init__(self, batch_size, image_height, image_width, dataset):
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.dataset = dataset


    def _get_class(self, path):
        path_splited = tf.strings.split(path, os.path.sep)
        print(path_splited[-2])
        return  path_splited[-2]


    def _load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # ADD PREPROCESSING HERE
        return tf.image.resize(image, [self.image_height, self.image_width])

    def _load_labeled_data(self, path):
        label = self._get_class(path)
        image = self._load_image(path)
        return image, label

    def load_process(self, shuffle_size=1000):
        self.loaded_dataset = self.dataset.map(
            self._load_labeled_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.loaded_dataset = self.loaded_dataset.cache()

        # Shuffle data and create batches
        self.loaded_dataset = self.loaded_dataset.shuffle(
            buffer_size=shuffle_size)
        self.loaded_dataset = self.loaded_dataset.repeat()
        self.loaded_dataset = self.loaded_dataset.batch(self.batch_size)

        # Make dataset fetch batches in the background during the training of the model.
        self.loaded_dataset = self.loaded_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

    def get_batch(self):
        return next(iter(self.loaded_dataset))

# # How to load data
# dataProcessor = DataSetCreator(32, 64, 64, list_dataset)
# dataProcessor.load_process()

# image_batch, label_batch = dataProcessor.get_batch()

# for img in image_batch:
#     plt.imshow(img)
#     time.sleep(3)
