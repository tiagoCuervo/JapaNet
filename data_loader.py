import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import importlib
from tqdm.auto import tqdm
import os
from pathlib import Path


def _addIdentificationLabels(df):
    for i in range(len(df)):
        pageData = np.array(df.loc[i, "labels"].split(" ")).reshape(-1, 5)
        # We drop character label, we won't need it for detection
        pageData = pageData[:, 1:].astype('uint32')
        pageData[:, 0] += pageData[:, 2] // 2  # Center on X
        pageData[:, 1] += pageData[:, 3] // 2  # Center on Y
        df.loc[i, "labels"] = pageData[:, :2].ravel()
    return df


def _addClassificationLabels(imgs_folder_path):
    pair_img_label = []
    for CLASS in os.listdir(imgs_folder_path):
        if CLASS =='train_char.zip':
            continue
        else:
            class_dir = os.path.join(folders_dir, CLASS)
            for img_id in os.listdir(class_dir):
                pair_img_label.append([img_id.split('.')[0], CLASS])
    
    df = pd.DataFrame(pair_img_label, columns=['image_id','label'])
    return df


def _bytesFeature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floatFeature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class IdentifierDataset:
    def __init__(self, config):
        self.config = config
        self.recordPath = 'train/train.tfrecord'
        self.writer = tf.io.TFRecordWriter(self.recordPath)
        self.feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.VarLenFeature(dtype=tf.float32)
        }

    def _write(self, image, label):
        feature = {
            'image': _bytesFeature(image),
            'labels': _floatFeature(label.astype(np.float32))
        }
        sample = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(sample.SerializeToString())

    def _processExample(self, example):
        pmap = tf.io.parse_single_example(example, self.feature_description)
        imageDecoded = tf.image.decode_jpeg(pmap['image'], channels=3) / 255
        imageResized = tf.image.resize(imageDecoded, [self.config.identifierInputWidth,
                                                      self.config.identifierInputHeight])
        # label = tf.sparse.to_dense(pmap['labels'])
        # return imageResized, label
        return imageResized

    def _read(self):
        record = tf.data.TFRecordDataset(self.recordPath)
        trainData = record.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.shuffle(buffer_size=100)
        trainData = trainData.batch(self.config.batchSize, drop_remainder=True)
        trainData = trainData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return trainData

    def createDataset(self):
        dfTrain = pd.read_csv('data/train.csv')
        dfTrain = _addIdentificationLabels(dfTrain)
        for i in tqdm(range(len(dfTrain))):
            sample = dfTrain.iloc[i]
            self._write(open("data/train/" + sample['image_id'] + ".jpg", 'rb').read(), sample['labels'])
        self.writer.flush()
        self.writer.close()


class ClassifierDataset:
    def __init__(self, config):
        self.config = config
        self.recordPath = 'train/train_classifier.tfrecord'
        self.writer = tf.io.TFRecordWriter(self.recordPath)
        self.feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.VarLenFeature(dtype=tf.string)
        }

    def _write(self, image, label):
        feature = {
            'image': _bytesFeature(image),
            'label': _bytresFeature(label)
        }
        sample = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(sample.SerializeToString())

    def _processExample(self, example):
        pmap = tf.io.parse_single_example(example, self.feature_description)
        imageDecoded = tf.image.decode_jpeg(pmap['image'], channels=3) / 255
        imageResized = tf.image.resize(imageDecoded, [self.config.identifierInputWidth,
                                                      self.config.identifierInputHeight])
        # label = tf.sparse.to_dense(pmap['labels'])
        # return imageResized, label
        return imageResized

    def _read(self):
        record = tf.data.TFRecordDataset(self.recordPath)
        trainData = record.map(
            self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.shuffle(buffer_size=100)
        trainData = trainData.batch(self.config.batchSize, drop_remainder=True)
        trainData = trainData.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        return trainData

    def createDataset(self):
        train_char_dir = Path("data/train_char")
        dfTrain = _addClassificationLabels(train_char_dir)
        for i in tqdm(range(len(dfTrain))):
            sample = dfTrain.iloc[i]
            self._write(
                open("data/train_char/"+sample['label'] + "/" + sample['image_id'] + ".jpg", 'rb').read(), sample['label'])
        self.writer.flush()
        self.writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='default',
                        help='config file name')
    parser.add_argument('--identifier', dest='identifier', action='store_true')
    parser.set_defaults(identifier=False)

    args = parser.parse_args()

    config = importlib.import_module(f"config.{args.config}")

    if args.identifier:
        dataset = IdentifierDataset(config.datasetParams)
        dataset.createDataset()

    if not arg.identifier:
        dataset = ClassifierDataset(config.datasetParams)
        dataset.createDataset()