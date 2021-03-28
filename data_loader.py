import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import importlib
from tqdm.auto import tqdm
from PIL import Image
import io
import random
import os
from pathlib import Path
import tensorflow_addons as tfa
import cv2


def bytesFeature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def floatFeature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64Feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image2Bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()


class IdentifierDataset:
    def __init__(self, config):
        self.config = config
        self.trainRecordPath = 'data/train/train_identifier.tfrecord'
        self.validationRecordPath = 'data/train/validation_identifier.tfrecord'
        self.feature_description = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'labels': tf.io.FixedLenFeature([128 * 128 * 5], dtype=tf.float32)
        }
        self._trainWriter = None
        self._validationWriter = None

    def _write(self, image, label):
        feature = {
            'image': bytesFeature(image),
            'labels': floatFeature(label.ravel())
        }
        sample = tf.train.Example(features=tf.train.Features(feature=feature))
        if random.random() < self.config['validationFraction']:
            self._validationWriter.write(sample.SerializeToString())
        else:
            self._trainWriter.write(sample.SerializeToString())

    def _createLabel(self, rawLabel, resizedRatioWidth=1.0, resizedRatioHeight=1.0):
        outputWidth = self.config['identifierInputWidth'] // self.config['identifierOutputStride']
        outputHeight = self.config['identifierInputHeight'] // self.config['identifierOutputStride']
        pageData = np.array(rawLabel.split(" ")).reshape(-1, 5)
        pageData = pageData[:, 1:].astype('uint32')
        pageData[:, [0, 2]] = pageData[:, [0, 2]] // resizedRatioWidth
        pageData[:, [1, 3]] = pageData[:, [1, 3]] // resizedRatioHeight
        xCenters = pageData[:, 0] + pageData[:, 2] // 2  # Center on X
        yCenters = pageData[:, 1] + pageData[:, 3] // 2  # Center on Y
        heatMapXCenters = (xCenters / self.config['identifierOutputStride']).astype(np.uint32)
        heatMapYCenters = (yCenters / self.config['identifierOutputStride']).astype(np.uint32)
        xOffset = (xCenters / self.config['identifierOutputStride'] - heatMapXCenters)
        yOffset = (yCenters / self.config['identifierOutputStride'] - heatMapYCenters)
        xSizes = pageData[:, 2] / self.config['identifierOutputStride']
        ySizes = pageData[:, 3] / self.config['identifierOutputStride']

        label = np.zeros((outputHeight, outputWidth, 5))
        for i in range(len(xCenters)):
            xCenter = heatMapXCenters[i]
            yCenter = heatMapYCenters[i]
            heatMap = ((np.exp(-(((np.arange(outputWidth) - xCenter) / (xSizes[i] / 10)) ** 2) / 2)).reshape(1, -1)
                       * (np.exp(-(((np.arange(outputHeight) - yCenter) / (ySizes[i] / 10)) ** 2) / 2)).reshape(-1, 1))
            label[:, :, 0] = np.maximum(label[:, :, 0], heatMap)
            label[yCenter, xCenter, 1] = xSizes[i] / outputWidth
            label[yCenter, xCenter, 2] = ySizes[i] / outputHeight
            label[yCenter, xCenter, 3] = xOffset[i] / outputWidth
            label[yCenter, xCenter, 4] = yOffset[i] / outputHeight
        return label

    def _processSample(self, rawSample):
        # Resizes image, gets its JPEG compressed data, and computes the new bounding boxes after resizing
        image = Image.open("data/train/" + rawSample['image_id'] + ".jpg")
        originalWidth = image.size[0]
        originalHeight = image.size[1]
        resizedImage = image.resize((self.config['identifierInputWidth'], self.config['identifierInputHeight']))
        imageBytes = image2Bytes(resizedImage)
        # Creates label (heatmap, xSize, ySize, xOffset, yOffset)
        label = self._createLabel(rawSample['labels'], originalWidth / self.config['identifierInputWidth'],
                                  originalHeight / self.config['identifierInputHeight'])
        return imageBytes, label

    def createDataset(self):
        dfTrain = pd.read_csv('data/train.csv')
        self._trainWriter = tf.io.TFRecordWriter(self.trainRecordPath)
        self._validationWriter = tf.io.TFRecordWriter(self.validationRecordPath)
        for i in tqdm(range(len(dfTrain))):
            sample = dfTrain.iloc[i]
            imageBytes, label = self._processSample(sample)
            self._write(imageBytes, label)
        self._trainWriter.flush()
        self._trainWriter.close()
        self._validationWriter.flush()
        self._validationWriter.close()

    def _processExample(self, example):
        pmap = tf.io.parse_single_example(example, self.feature_description)
        image = tf.image.decode_jpeg(pmap['image'], channels=3) / 255
        outputWidth = self.config['identifierInputWidth'] // self.config['identifierOutputStride']
        outputHeight = self.config['identifierInputHeight'] // self.config['identifierOutputStride']
        label = tf.reshape(pmap['labels'], (outputWidth, outputHeight, 5))
        return image, label

    def load(self):
        trainRecord = tf.data.TFRecordDataset(self.trainRecordPath)
        trainData = trainRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.cache()
        trainData = trainData.shuffle(buffer_size=self.config['identifierShufflingBufferSize'])
        trainData = trainData.batch(self.config['batchSize'], drop_remainder=True)
        trainData = trainData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        validationRecord = tf.data.TFRecordDataset(self.validationRecordPath)
        validationData = validationRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validationData = validationData.cache()
        validationData = validationData.batch(self.config['batchSize'], drop_remainder=True)
        validationData = validationData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return trainData, validationData


def _addClassificationLabels(imgs_folder_path):
    pair_img_label = []
    for CLASS in os.listdir(imgs_folder_path):
        if CLASS == 'train_char.zip':
            continue
        else:
            class_dir = os.path.join(imgs_folder_path, CLASS)
            for img_id in os.listdir(class_dir):
                pair_img_label.append([img_id.split('.')[0], CLASS])

    df = pd.DataFrame(pair_img_label, columns=['image_id', 'unicode'])
    label, unique = pd.factorize(df.unicode)
    df['label'] = label
    return df


class ClassifierDataset:
    def __init__(self, config):
        self.config = config
        self.trainRecordPath = 'data/train/train_classifier.tfrecord'
        self.validationRecordPath = 'data/train/validation_classifier.tfrecord'

        self.feature_description = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.int64)
        }
        self._trainWriter = None
        self._validationWriter = None
        self.dfTrain = _addClassificationLabels(Path("data/train_char"))
        self.dfCharFreq = pd.read_csv(Path("data/char_freq.csv"))

        label, unique = pd.factorize(self.dfCharFreq.Unicode)
        label_to_code_dict = {key: value for key, value in zip(label, unique)}

        # build a lookup table
        self.label_to_code = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(list(label_to_code_dict.keys()), dtype=tf.int64),
                                                tf.constant(list(
                                                    label_to_code_dict.values())), value_dtype=tf.string),
            default_value='-1')

        # self.code_to_freq = {key: value for key,value in zip(self.dfCharFreq.Unicode,self.dfCharFreq.Frequency)}
        self.code_to_freq = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(self.dfCharFreq.Unicode), tf.constant(self.dfCharFreq.Frequency)),
            default_value=-1)

    def _write(self, image, label):
        feature = {
            'image': bytesFeature(image),
            'label': int64Feature(label)
        }
        sample = tf.train.Example(features=tf.train.Features(feature=feature))

        if random.random() < self.config['validationFraction']:
            self._validationWriter.write(sample.SerializeToString())
        else:
            self._trainWriter.write(sample.SerializeToString())

    def _processSample(self, rawSample):
        image = Image.open("data/train_char/" + rawSample['unicode'] + "/" + rawSample['image_id'] + ".jpg")
        resizedImage = image.resize((self.config['classifierInputWidth'], self.config['classifierInputHeight']))
        return image2Bytes(resizedImage)

    def createDataset(self):

        self.dfTrain = self.dfTrain.sample(frac=1)
        self._trainWriter = tf.io.TFRecordWriter(self.trainRecordPath)
        self._validationWriter = tf.io.TFRecordWriter(self.validationRecordPath)

        for i in tqdm(range(len(self.dfTrain))):
            sample = self.dfTrain.iloc[i]
            imageBytes = self._processSample(sample)
            self._write(imageBytes, sample['label'])
        self._trainWriter.flush()
        self._trainWriter.close()
        self._validationWriter.flush()
        self._validationWriter.close()

    def _processExample(self, example):
        pmap = tf.io.parse_single_example(example, self.feature_description)
        image = tf.image.decode_jpeg(pmap['image'], channels=3) / 255
        label = pmap['label']

        return image, label

    def _augmenter(self, image, label):

        code = self.label_to_code.lookup(label)
        p_augment = 1 / self.code_to_freq[code]
        if np.random.rand() < p_augment:
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

            image = tf.clip_by_value(image, 0.0, 1.0)

            image = tfa.image.rotate(image, tf.random.uniform([], minval=-0.174533 / 2,
                                                              maxval=0.174533 / 2, dtype=tf.float32), fill_value=1.0)
        return image, label

    def _binarizing(self, image, label):
        img_0_orig = image * 255.0
        img_0 = cv2.cvtColor(img_0_orig, cv2.COLOR_RGB2BGR)
        img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(img_0, (1, 1), 0)
        blur = np.array(blur, dtype=np.uint8)
        sharp_mask = np.subtract(img_0, blur)
        img_0 = cv2.addWeighted(img_0, 1, sharp_mask, 10, 0)
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_1 = np.ones((3, 3), np.uint8)
        kernel_2 = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_2)

        mask = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
        img = cv2.add(img_0_orig, mask, dtype=cv2.CV_32F)
        blur_1 = cv2.GaussianBlur(img, (13, 13), 0)
        sharp_mask_1 = img - blur_1
        sharp_mask_1 = cv2.GaussianBlur(sharp_mask_1, (7, 7), 0)
        img = cv2.addWeighted(img, 1, sharp_mask_1, -10, 0, dtype=cv2.CV_32F)

        img = img / 255.0
        img = tf.clip_by_value(img, 0, 1)

        return img, label

    def _oversample_classes(self, image, label, oversampling_coef=0.9):
        """
        Returns the number of copies of given example
        """

        code = self.label_to_code.lookup(label)
        class_prob = self.code_to_freq[code] / self.dfCharFreq.Frequency.sum()
        class_target_prob = 1 / 4206
        prob_ratio = tf.cast(class_target_prob / class_prob, dtype=tf.float32)
        # soften ratio is oversampling_coef==0 we recover original distribution
        prob_ratio = prob_ratio ** oversampling_coef
        # for classes with probability higher than class_target_prob we
        # want to return 1
        prob_ratio = tf.maximum(prob_ratio, 1)
        # for low probability classes this number will be very large
        repeat_count = tf.floor(prob_ratio)
        # prob_ratio can be e.g 1.9 which means that there is still 90%
        # of change that we should return 2 instead of 1
        repeat_residual = prob_ratio - repeat_count  # a number between 0-1
        residual_acceptance = tf.less_equal(
            tf.random.uniform([], dtype=tf.float32), repeat_residual
        )

        residual_acceptance = tf.cast(residual_acceptance, tf.int64)
        repeat_count = tf.cast(repeat_count, dtype=tf.int64)

        return tf.data.Dataset.from_tensors((image, label)).repeat(repeat_count + residual_acceptance)

    def load(self):
        trainRecord = tf.data.TFRecordDataset(self.trainRecordPath)
        trainData = trainRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Oversampling low frequency classes
        trainData = trainData.map(self._oversample_classes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.flat_map(lambda x: x)

        trainData = trainData.shuffle(buffer_size=self.config['classifierShufflingBufferSize'])
        # augmenter
        trainData = trainData.map(self._augmenter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # binarizing
        trainData = trainData.map(lambda x, y: tf.numpy_function(self._binarizing, [x, y], [tf.float32, tf.int64]),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        trainData = trainData.batch(self.config['batchSize'], drop_remainder=True)
        trainData = trainData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        validationRecord = tf.data.TFRecordDataset(self.validationRecordPath)
        validationData = validationRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validationData = validationData.batch(self.config['batchSize'], drop_remainder=True)
        validationData = validationData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return trainData, validationData


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='default',
                        help='config file name')
    parser.add_argument('--identifier', dest='identifier', action='store_true')
    parser.add_argument('--classifier', dest='classifier', action='store_true')
    parser.set_defaults(identifier=False)
    parser.set_defaults(classifier=False)

    args = parser.parse_args()

    config = importlib.import_module(f"config.{args.config}")

    if args.identifier:
        dataset = IdentifierDataset(config.datasetParams)
        dataset.createDataset()
    else:
        dataset = ClassifierDataset(config.datasetParams)
        dataset.createDataset()
