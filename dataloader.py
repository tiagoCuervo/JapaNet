import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from PIL import Image
import io
import random
import os
from pathlib import Path
import tensorflow_addons as tfa
import cv2
import json
from zipfile import ZipFile


def image2Bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()


def bytesFeature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def floatFeature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64Feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class DetectorDataset:
    def __init__(self, setup):
        self.config = setup
        self.trainRecordPath = 'data/train/detector_train.tfrecord'
        self.validationRecordPath = 'data/train/detector_validation.tfrecord'
        self.feature_description = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'labels': tf.io.FixedLenFeature([128 * 128 * 5], dtype=tf.float32),
            'originalShape': tf.io.FixedLenFeature([2, ], dtype=tf.float32),
        }
        self._trainWriter = None
        self._validationWriter = None

    def _write(self, image, label, originalWidth, originalHeight):
        feature = {
            'image': bytesFeature(image),
            'labels': floatFeature(label.ravel()),
            'originalShape': floatFeature([originalWidth, originalHeight]),
        }
        sample = tf.train.Example(features=tf.train.Features(feature=feature))
        if random.random() < self.config['validationFraction']:
            self._validationWriter.write(sample.SerializeToString())
        else:
            self._trainWriter.write(sample.SerializeToString())

    def _createLabel(self, rawLabel, resizedRatioWidth=1.0, resizedRatioHeight=1.0):
        outputWidth = self.config['detectorInputWidth'] // self.config['detectorOutputStride']
        outputHeight = self.config['detectorInputHeight'] // self.config['detectorOutputStride']
        pageData = np.array(rawLabel.split(" ")).reshape(-1, 5)
        pageData = pageData[:, 1:].astype('uint32')
        pageData[:, [0, 2]] = pageData[:, [0, 2]] // resizedRatioWidth
        pageData[:, [1, 3]] = pageData[:, [1, 3]] // resizedRatioHeight
        xCenters = pageData[:, 0] + pageData[:, 2] // 2  # Center on X
        yCenters = pageData[:, 1] + pageData[:, 3] // 2  # Center on Y
        heatMapXCenters = (xCenters / self.config['detectorOutputStride']).astype(np.uint32)
        heatMapYCenters = (yCenters / self.config['detectorOutputStride']).astype(np.uint32)
        xOffset = (xCenters / self.config['detectorOutputStride'] - heatMapXCenters)
        yOffset = (yCenters / self.config['detectorOutputStride'] - heatMapYCenters)
        xSizes = pageData[:, 2]
        ySizes = pageData[:, 3]

        label = np.zeros((outputHeight, outputWidth, 5))
        for i in range(len(xCenters)):
            xCenter = heatMapXCenters[i]
            yCenter = heatMapYCenters[i]
            heatMap = ((np.exp(-(((np.arange(outputWidth) - xCenter) / (xSizes[i] / 10)) ** 2) / 2)).reshape(1, -1)
                       * (np.exp(-(((np.arange(outputHeight) - yCenter) / (ySizes[i] / 10)) ** 2) / 2)).reshape(-1, 1))
            label[:, :, 0] = np.maximum(label[:, :, 0], heatMap)
            label[yCenter, xCenter, 1] = xSizes[i].astype('float32')
            label[yCenter, xCenter, 2] = ySizes[i].astype('float32')
            label[yCenter, xCenter, 3] = xOffset[i].astype('float32')
            label[yCenter, xCenter, 4] = yOffset[i].astype('float32')
        return label

    def _processSample(self, rawSample, zipObject):
        # Resizes image, gets its JPEG compressed data, and computes the new bounding boxes after resizing
        image = tf.image.decode_jpeg(zipObject.read(rawSample['image_id'] + ".jpg"))
        originalWidth = image.shape[1]
        originalHeight = image.shape[0]
        resizedImage = tf.image.resize(image,
                                       [self.config['detectorInputWidth'], self.config['detectorInputHeight']])
        imageBytes = tf.image.encode_jpeg(tf.cast(resizedImage, tf.uint8)).numpy()
        # Creates label (heatmap, xSize, ySize, xOffset, yOffset)
        label = self._createLabel(rawSample['labels'], originalWidth / self.config['detectorInputWidth'],
                                  originalHeight / self.config['detectorInputHeight'])
        return imageBytes, label, originalWidth, originalHeight

    def createDataset(self):
        dfTrain = pd.read_csv('data/train.csv')
        zipObject = ZipFile('data/train_images.zip', 'r')
        self._trainWriter = tf.io.TFRecordWriter(self.trainRecordPath)
        self._validationWriter = tf.io.TFRecordWriter(self.validationRecordPath)
        for i in tqdm(range(len(dfTrain))):
            sample = dfTrain.iloc[i]
            imageBytes, label, originalWidth, originalHeight = self._processSample(sample, zipObject)
            self._write(imageBytes, label, originalWidth, originalHeight)
        self._trainWriter.flush()
        self._trainWriter.close()
        self._validationWriter.flush()
        self._validationWriter.close()

    def _processExample(self, example):
        pmap = tf.io.parse_single_example(example, self.feature_description)
        imageDecoded = tf.image.decode_jpeg(pmap['image'], channels=3) / 255
        imageResized = tf.image.resize(imageDecoded, [self.config['detectorInputWidth'],
                                                      self.config['detectorInputHeight']])
        outputWidth = self.config['detectorInputWidth'] // self.config['detectorOutputStride']
        outputHeight = self.config['detectorInputHeight'] // self.config['detectorOutputStride']
        label = tf.reshape(pmap['labels'], (outputWidth, outputHeight, 5))
        return imageResized, label

    def load(self):
        trainRecord = tf.data.TFRecordDataset(self.trainRecordPath)
        trainData = trainRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.shuffle(buffer_size=self.config['shufflingBufferSize'])
        trainData = trainData.batch(self.config['batchSize'], drop_remainder=True)
        trainData = trainData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        trainData = trainData.filter(
            lambda x, y: not tf.reduce_any(tf.math.is_nan(x)) and not tf.reduce_any(tf.math.is_nan(y)))
        validationRecord = tf.data.TFRecordDataset(self.validationRecordPath)
        validationData = validationRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validationData = validationData.batch(self.config['batchSize'], drop_remainder=True)
        validationData = validationData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        validationData = validationData.filter(
            lambda x, y: not tf.reduce_any(tf.math.is_nan(x)) and not tf.reduce_any(tf.math.is_nan(y)))
        return trainData, validationData


def classifierAugmenter(image, label):
    if random.random() < 0.5:
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tfa.image.rotate(image,
                                 tf.random.uniform([], minval=-0.174533 / 2, maxval=0.174533 / 2, dtype=tf.float32),
                                 fill_value=1.0)
    return image, label


class _ClassifierDataset:
    def __init__(self, config):
        self.config = config
        self.trainRecordPath = 'data/classifier_train.tfrecord'
        self.validationRecordPath = 'data/classifier_validation.tfrecord'
        self.feature_description = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.int64),
            'probability': tf.io.FixedLenFeature([], dtype=tf.float32)
        }
        self._trainWriter = None
        self._validationWriter = None
        self.charDF = None

    def _write(self, image, label, probability):
        feature = {
            'image': bytesFeature(image),
            'label': int64Feature(label),
            'probability': floatFeature([probability])
        }
        sample = tf.train.Example(features=tf.train.Features(feature=feature))
        if random.random() < self.config['validationFraction']:
            self._validationWriter.write(sample.SerializeToString())
        else:
            self._trainWriter.write(sample.SerializeToString())

    def _processSample(self, fileName, zipObject):
        label = fileName.split('_')[0]
        image = tf.image.decode_jpeg(zipObject.read(fileName))
        resizedImage = tf.image.resize(image,
                                       [self.config['classifierInputWidth'], self.config['classifierInputHeight']])
        imageBytes = tf.image.encode_jpeg(tf.cast(resizedImage, tf.uint8)).numpy()
        return imageBytes, label

    def createDataset(self):
        self.charDF = pd.read_csv('data/char_data.csv')
        totalNumSamples = self.charDF['Frequency'].sum()
        self._trainWriter = tf.io.TFRecordWriter(self.trainRecordPath)
        self._validationWriter = tf.io.TFRecordWriter(self.validationRecordPath)
        zipObject = ZipFile('data/characters.zip', 'r')
        listOfFileNames = zipObject.namelist()
        missingCounter = 0
        for fileName in tqdm(listOfFileNames):
            imageBytes, label = self._processSample(fileName, zipObject)
            labelAndFreq = self.charDF[self.charDF["Unicode"] == label][["Unicode_cat", "Frequency"]].values
            if len(labelAndFreq) > 0:
                labelCode, frequency = labelAndFreq[0]
                self._write(imageBytes, int(labelCode), frequency / totalNumSamples)
            else:
                missingCounter += 1
        self._trainWriter.flush()
        self._trainWriter.close()
        self._validationWriter.flush()
        self._validationWriter.close()
        print(f"In total we have {missingCounter} missing samples out of {totalNumSamples}")

    def _processExample(self, example):
        pmap = tf.io.parse_single_example(example, self.feature_description)
        imageDecoded = tf.image.decode_jpeg(pmap['image'], channels=3)
        imageResized = tf.image.resize(imageDecoded, [self.config['classifierInputWidth'],
                                                      self.config['classifierInputHeight']])
        label = pmap['label']
        # probability = pmap['probability']
        return imageResized / 255.0, label  # , probability * self.charDF['Frequency'].sum()

    def load(self):
        trainRecord = tf.data.TFRecordDataset(self.trainRecordPath)
        trainData = trainRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.map(classifierAugmenter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.shuffle(buffer_size=self.config['classifierShufflingBufferSize'])
        trainData = trainData.batch(self.config['batchSize'], drop_remainder=True)
        trainData = trainData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        validationRecord = tf.data.TFRecordDataset(self.validationRecordPath)
        validationData = validationRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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


def _binarizing(image, label):
    img_0_orig = image * 255.0
    img_0 = cv2.cvtColor(img_0_orig, cv2.COLOR_RGB2BGR)
    img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_0, (1, 1), 0)
    blur = np.array(blur, dtype=np.uint8)
    # sharp_mask = np.subtract(img_0, blur)
    # img_0 = cv2.addWeighted(img_0, 1, sharp_mask, 10, 0)
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
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    label = tf.convert_to_tensor(label, dtype=tf.int64)

    return img, label


def _reset_shapes(image, label):
    image.set_shape([64, 64, 3])
    label.set_shape([])
    return image, label


class ClassifierDataset:
    def __init__(self, setup):
        self.config = setup
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
        trainData = trainData.map(lambda x, y: tf.numpy_function(_binarizing, [x, y], [tf.float32, tf.int64]),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.map(_reset_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
    parser.add_argument('--detectorShufflingBufferSize', type=int, default=4096,
                        help='If there is enough memory should be greater or equal to the number of samples')
    parser.add_argument('--classifierShufflingBufferSize', type=int, default=2 ** 17,
                        help='If there is enough memory should be greater or equal to the number of samples')
    parser.add_argument('--batchSizeDetector', type=int, default=128)
    parser.add_argument('--batchSizeClassifier', type=int, default=4096)
    parser.add_argument('--detectorInputHeight', type=int, default=512)
    parser.add_argument('--detectorInputWidth', type=int, default=512)
    parser.add_argument('--classifierInputWidth', type=int, default=64)
    parser.add_argument('--classifierInputHeight', type=int, default=64)
    parser.add_argument('--detectorOutputStride', type=int, default=4)
    parser.add_argument('--validationFraction', type=float, default=0.2,
                        help='Fraction of the total data to use as validation set')
    parser.add_argument('--detector', dest='detector', action='store_true')
    parser.add_argument('--classifier', dest='classifier', action='store_true')
    parser.set_defaults(detector=False)
    parser.set_defaults(classifier=False)

    args = parser.parse_args()
    args = vars(args)

    if args['detector'] and not args['classifier']:
        print("Creating detector dataset")
        DetectorDataset(args).createDataset()
    elif args['classifier'] and not args['detector']:
        print("Creating classifier dataset")
        ClassifierDataset(args).createDataset()
    else:
        print("Creating detector dataset")
        DetectorDataset(args).createDataset()
        print("Creating classifier dataset")
        ClassifierDataset(args).createDataset()
    print("Saving setup")
    with open('config/config.json', 'w') as fp:
        json.dump(args, fp)
    print("Input pipeline built! :D")
