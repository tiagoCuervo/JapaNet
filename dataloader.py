import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
import io
import random
import os
import tensorflow_addons as tfa
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
        self.trainRecordPath = 'data/detector_train.tfrecord'
        self.validationRecordPath = 'data/detector_validation.tfrecord'
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
        trainData = trainData.shuffle(buffer_size=self.config['detectorShufflingBufferSize'])
        trainData = trainData.batch(self.config['batchSizeDetector'], drop_remainder=True)
        trainData = trainData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        trainData = trainData.filter(
            lambda x, y: not tf.reduce_any(tf.math.is_nan(x)) and not tf.reduce_any(tf.math.is_nan(y)))
        validationRecord = tf.data.TFRecordDataset(self.validationRecordPath)
        validationData = validationRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validationData = validationData.batch(self.config['batchSizeDetector'], drop_remainder=True)
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
        return imageResized / 255.0, label

    def load(self):
        trainRecord = tf.data.TFRecordDataset(self.trainRecordPath)
        trainData = trainRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.map(classifierAugmenter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.shuffle(buffer_size=self.config['classifierShufflingBufferSize'])
        trainData = trainData.batch(self.config['batchSizeClassifier'], drop_remainder=True)
        trainData = trainData.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        validationRecord = tf.data.TFRecordDataset(self.validationRecordPath)
        validationData = validationRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validationData = validationData.batch(self.config['batchSizeClassifier'], drop_remainder=True)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detectorShufflingBufferSize', type=int, default=4096,
                        help='If there is enough memory should be greater or equal to the number of samples')
    parser.add_argument('--classifierShufflingBufferSize', type=int, default=2 ** 17,
                        help='If there is enough memory should be greater or equal to the number of samples')
    parser.add_argument('--batchSizeDetector', type=int, default=32)
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
        _ClassifierDataset(args).createDataset()
    else:
        print("Creating detector dataset")
        DetectorDataset(args).createDataset()
        print("Creating classifier dataset")
        _ClassifierDataset(args).createDataset()
    print("Saving setup")
    with open('config/config.json', 'w') as fp:
        json.dump(args, fp)
    print("Input pipeline built! :D")
