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
        trainData = trainData.shuffle(buffer_size=self.config['identifierShufflingBufferSize'])
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
    codes, unique = pd.factorize(df.unicode)
    df['label'] = codes

    return df, unique

class ClassifierDataset:
    def __init__(self, config):
        self.config = config
        self.trainRecordPath = 'data/train/train_classifier.tfrecord'
        self.validationRecordPath = 'data/train/validation_classifier.tfrecord'
        
        self.feature_description = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.int16)
        }
        self._trainWriter = None
        self._validationWriter = None
        self.label_to_code=None

    def _write(self, image, label):
        feature = {
            'image': bytesFeature(image),
            'label': int64Feature(label)
        }
        sample = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # TO BE MODIFIED
        if random.random() < self.config['validationFraction']:
            self._validationWriter.write(sample.SerializeToString())
        else:
            self._trainWriter.write(sample.SerializeToString())
    
    

    def _processSample(self, rawSample):
        image = Image.open("data/train_char/" + rawSample['unicode'] + "/" + rawSample['image_id'] + ".jpg")
        resizedImage = image.resize((self.config['classifierInputWidth'], self.config['classifierInputHeight']))
        return image2Bytes(resizedImage) 

    def createDataset(self):
        trainCharDir = Path("data/train_char")
        dfTrain, label_to_code = _addClassificationLabels(trainCharDir)
        self._trainWriter = tf.io.TFRecordWriter(self.trainRecordPath)
        self._validationWriter = tf.io.TFRecordWriter(self.validationRecordPath)
        self.label_to_code = label_to_code
        for i in tqdm(range(len(dfTrain))):
            sample = dfTrain.iloc[i]
            imageBytes = self._processSample(sample)
            self._write(imageBytes, sample['label'])
        self._trainWriter.flush()
        self._trainWriter.close()
        self._validationWriter.flush()
        self._validationWriter.close()

    def _processExample(self, example):
        pmap = tf.io.parse_single_example(example, self.feature_description)
        image = tf.image.decode_jpeg(pmap['image'], channels=3) / 255
        
        return image, pmap['label']


    def load(self):
        trainRecord = tf.data.TFRecordDataset(self.trainRecordPath)
        trainData = record.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.shuffle( buffer_size=self.config['classifierShufflingBufferSize'])
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
