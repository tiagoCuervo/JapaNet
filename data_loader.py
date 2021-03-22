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
    label, unique = pd.factorize(df.unicode)
    label_to_code = {key : value for key,value in zip(label,unique)}
    df['label'] = label

    return df, label_to_code




 


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
        self.dfTrain, self.label_to_code = _addClassificationLabels(Path("data/train_char"))
        self.dfCharFreq = pd.read_csv(Path("data/char_freq.csv"))

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

    def _augmenter(self,image, label):
        p_augment = self.dfCharFreq[self.dfCharFreq['Unicode']== self.label_to_code[label.ref()]].probability.item()

        if np.random.rand()<p_augment:
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

            image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    def oversample_classes(self,example, oversampling_coef=0.9):
        """
        Returns the number of copies of given example
        """
        pmap  = tf.io.parse_single_example(example, self.feature_description)
        label = pmap['label']
        code  = self.label_to_code[label.ref()]
        class_prob = self.dfCharFreq[self.char_freq['Unicode']==code].Frequency.item()/self.char_freq.Frequency.su()
        class_target_prob = 1/4206
        prob_ratio = tf.cast(class_target_prob/class_prob, dtype=tf.float32)
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

        return repeat_count + residual_acceptance


    def load(self):
        trainRecord = tf.data.TFRecordDataset(self.trainRecordPath)
        #Oversampling low frequency classes
        # trainData = trainRecord.flat_map(lambda x: tf.data.Dataset.from_tensors(x).repeat(self.oversample_classes(x)))
        trainData = trainRecord.map(self._processExample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        trainData = trainData.shuffle( buffer_size=self.config['classifierShufflingBufferSize'])
        #augmenter
        # trainData = trainData.map(self._augmenter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
