import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, LeakyReLU, Softmax, Conv2DTranspose, \
    Concatenate, Add, AveragePooling2D, GlobalAveragePooling2D, Activation, MaxPool2D, Flatten, Dropout
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobil_preprocess
import keras.backend as K
import numpy as np


def convBatchReLU(x, numFilters, kernelSize=3, stride=1, padding="same", upsample=False):
    if upsample:
        x = Conv2DTranspose(numFilters, kernel_size=kernelSize, strides=stride, padding=padding)(x)
    else:
        x = Conv2D(numFilters, kernel_size=kernelSize, strides=stride, padding=padding)(x)
    x = BatchNormalization()(x)
    return LeakyReLU(alpha=0.1)(x)


def residualBlock(x, numFilters, stride):
    y = convBatchReLU(x, numFilters, stride=stride)
    y = Conv2D(numFilters, 3, strides=1, padding='same')(y)
    if stride == 2 or numFilters != x.shape[-1]:
        x = Conv2D(numFilters, 3, strides=stride, padding='same')(x)
    return LeakyReLU(alpha=0.1)(Add()([x, y]))


class ResNet:
    def __init__(self, inputShape, numBlocks, numBlockFilters, numClasses=None, outputBias=None):
        super(ResNet, self).__init__()
        self.inputShape = inputShape
        self.numBlocks = numBlocks
        self.numBlockFilters = numBlockFilters
        self.numClasses = numClasses
        self.outputBias = outputBias
        self.model = None

    def predict(self, x):
        return self.model.predict(x)

    def buildModel(self):
        inputLayer = Input(self.inputShape)
        x = convBatchReLU(inputLayer, 64, 7, 2)
        x = MaxPool2D((3, 3), strides=2, padding="same")(x)
        stride = 1
        for i in range(4):
            for _ in range(self.numBlocks[i]):
                x = residualBlock(x, self.numBlockFilters[i], stride=stride)
                stride = 1
            stride = 2
        # At this point the dims are original / 2^5
        x = GlobalAveragePooling2D()(x)  # Now the outputs are a 1D vector of size numBlockFilters[-1]
        if self.numClasses is not None:
            x = Dense(1000)(x)
            x = LeakyReLU(alpha=0.1)(x)
            if self.outputBias is not None:
                self.outputBias = tf.keras.initializers.Constant(self.outputBias)
            x = Dense(self.numClasses, bias_initializer=self.outputBias)(x)
            x = Softmax()(x)
        return Model(inputLayer, x)


class ResNet18(ResNet):
    def __init__(self, inputShape, numClasses=None, outputBias=None):
        super(ResNet18, self).__init__(inputShape, [2] * 4, [64, 128, 256, 512], numClasses, outputBias)
        self.model = self.buildModel()


class ResNet34(ResNet):
    def __init__(self, inputShape, numClasses=None, outputBias=None):
        super(ResNet34, self).__init__(inputShape, [3, 4, 6, 3], [64, 128, 256, 512], numClasses, outputBias)
        self.model = self.buildModel()


def aggregationBlock(x, xSkip, numChannels, stride=2):
    x = convBatchReLU(x, numChannels, kernelSize=2, stride=stride, upsample=True)
    x = Concatenate()([x, xSkip])
    return convBatchReLU(x, numChannels, kernelSize=1, stride=1)


def heatMapLoss(target, pred, mask=None):
    alpha = 2.
    beta = 4.
    if mask is None:
        mask = K.sign(target[:, :, :, 1])
    return -K.sum(target[:, :, :, 0] * ((1 - pred[:, :, :, 0]) ** alpha) * K.log(pred[:, :, :, 0] + 1e-6) + (
            1 - target[:, :, :, 0]) * ((1 - mask) ** beta) * (pred[:, :, :, 0] ** alpha) * K.log(
        1 - pred[:, :, :, 0] + 1e-6))


def offsetLoss(target, pred, mask=None):
    if mask is None:
        mask = K.sign(target[:, :, :, 1])
    return K.sum(
        K.abs(target[:, :, :, 3] - pred[:, :, :, 3] * mask) + K.abs(target[:, :, :, 4] - pred[:, :, :, 4] * mask))


def sizeLoss(target, pred, mask=None):
    if mask is None:
        mask = K.sign(target[:, :, :, 1])
    return K.sum(
        K.abs(target[:, :, :, 1] - pred[:, :, :, 1] * mask) + K.abs(target[:, :, :, 2] - pred[:, :, :, 2] * mask))


def centerNetLoss(target, pred):
    mask = K.sign(target[:, :, :, 1])
    N = K.sum(mask)
    heatloss = heatMapLoss(target, pred, mask)
    offsetloss = offsetLoss(target, pred, mask)
    sizeloss = sizeLoss(target, pred, mask)
    return (heatloss + 1.0 * offsetloss + 0.1 * sizeloss) / N


class CenterNet(ResNet):
    def __init__(self, inputShape, numBlocks, numBlockFilters, numClasses, inOutRatio=4):
        super(CenterNet, self).__init__(inputShape, numBlocks, numBlockFilters, numClasses)
        self.numOutputs = numClasses + 4
        assert isinstance(inOutRatio, int) and 0 < inOutRatio < 6
        self.inOutRatio = inOutRatio
        self.model = self.buildModel()

    def buildModel(self):
        inputLayer = Input(self.inputShape)
        xDown = []
        x = convBatchReLU(inputLayer, 64, 7, 2)
        xDown.append(x)
        x = MaxPool2D((3, 3), strides=2, padding="same")(x)
        xDown.append(x)
        stride = 1
        for i in range(4):
            for _ in range(self.numBlocks[i]):
                x = residualBlock(x, self.numBlockFilters[i], stride=stride)
                stride = 1
            stride = 2
            if i > 0:
                xDown.append(x)

        x = AveragePooling2D((2, 2))(x)  # At this point the dims are original / 2^6
        x = convBatchReLU(x, 1024, 1)

        for i in range(4, -1, -1):
            x = aggregationBlock(x, xDown[i], self.numBlockFilters[max(0, i - 1)])
            for _ in range(self.numBlocks[max(0, i - 1)] - 1):
                x = residualBlock(x, self.numBlockFilters[max(0, i - 1)], stride=1)
            if self.inputShape[0] // x.shape[1] <= self.inOutRatio:
                break
        heatMap = Conv2D(1, kernel_size=3, strides=1, padding="same")(x)
        heatMap = Activation("sigmoid")(heatMap)
        sizes = Conv2D(2, kernel_size=3, strides=1, padding="same")(x)
        offsets = Conv2D(2, kernel_size=3, strides=1, padding="same")(x)
        offsets = Activation("sigmoid")(offsets)
        x = Concatenate()([heatMap, sizes, offsets])
        return Model(inputLayer, x)

    def predictBoundingBox(self, x, confidenceThreshold=0.3, ioUThreshold=0.4):
        prediction = self.predict(x)
        yHat = prediction[:, :, :, 0]
        sHatX = prediction[:, :, :, 1]
        sHatY = prediction[:, :, :, 2]
        oHatX = prediction[:, :, :, 3]
        oHatY = prediction[:, :, :, 4]
        boundingBoxes = []
        for i in range(yHat.shape[0]):
            detectedIdxs = np.argwhere(yHat[i, :, :])
            boxes = np.zeros((detectedIdxs.shape[0], 4))
            scores = np.zeros((detectedIdxs.shape[0],))
            for j in range(detectedIdxs.shape[0]):
                scores[j] = yHat[i, detectedIdxs[j, 0], detectedIdxs[j, 1]]
                boxes[j, 0] = (detectedIdxs[j, 0] + oHatY[i, detectedIdxs[j, 0], detectedIdxs[j, 1]]) * 4 - sHatY[
                    i, detectedIdxs[j, 0], detectedIdxs[j, 1]] / 2
                boxes[j, 1] = (detectedIdxs[j, 1] + oHatX[i, detectedIdxs[j, 0], detectedIdxs[j, 1]]) * 4 - sHatX[
                    i, detectedIdxs[j, 0], detectedIdxs[j, 1]] / 2
                boxes[j, 2] = boxes[j, 0] + sHatY[i, detectedIdxs[j, 0], detectedIdxs[j, 1]]
                boxes[j, 3] = boxes[j, 1] + sHatX[i, detectedIdxs[j, 0], detectedIdxs[j, 1]]
            boxes = boxes / 512.0
            selectedIndices = tf.image.non_max_suppression(boxes, scores, max_output_size=detectedIdxs.shape[0],
                                                           iou_threshold=ioUThreshold,
                                                           score_threshold=confidenceThreshold)
            boundingBoxes.append(tf.gather(boxes, selectedIndices))
        return boundingBoxes


class ConvNetBaseline:
    def __init__(self, inputShape, numClasses=None, outputBias=None):
        self.inputShape = inputShape
        self.numClasses = numClasses
        self.outputBias = outputBias
        self.model = self.buildModel()

    def predict(self, x):
        return self.model.predict(x)

    def buildModel(self):
        inputLayer = Input(shape=self.inputShape)

        x = Conv2D(filters=32, kernel_size=[5, 5], padding="same")(inputLayer)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Input Shape: [batch_size, 64, 64, 32] -> Output Shape: [batch_size, 32, 32, 32]

        x = MaxPool2D(pool_size=[2, 2], strides=2)(x)

        # Input Shape: [batch_size, 32, 32, 32]-->Output Shape: [batch_size, 32, 32, 64]

        x = Conv2D(filters=64, kernel_size=[5, 5], padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Input Shape: [batch_size, 32, 32, 64]----> Output Shape: [batch_size, 16, 16, 64]

        x = MaxPool2D(pool_size=[2, 2], strides=2)(x)

        # Input Shape: [batch_size, 16, 16, 64]----> Output  Shape: [batch_size, 16, 16, 128]
        x = Conv2D(filters=128, kernel_size=[5, 5], padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Input Shape: [batch_size, 16, 16, 128] ---> Output Shape: [batch_size, 8, 8, 128]
        x = MaxPool2D(pool_size=[2, 2], strides=2)(x)

        # Input Shape: [batch_size, 8, 8, 128]----> Output Shape: [batch_size, 8, 8, 256]
        x = Conv2D(filters=256, kernel_size=[5, 5], padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Input Shape: [batch_size, 8, 8, 256]----> Output Shape: [batch_size, 4, 4, 256]
        x = MaxPool2D(pool_size=[2, 2], strides=2)(x)

        # Input Shape: [batch_size, 4, 4, 256]--->Output Shape: [batch_size, 4, 4, 512]
        x = Conv2D(filters=512, kernel_size=[5, 5], padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Input Shape: [batch_size, 4, 4, 512]----> Output Shape: [batch_size, 2, 2, 512]
        x = MaxPool2D(pool_size=[2, 2], strides=2)(x)

        # Input Shape: [batch_size, 2, 2, 512]----> Output Shape: [batch_size, 2 * 2 * 512]
        x = Flatten()(x)
        x = Dense(units=1024)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.25)(x)

        x = Dense(units=1024)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.25)(x)

        if self.outputBias is not None:
            self.outputBias = tf.keras.initializers.Constant(self.outputBias)
        x = Dense(self.numClasses, bias_initializer=self.outputBias)(x)
        x = Softmax()(x)

        return Model(inputLayer, x)


class MobileNetV3:
    def __init__(self, inputShape, numClasses=None, outputBias=None):
        self.inputShape = inputShape
        self.numClasses = numClasses
        self.outputBias = outputBias
        self.model = self.buildModel()

    def buildModel(self):
        inputLayer = Input(shape=self.inputShape)

        x = mobil_preprocess(inputLayer * 255)

        base_model = tf.keras.applications.MobileNetV3Large(input_shape=self.inputShape,
                                                            include_top=False,
                                                            pooling='avg',
                                                            weights=None)  # random weights initialization
        x = base_model(x, training=True)

        x = Dense(units=1024)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(rate=0.25)(x)

        if self.outputBias is not None:
            self.outputBias = tf.keras.initializers.Constant(self.outputBias)
        x = Dense(units=self.numClasses, bias_initializer=self.outputBias)(x)
        x = Softmax()(x)

        return Model(inputLayer, x)
