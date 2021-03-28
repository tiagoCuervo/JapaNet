from comet_ml import Experiment
experiment = Experiment(
    api_key="XEXW8fsoCNViIiUNbIVQUphhm",
    project_name="japanet",
    workspace="rcofre",
)
print("Logging experiment to COMET initialized")


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU,Conv2DTranspose, AveragePooling2D, Concatenate, Add, UpSampling2D, Activation, MaxPooling2D, Softmax, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import sys
import importlib
import matplotlib.pyplot as plt
import time
import tensorflow.keras.backend as K
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float,
                        required=False, help='learning rate for the optimizer')
    parser.add_argument('--n_epoch', default=30, type=int,
                        required=True, help='number of epochs to train')
    parser.add_argument('--gpu', default=1, type=int, required=True,
                        help='enable (1) or disable (0) training on GPU (if available)')
    args = parser.parse_args()
    return args

args = parse_args()

if args.gpu == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from dataloader import ClassifierDataset


datasetParams = {
    'identifierShufflingBufferSize': 100,
    'classifierShufflingBufferSize': 100,
    'batchSize': 32,
    'identifierInputHeight': 512,
    'identifierInputWidth': 512,
    'identifierOutputStride': 4,
    'validationFraction': 0.2,
    'classifierInputWidth': 64,
    'classifierInputHeight': 64
}

dataset = ClassifierDataset(datasetParams)
trainData, validationData = dataset.load()

#input shape should come from datasetParams?
def create_model(n_classes, input_shape = (64,64,3)):
    
    input_layer = Input(shape = input_shape)

    x = Conv2D(filters=32, kernel_size=[5, 5], padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Input Shape: [batch_size, 64, 64, 32] -> Output Shape: [batch_size, 32, 32, 32]
    
    x = MaxPooling2D(pool_size=[2, 2], strides=2)(x)

    # Input Shape: [batch_size, 32, 32, 32]-->Output Shape: [batch_size, 32, 32, 64]
    
    x = Conv2D(filters=64, kernel_size=[5, 5], padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Input Shape: [batch_size, 32, 32, 64]----> Output Shape: [batch_size, 16, 16, 64]

    x = MaxPooling2D(pool_size=[2, 2], strides=2)(x)

    # Input Shape: [batch_size, 16, 16, 64]----> Output  Shape: [batch_size, 16, 16, 128]
    x = Conv2D(filters=128, kernel_size=[5, 5], padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Input Shape: [batch_size, 16, 16, 128] ---> Output Shape: [batch_size, 8, 8, 128]
    x = MaxPooling2D(pool_size=[2, 2], strides=2)(x)

    # Input Shape: [batch_size, 8, 8, 128]----> Output Shape: [batch_size, 8, 8, 256]
    x = Conv2D(filters=256, kernel_size=[5, 5], padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)


    # Input Shape: [batch_size, 8, 8, 256]----> Output Shape: [batch_size, 4, 4, 256]
    x = MaxPooling2D(pool_size=[2, 2], strides=2)(x)

    # Input Shape: [batch_size, 4, 4, 256]--->Output Shape: [batch_size, 4, 4, 512]
    x = Conv2D(filters=512, kernel_size=[5, 5], padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Input Shape: [batch_size, 4, 4, 512]----> Output Shape: [batch_size, 2, 2, 512]
    x = MaxPooling2D(pool_size=[2, 2], strides=2)(x)

    # Input Shape: [batch_size, 2, 2, 512]----> Output Shape: [batch_size, 2 * 2 * 512]
    x = Flatten()(x)
    x = Dense( units = 1024)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.25)(x)

    x = Dense(units = 1024)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.25)(x)

    x = Dense(units = n_classes)(x)
    out = Softmax()(x)
    
    model = Model(input_layer,out)
    return model


# Create instance of model
n_classes = dataset.dfTrain['label'].nunique()
model = create_model(n_classes)


# Callbacks
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint

def lrs(epoch):
    lr = 0.001
    if epoch >= 20: 
        lr = 0.0002
    return lr

lr_schedule = LearningRateScheduler(lrs)

checkpoint_path = "./trained_models/convnet_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=True,
                              verbose=-1, save_freq = 1000)



# loss
cce_loss = tf.keras.losses.SparseCategoricalCrossentropy()


#metric
def recall_m(y_true, y_pred):

    true_positives  =  tf.cast(tf.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.argmax(y_pred,0),tf.int64), tf.cast(y_true, tf.int64)), tf.int64)),tf.float32)
    possible_positives = tf.cast(32, tf.float32)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = tf.cast(tf.reduce_sum(tf.cast(tf.math.equal(
        tf.cast(tf.argmax(y_pred, 0), tf.int64), tf.cast(y_true, tf.int64)), tf.int64)), tf.float32)
    predicted_positives = tf.cast(32, tf.float32)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def model_fit_(model, trainData, valData, n_epoch):
    hist = model.fit(
        trainData,
        epochs = n_epoch,
        # validation_data=valData,
        callbacks = [lr_schedule, cp_callback],
        verbose = -1
    )
    return hist


learning_rate= args.lr
n_epoch= args.n_epoch
batch_size=32
model.compile(loss=cce_loss, optimizer=Adam(lr=learning_rate), metrics = ['accuracy', f1_m, precision_m, recall_m])
hist = model_fit_(model, trainData, validationData, 30)


model.save('./trained_models/convnet_model')
