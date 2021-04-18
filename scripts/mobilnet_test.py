
from comet_ml import Experiment
def comet_logger(api_key="vWqj2VRXub2HRuk5dzXl5yj3t",
                 project_name="japanet",
                 workspace="rcofre"):

    print("Logging experiment to COMET initialized")

    experiment = Experiment(
    api_key=api_key,
    project_name=project_name,
    workspace=workspace,
    )

    return experiment

experiment = comet_logger()

import argparse
# currently the params are WPC-specific
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='learning rate for the optimizer')
    parser.add_argument('--n_epoch', default=10, type=int, required=True, help='number of epochs to train')
    parser.add_argument('--gpu', default=1, type=int, required=False, help='enable (1) or disable (0) training on GPU (if available)')
    parser.add_argument('--batch_size', default=100, type=int, required=True, help='number of images in a train/validation batch')
    parser.add_argument('--restore', default=None, type=str, required=False, help='if filepath provided, restores training from that checkpoint file')
    args = parser.parse_args()
    return args

args = parse_args()

from dataloader import ClassifierDataset
defaultParams = datasetParams = {
    'identifierShufflingBufferSize': 100,
    'classifierShufflingBufferSize': 100,
    'batchSize': args.batch_size,
    'identifierInputHeight': 512,
    'identifierInputWidth': 512,
    'identifierOutputStride': 4,
    'validationFraction': 0.2,
    'classifierInputWidth': 64,
    'classifierInputHeight': 64
}

dataset = ClassifierDataset(defaultParams)
trainData, validationData = dataset.load()
print("loading data done")


import os
if args.gpu == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Dense, Dropout, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobil_preprocess
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives

def create_model(n_classes, input_shape=(64, 64, 3)):

    # more info about the architecture can be found here: https://tinyurl.com/p25w428k

    input_layer = Input(input_shape)

    x = mobil_preprocess(input_layer * 255)

    base_model = tf.keras.applications.MobileNetV3Large(input_shape=input_shape,
                                                        include_top=False,
                                                        pooling='avg',
                                                        weights=None) # random weights initialization

    x = base_model(x, training=True)

    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(rate=0.2)(x)

    x = Dense(n_classes)(x)
    output = Activation('softmax')(x)

    model = Model(input_layer, output)

    return model

checkpoint_path = "./trained_models/mobilnet_training/mobilenet.{epoch:02d}-{val_accuracy:.2f}.ckpt"

cp_callback = ModelCheckpoint(
                    checkpoint_path, monitor='val_accuracy', verbose=0, save_best_only=True,
                    save_weights_only=False, mode='auto', save_freq='epoch'
                    )

n_classes = dataset.dfTrain['label'].nunique()

mobilnet_model = create_model(n_classes)

mobilnet_model.compile(loss=SparseCategoricalCrossentropy(),
                       optimizer=Adam(lr=args.lr),
                       metrics=['accuracy'])

history = mobilnet_model.fit(x = trainData,
                            epochs=args.n_epoch, 
                            validation_data=validationData,
                            verbose = 0,
                            callbacks=[cp_callback])

mobilnet_model.save('./trained_models/mbv3_model')
