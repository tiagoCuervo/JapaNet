
from comet_ml import Experiment
experiment = Experiment(
    api_key="vWqj2VRXub2HRuk5dzXl5yj3t",
    project_name="japanet",
    workspace="rcofre",
)
print("Logging experiment to COMET initialized")


from dataloader import ClassifierDataset

# default params for the data loading
defaultParams = datasetParams = {
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

dataset = ClassifierDataset(defaultParams)
trainData, validationData = dataset.load()
print("loading data done")


import argparse

# accept parameters for the NN from the command line
# currently the params are Wojtek-specific
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='learning rate for the optimizer')
    parser.add_argument('--n_epoch', default=None, type=int, required=True, help='number of epochs to train')
    parser.add_argument('--gpu', default=1, type=int, required=True, help='enable (1) or disable (0) training on GPU (if available)')
    args = parser.parse_args()
    return args

args = parse_args()

import os
if args.gpu == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Dense, Dropout, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobil_preprocess
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy

n_classes = dataset.dfTrain['label'].nunique()

def create_model(n_classes, input_shape=(64, 64, 3)):

    # more info about the architecture can be found here: https://tinyurl.com/p25w428k

    input_layer = Input(input_shape)

    x = tf.cast(input_layer, tf.float32)

    x = mobil_preprocess(x)

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights=None) # random weights initialization

    x = base_model(x)

    x = Flatten()(x)

    x = Dense(1280)(x)

    x = Activation('relu')(x)

    x = Dropout(rate=0.2)(x)

    x = Dense(n_classes)(x)

    output = Activation('softmax')(x)

    model = Model(input_layer, output)

    return model


checkpoint_path = "./trained_models/mobilnet_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=True,
                              verbose=-1)

mobilnet_model = create_model(n_classes)

mobilnet_model.compile(loss=SparseCategoricalCrossentropy(),
                       optimizer=Adam(lr=args.lr),
                       metrics='accuracy')

history = mobilnet_model.fit(x = trainData,
                            epochs=args.n_epoch, 
                            validation_data=validationData,
                            verbose = -1,
                            callbacks=[cp_callback])

mobilnet_model.save('./trained_models/mobilnet_model')
