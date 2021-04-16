import argparse
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from model import CenterNet, ResNet18, ConvNetBaseline, centerNetLoss, heatMapLoss, sizeLoss, offsetLoss
from dataloader import DetectorDataset, ClassifierDataset
import datetime
import json
import numpy as np
import pandas as pd


def trainDetector(setup):
    with open('config.json') as fp:
        dataConfig = json.load(fp)
    dataset = DetectorDataset(dataConfig)
    trainData, validationData = dataset.load()
    centerNet = CenterNet([512, 512, 3], [3, 4, 6, 3], [64, 128, 256, 512], 1)
    centerNet.model.compile(loss=centerNetLoss, optimizer=Adam(lr=setup.initLr),
                            metrics=[heatMapLoss, sizeLoss, offsetLoss])
    lrSchedule = ReduceLROnPlateau(monitor='loss', factor=setup.lrDecay, patience=setup.lrPatience,
                                   min_lr=setup.minLr)
    checkpointPath = "trained_models/detector.{epoch:02d}-{val_loss:.2f}.h5"
    modelSavior = ModelCheckpoint(filepath=checkpointPath, save_best_only=True, save_freq='epoch')
    try:
        centerNet.model.fit(
            trainData,
            epochs=setup.numEpochs,
            validation_data=validationData,
            callbacks=[lrSchedule, modelSavior],
            verbose=1
        )
    except KeyboardInterrupt:
        centerNet.model.save('trained_models/detector_' + str(datetime.datetime.now()) + '.hdf5')
        print('Last model saved')
        pass


def trainClassifier(setup):
    with open('config.json') as fp:
        dataConfig = json.load(fp)
    dataset = ClassifierDataset(dataConfig)
    trainData, validationData = dataset.load()

    charDF = pd.read_csv('data/char_data.csv')
    countsByClass = charDF.sort_values('Unicode_cat').Frequency.values
    totalNumSamples = charDF['Frequency'].sum()
    beta = (totalNumSamples - 1) / totalNumSamples
    classWeights = pd.DataFrame((1 - beta) / (1 - beta ** countsByClass))
    classWeights = classWeights.to_dict()[0]
    probs = countsByClass / totalNumSamples
    classifier = ResNet18([64, 64, 3], numClasses=4206, outputBias=np.log(probs))

    classifier.model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(lr=setup.initLr),
                             metrics=['accuracy'])
    lrSchedule = ReduceLROnPlateau(monitor='loss', factor=setup.lrDecay, patience=setup.lrPatience,
                                   min_lr=setup.minLr)
    checkpointPath = "trained_models/classifier.{epoch:02d}-{val_loss:.2f}.h5"
    modelSavior = ModelCheckpoint(filepath=checkpointPath, save_best_only=True, save_freq='epoch')
    try:
        classifier.model.fit(
            trainData,
            epochs=setup.numEpochs,
            validation_data=validationData,
            callbacks=[lrSchedule, modelSavior],
            verbose=1,
            class_weight=classWeights
        )
    except KeyboardInterrupt:
        classifier.model.save('trained_models/classifier_' + str(datetime.datetime.now()) + '.hdf5')
        print('Last model saved')
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='evaluate', help='config file name')
    parser.add_argument('--numEpochs', type=int, default=100)
    parser.add_argument('--initLr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrDecay', type=float, default=0.75,
                        help='Factor to which multiply learning rate at each training plateau')
    parser.add_argument('--lrPatience', type=int, default=1,
                        help='How many epochs to wait before decaying learning rate')
    parser.add_argument('--minLr', type=float, default=1e-12, help='Minimum learning rate')
    parser.add_argument('--detector', dest='detector', action='store_true')
    parser.add_argument('--classifier', dest='classifier', action='store_true')
    parser.set_defaults(detector=False)
    parser.set_defaults(classifier=False)
    args = parser.parse_args()

    if args.mode == 'train' or args.mode == 'train_and_evaluate':
        if args.detector and not args.classifier:
            print("Training detector")
            trainDetector(args)
        elif args['classifier'] and not args['detector']:
            print("Training classifier")
            trainClassifier(args)
        else:
            raise NotImplementedError

    if args.mode == 'evaluate' or args.mode == 'train_and_evaluate':
        raise NotImplementedError
