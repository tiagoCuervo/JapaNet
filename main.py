import argparse
from zipfile import ZipFile
from tqdm.auto import tqdm
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from model import CenterNet, ResNet18, ResNet34, ConvNetBaseline, MobileNetV3, centerNetLoss, heatMapLoss, sizeLoss, \
    offsetLoss
from dataloader import DetectorDataset, _ClassifierDataset
import datetime
import json
import numpy as np
import pandas as pd
import os
from utils import kuzushijiF1


def trainDetector(setup):
    with open('config/config.json') as fp:
        dataConfig = json.load(fp)
    dataset = DetectorDataset(dataConfig)
    trainData, validationData = dataset.load()
    centerNet = CenterNet([dataConfig['detectorInputWidth'], dataConfig['detectorInputHeight'], 3],
                          [3, 4, 6, 3], [64, 128, 256, 512], 1)
    if args.detectorPath is not None:
        centerNet.model = tf.keras.models.load_model(args.detectorPath, custom_objects={
            "loss": centerNetLoss,
            "heatMapLoss": heatMapLoss,
            "sizeLoss": sizeLoss,
            "offsetLoss": offsetLoss
        })
    else:
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
        centerNet.model.save('trained_models/detector_' + str(datetime.datetime.now()).split(' ')[0] + '.hdf5')
        print('Last model saved')
        pass


def trainClassifier(setup):
    with open('config/config.json') as fp:
        dataConfig = json.load(fp)
    dataset = _ClassifierDataset(dataConfig)
    trainData, validationData = dataset.load()

    charDF = pd.read_csv('data/char_data.csv')
    countsByClass = charDF.sort_values('Unicode_cat').Frequency.values
    totalNumSamples = charDF['Frequency'].sum()
    # beta = (totalNumSamples - 1) / totalNumSamples
    beta = 0.999
    classWeights = pd.DataFrame((1 - beta) / (1 - beta ** countsByClass))
    classWeights = classWeights.to_dict()[0]
    probs = countsByClass / totalNumSamples

    if setup.classifierName == 'ResNet18':
        classifier = ResNet18([dataConfig['classifierInputWidth'], dataConfig['classifierInputHeight'], 3],
                              numClasses=4206, outputBias=np.log(probs))
    elif setup.classifierName == 'ResNet34':
        classifier = ResNet34([dataConfig['classifierInputWidth'], dataConfig['classifierInputHeight'], 3],
                              numClasses=4206, outputBias=np.log(probs))
    elif setup.classifierName == 'ConvNetBaseline':
        classifier = ConvNetBaseline([dataConfig['classifierInputWidth'], dataConfig['classifierInputHeight'], 3],
                                     numClasses=4206, outputBias=np.log(probs))
    elif setup.classifierName == 'MobileNetV3':
        classifier = MobileNetV3([dataConfig['classifierInputWidth'], dataConfig['classifierInputHeight'], 3],
                                 numClasses=4206, outputBias=np.log(probs))
    else:
        raise NotImplementedError

    if args.classifierPath is not None:
        classifier.model = tf.keras.models.load_model(args.classifierPath)
    else:
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
        classifier.model.save('trained_models/classifier_' + str(datetime.datetime.now()).split(' ')[0] + '.hdf5')
        print('Last model saved')
        pass


def evaluateModel(setup):
    assert setup.classifierPath is not None and setup.detectorPath is not None
    with open('config/config.json') as fp:
        dataConfig = json.load(fp)
    centerNet = CenterNet([dataConfig['detectorInputWidth'], dataConfig['detectorInputHeight'], 3],
                          [3, 4, 6, 3], [64, 128, 256, 512], 1)
    centerNet.model = tf.keras.models.load_model(setup.detectorPath, custom_objects={
        "loss": centerNetLoss,
        "heatMapLoss": heatMapLoss,
        "sizeLoss": sizeLoss,
        "offsetLoss": offsetLoss
    })
    if setup.classifierName == 'ResNet18':
        classifier = ResNet18([dataConfig['classifierInputWidth'], dataConfig['classifierInputHeight'], 3],
                              numClasses=4206)
    elif setup.classifierName == 'ResNet34':
        classifier = ResNet34([dataConfig['classifierInputWidth'], dataConfig['classifierInputHeight'], 3],
                              numClasses=4206)
    elif setup.classifierName == 'ConvNetBaseline':
        classifier = ConvNetBaseline([dataConfig['classifierInputWidth'], dataConfig['classifierInputHeight'], 3],
                                     numClasses=4206)
    elif setup.classifierName == 'MobileNetV3':
        classifier = MobileNetV3([dataConfig['classifierInputWidth'], dataConfig['classifierInputHeight'], 3],
                                 numClasses=4206)
    else:
        raise NotImplementedError
    classifier.model = tf.keras.models.load_model(setup.classifierPath)

    result = pd.read_csv('data/train.csv')
    charDF = pd.read_csv('data/char_data.csv')
    zipObject = ZipFile('data/train_images.zip', 'r')
    listOfFileNames = zipObject.namelist()
    imageIds = []
    labels = []
    for fileName in tqdm(listOfFileNames):
        imageIds.append(fileName.split('.')[0])
        image = tf.image.decode_jpeg(zipObject.read(fileName))
        originalWidth = image.shape[1]
        originalHeight = image.shape[0]
        predictedBoxes = centerNet.predictBoundingBox(
            tf.reshape(tf.image.resize(image, [dataConfig['detectorInputWidth'],
                                               dataConfig['detectorInputHeight']]) / 255, (-1, 512, 512, 3)),
            confidenceThreshold=0.65, ioUThreshold=0.3)[0]
        prediction = ''
        if len(predictedBoxes) > 0:
            predictedBoxes = predictedBoxes.numpy()
            predictedBoxes[:, 1] = predictedBoxes[:, 1] * originalWidth
            predictedBoxes[:, 0] = predictedBoxes[:, 0] * originalHeight
            predictedBoxes[:, 3] = predictedBoxes[:, 3] * originalWidth
            predictedBoxes[:, 2] = predictedBoxes[:, 2] * originalHeight
            predictedBoxes = np.clip(predictedBoxes.astype(np.int32), 0, None)
            chars = []
            for i, box in enumerate(predictedBoxes):
                char = image[predictedBoxes[i, 0]:predictedBoxes[i, 2], predictedBoxes[i, 1]:predictedBoxes[i, 3], :]
                chars.append(tf.image.resize(char, [dataConfig['classifierInputWidth'],
                                                    dataConfig['classifierInputHeight']]) / 255)
            classPredictions = tf.argmax(classifier.predict(np.array(chars)), axis=1)
            for i, classPrediction in enumerate(classPredictions):
                predictedLabel = charDF[charDF['Unicode_cat'] == classPrediction]['Unicode'].values[0]
                prediction += predictedLabel + ' '
                prediction += str(predictedBoxes[i, 1] + (predictedBoxes[i, 3] - predictedBoxes[i, 1]) // 2) + ' '
                prediction += str(predictedBoxes[i, 0] + (predictedBoxes[i, 2] - predictedBoxes[i, 0]) // 2) + ' '
            prediction = prediction[:-1]
        labels.append(prediction)
    labels = np.array(labels).reshape(-1)
    imageIds = np.array(imageIds).reshape(-1)
    result['image_id'] = imageIds
    result['labels'] = labels
    score = kuzushijiF1(result, pd.read_csv('data/train.csv'))
    print(f"F1 score: {score}")


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
    parser.add_argument('--gpu', type=int, default=1,
                        help='Whether to perform traning on GPU (1) or not (0), if available')
    parser.add_argument('--minLr', type=float, default=1e-4, help='Minimum learning rate')
    parser.add_argument('--classifierName', type=str, default="ConvNetBaseline",
                        help="Classification model to train. Currently supporting ResNet18, ResNet34, ConvNetBaseline"
                             " or MobileNetV3. Use only with the --classifier flag")
    parser.add_argument('--detector', dest='detector', action='store_true')
    parser.add_argument('--classifier', dest='classifier', action='store_true')
    parser.add_argument('--classifierPath', type=str,
                        help='Path to saved classifier to resume training or evaluate.')
    parser.add_argument('--detectorPath', type=str,
                        help='Path to saved detector to resume training or evaluate.')
    parser.set_defaults(detector=False)
    parser.set_defaults(classifier=False)
    args = parser.parse_args()

    if args.gpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.mode == 'train' or args.mode == 'train_and_evaluate':
        if args.detector and not args.classifier:
            print("Training detector")
            trainDetector(args)
        elif args.classifier and not args.detector:
            print("Training classifier")
            trainClassifier(args)
        else:
            raise NotImplementedError

    if args.mode == 'evaluate' or args.mode == 'train_and_evaluate':
        evaluateModel(args)
