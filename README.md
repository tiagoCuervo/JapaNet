# Detection and classification of ancient Japanese Kuzushiji characters

# Table of Contents

- [Aim](#aim)
- [Requirements](#requirments)
- [Repository Structure](#repository-structure)
- [Project Architecture](#project-architecture)
- [Main Components](#main-components)
    - [download_data.py](#download_data.py)
    - [dataloader.py](#dataloader.py)
    - [model.py](#model.py)
    - [main.py](#main.py)
- [Suggested Usage](#suggested-usage)
- [Results](#results)
    - [Detection](#detection)
    - [Classification](#classification) 
- [References](#references)

# Aim

The main goal of this project has been to develop a model (models) that would perform detection and classification of ancient Japanese characters (Kuzushiji cursive script), in which classification consists of mapping the Kuzushiji characters to their modern Japanese counterparts. 



The main motivation behind the project choice has been to utilize artificial intelligence tools to contribute to a wider ongoing research aimed at making ancient Japanese culture and history more available to people[1]. Sources written in Kuzushiji cannot be read nor appreciated without appropriate translation by anyone except only a small number of experts. Being able to make the ancient Japanese heritage more accessible to a wider public seemed like a fantastic real-life application of Machine Learning.

Data for the project has been taken from the Kaggle competition[2] aimed at improving the current models developed for Kuzushiji recognition. 

# Requirements

- Python 3.8
- TensorFlow 2.0

# Repository Structure

    .
    ├── config/                 # Config files
    ├── data/                   # Dataset path
    ├── notebooks/              # Prototyping
    ├── scripts/                # Download dataset and miscelaneus scripts
    ├── trained_models/         # Trained weights
    ├── dataloader.py           # Data reader, preprocessing, batch iterator
    ├── download_dataset.py     # Data downloader
    ├── main.py                 # Running an experiment (different modes below)
    └── model.py                # Defines model
    

# Project Architecture

![Simplified architecture of the project](./figures/arch.png?raw=true)

# Main Components

## download_data.py

Script for downloading the data available on Kaggle website[2]. Zip files are downloaded directly to the data/ directory.

## dataloader.py

Script for unpacking the zipped data and creating TensorFlow.records input pipelines for the detection and classification tasks, as well as the config json file to be used later by `main.py`. Users need to specify the kind of model for which they intend to use the data using the flags (names are self-explanatory):

- `--detector`
- `--classifier`

`dataloader.py` has a set of default parameters to be saved in the config file, but accepts custom values through the appropriate flags. See --help for more information.

## model.py

Script containing the detection and classification models used in `main.py`. At the moment, detection is performed using the CenterNet[3] model only. For classification, users can use the `--classifierName` flag to choose one of the currently supported models: ConvNetBaseline (custom default), ResNet18[4] or MobileNetV3 Large[5].

## main.py

Script for running the detection and classification experiments. 

The following modes are accepted for both tasks through the `--mode` flag:

- `evaluate` : Evaluate on the evaluation data.
- `test` : Tests training, evaluating and exporting the estimator for a single step.
- `train` : Fit the estimator using the training data.
- `train_and_evaluate` : Interleaves training and evaluation.

# Suggested Usage

The suggested usage of the project's resources available here is as follows (the users are however free to use them at their will):

1. Install requirements.

    ```shell
    pip install -r requirements.txt
    ```

2. Download the raw data set[2]:

    ```shell
    python download_dataset.py
    ```
    
3. Unpack the zipped data and pre-process it to create a TensorFlow input pipeline and a config json file used by `main.py` for a desired task using `dataloader.py` (remember to specify the kind of model for which you intend to use the data using appropriate flag):

    ```shell
    python dataloader.py --detector
    ```

4. Finally, train the desired model using `main.py`:

    ```shell
    python main.py --detector --mode train --numEpochs 20 --gpu 1 --minLr 1e-4
    ```
    
The model hyperparameters should be supplied through appropriate flags. See --help for more information.

# Results

## Detection

Below we present sample images showing the results of our experiments regarding the detection task.

- learaning curves obtained from the training process of the CenterNet detection model:
![Learning curves from the CenterNet detection model](./figures/centernet_curves.png?raw=true)

- predicted positions of Kuzushiji characters on a sample page obtained from the trained CenterNet:

![Positions of characters on sample page predicted by CenterNet](./figures/positions.png?raw=true)

- predicted bounding boxes obtained from the trained CenterNet:

![Bounding Boxes generated with CenterNet](./figures/boxes.png?raw=true)


## Classification

Below we present sample images showing the results of our experiments regarding the classification task.

### Baseline ConvNet

Training of the baseline convolutional net has been performed with a constant learning rate of 0.001, sparse categorical cross-entropy loss, Adam optimizer, batch size of 512 and for 20 epochs.

**brief results?**

- sample learning curves obtained from the Baseline Convolutional classification model:

![Learning curves from the ConvNetBaseline classification model](./figures/convnet_curves.png?raw=true)


### ResNet18

The core of the MobileNetV3 Large with an additional dense layer of size 1024 before the output layer with suitable number of outputs (4206) has been used for the purposes of our experiments. The training process has been performed with a *reduce-on-plateau* learning schedule, sparse categorical cross-entropy loss, Adam optimizer, batch size of 256 and for 100 epochs.

**brief results?**

- sample learning curves obtained from the ResNet18 classification model:

![Learning curves from the uniweighted ResNet18 classification model](./figures/resnet_unweighted.png?raw=true)


### MobileNetV3

The core of the MobileNetV3 Large[5] with an additional dense layer of size 1024 before the output layer with suitable number of outputs (4206) has been used for the purposes of our experiments. The training process has been performed with a *reduce-on-plateau* learning schedule, sparse categorical cross-entropy loss, Adam optimizer, batch size of 256 and for 100 epochs

**brief results?**

- sample learning curves obtained from the MobilenetV3 classification model:

**image with learning curves**


- **table with results here?**

# References

- [1] A. Lamb, *How Machine Learning Can Help Unlock the World of Ancient Japan*, The Gradient https://thegradient.pub/machine-learning-ancient-japan/ (2019), last accessed 15.03.2021

- [2] *Kuzushiji Recognition*, URL: https://www.kaggle.com/c/kuzushiji-recognition/data, last accessed 18.04.2021

- [3] Zhou et al. [*Objects as Points*](https://arxiv.org/abs/1904.07850), 	Computer Vision and Pattern Recognition (2019)

- [4] He, Kaiming et al. [*Deep residual learning for image recognition*](https://arxiv.org/abs/1512.03385), Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2016)

- [5] A. Howard et al, [*Searching for MobileNetV3*](https://arxiv.org/abs/1905.02244), IEEE/CVF International Conference on Computer Vision (ICCV) (2019)
