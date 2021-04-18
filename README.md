# Segmentation and classification of ancient Japanese Kuzushiji characters

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
  - [utils.py](#utils.py)
- [Suggested Usage](#suggested-usage)
- [Results](#results)
- [References](#references)

# Aim

The main goal of this project has been to develop a model (models) that would perform detection and classification of ancient Japanese characters (Kuzushiji cursive script), in which classification consists of mapping the Kuzushiji characters to their modern Japanese counterparts. 

![](https://github.com/tiagoCuervo/JapaNet/tree/main/figures/boxes.png)

The main motivation behind the project choice has been to utilize the artificial intelligence tools to contribute to a wider ongoing research aimed at making ancient Japanese culture and history more available to people[1]. Sources written in Kuzushiji cannot be read nor appreciated without appropriate translation by anyone except only a small number of experts. Being able to make the ancient Japanese heritage more accessible to a wider public seemed like a fantastic real-life application of Machine Learning.

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
    ├── model.py                # Defines model 
    └── utils.py                # Miscelaneus useful functions
    

# Project Architecture
![](https://github.com/tiagoCuervo/JapaNet/tree/main/figures/arch.png)


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

## utils

Script containing miscellaneous utility functions.

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



# References

- [1] A. Lamb, *How Machine Learning Can Help Unlock the World of Ancient Japan*, The Gradient https://thegradient.pub/machine-learning-ancient-japan/ (2019), last accessed 15.03.2021

- [2] *Kuzushiji Recognition*, URL: https://www.kaggle.com/c/kuzushiji-recognition/data, last accessed 18.04.2021

- [3] K.Duan et al. *CenterNet: Keypoint Triplets for Object Detection*, Computer Vision and Pattern Recognition (2019)

- [4] He, Kaiming et al. *Deep residual learning for image recognition*, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2016)

- [5] A. Howard et al, *Searching for MobileNetV3*, IEEE/CVF International Conference on Computer Vision (ICCV) (2019)
