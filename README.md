# Segmentation and classification of ancient Japanese characters

## Requirements

- Python 3.8
- TensorFlow 2.0

## Project Structure

    .
    ├── config/                 # Config files
    ├── data/                   # Dataset path
    ├── notebooks/              # Prototyping
    ├── scripts/                # Download dataset and miscelaneus scripts
    ├── trained_models/         # Trained weights
    ├── data_loader.py          # Data reader, preprocessing, batch iterator
    ├── main.py                 # Running an experiment (different modes below)
    ├── model.py                # Defines model      
    └── utils.py                # Miscelaneus useful functions

## Experiments mode

- `evaluate` : Evaluate on the evaluation data.
- `test` : Tests training, evaluating and exporting the estimator for a single step.
- `train` : Fit the estimator using the training data.
- `train_and_evaluate` : Interleaves training and evaluation.


## Usage

1. Install requirements.

    ```pip install -r requirements.txt```

2. Download raw data set and pre-process it to create a TensorFlow input pipeline using the file ```data_loader.py```:

    ```
    python scripts/download_dataset.py
    python data_loader.py --identifier --config default
    ```
    You should select the kind of model for which you intend to use the data with the flags ```--classifier``` or ```--identifier```. The parameters of the input pipeline should be specified in a config file in the folder ```./config``` (You can create your own set ups following the template specified in ```./config/default.py```).
    
3. Finally, train and/or evaluate a model:

    ```python main.py --identifier --config default --mode train_and_evaluate```
