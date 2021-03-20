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


## Usage Example

Install requirements.

```pip install -r requirements.txt```

Then, download raw dataset and pre-process it to create a TF Dataset. You should select the kind of model for which you intend to use the data with the flags ```--classifier``` or ```--identifier```. You can create your own config files in the folder ```./config```, following the template specified in ```./config/default.py```.

```
python scripts/download_dataset.py
python data_loader.py --config default --identifier
```

Finally, start train and evaluate model

```python main.py --config TODO --mode train_and_evaluate```

