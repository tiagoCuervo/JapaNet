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

Then, Download dataset and pre-processing.

```
python scripts/download_dataset.py
python data_loader.py --config TODO
```

Finally, start train and evaluate model

```python main.py --config TODO --mode train_and_evaluate```

