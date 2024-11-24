# Autogluon read split

This document provides instructions for training the `Auto-gluon` model, an ensembled machine learning workflow designed for fast and easy ML deployment, on methylation information based read-split tasks.

## Requirements
To use the scripts, ensure the following dependencies are installed:

```bash
pip install pandas numpy click autogluon
```


## Usage

1. Place your raw methylation data in the expected MQ format.
2. Run the command below for model training and test set prediction.
```bash
python at.py --train <path_to_training_data> --test <path_to_test_data> --usem <bool> --output <path_to_output_directory>
```
This step will encode the sequence to onehot format automatically and run the autogluon.

