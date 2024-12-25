import os
import click

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score, ConfusionMatrixDisplay, confusion_matrix

from autogluon.tabular import TabularPredictor


def dna_to_onehot(seq, usem):
    """
    Convert a DNA sequence into a one-hot encoded matrix representation.

    Parameters:
    ----------
    seq : str
        DNA sequence as a string (e.g., "ACGTMN").
        
    usem : bool
        Flag to determine the mapping for methylation base 'M':
        - If True, 'M' gets its own one-hot representation (5th position).
        - If False, 'M' is treated like 'C' in the one-hot encoding.

    Returns:
    -------
    np.ndarray
        Flattened one-hot encoded representation of the input DNA sequence.
        Each base is represented by a list of integers, and the entire sequence
        is flattened into a 1D numpy array of dtype int8.
        
    Example:
    -------
    dna_to_onehot("ACGT", usem=True)
    # Output: array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    """

    # Define the one-hot encoding scheme based on usem flag
    if usem:
        mapping = {'A': [1, 0, 0, 0, 0, 0],
                   'C': [0, 1, 0, 0, 0, 0],
                   'G': [0, 0, 1, 0, 0, 0],
                   'T': [0, 0, 0, 1, 0, 0],
                   'M': [0, 0, 0, 0, 1, 0],
                   'N': [0, 0, 0, 0, 0, 1]}
    else:
        mapping = {'A': [1, 0, 0, 0, 0],
                   'C': [0, 1, 0, 0, 0],
                   'G': [0, 0, 1, 0, 0],
                   'T': [0, 0, 0, 1, 0],
                   'M': [0, 1, 0, 0, 0],
                   'N': [0, 0, 0, 0, 1]}

    # Default vector for unknown bases (all zeros)
    default = [0] * len(next(iter(mapping.values())))

    # Convert each base to one-hot encoding, defaulting to zeros for unknowns
    onehot_encoded = np.array([mapping.get(base, default) for base in seq], dtype=np.int8)

    # Flatten the one-hot matrix to a 1D array and return
    return onehot_encoded.flatten()
    

# Calculate alpha value of each sequence, -1 by default for reads without methylation information
def calculate_alpha(seq):
    meth_count = seq.count('M')
    unmeth_count = seq.count('C')
    total_count = meth_count + unmeth_count
    return meth_count / total_count if total_count > 0 else -1
    

def convert_sequence(file_path, usem=True):
    """
    Convert a DNA sequence file in MQ format into a onehot-encoded DataFrame.

    This function reads a tab-separated file containing DNA sequences, applies 
    one-hot encoding to each sequence, and calculates the alpha value for each entry. 
    The resulting DataFrame includes categorical labels, the alpha feature, and the one-hot 
    encoded sequences.

    Parameters:
    ----------
    file_path : str
        Path to the input file containing DNA sequences in MQ format. 
        The file is expected to have six columns: 
        ['chr', 'start', 'end', 'seq', 'tag', 'label'].
    
    usem : bool, optional
        Whether to use the 'M' base as a unique one-hot vector. If True, 'M' 
        gets a separate encoding. Defaults to True.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing:
        - 'label': Categorical column for sequence labels.
        - 'alpha': Calculated feature from the DNA sequence.
        - One-hot encoded representation of the DNA sequences as multiple columns.
    """
    
    # Read the input file of MQ format
    df = pd.read_csv(file_path, sep='\t', names=['chr', 'start', 'end', 'seq', 'tag', 'label'])

    # Calculate alpha value for each sequence
    df['alpha'] = df['seq'].apply(calculate_alpha)

    # Apply one-hot encoding to the 'seq' column
    df['seq'] = df['seq'].apply(lambda x: dna_to_onehot(x, usem))
    
    onehot_df = pd.DataFrame(df['seq'].tolist(), index=df.index)
    onehot_df = onehot_df.fillna(0).astype(int)

    df['label'] = df['label'].astype('category')
    
    # Combine features and return the final DataFrame
    return pd.concat([df[['label', 'alpha']], onehot_df], axis=1)


def expand_df(df1, df2):
    # Determine the maximum number of columns
    max_cols = max(df1.shape[1], df2.shape[1])

    # Expand df1 if needed
    if df1.shape[1] < max_cols:
        df1 = pd.concat(
            [df1, pd.DataFrame(0, index=df1.index, columns=range(df1.shape[1], max_cols))],
            axis=1
        )

    # Expand df2 if needed
    if df2.shape[1] < max_cols:
        df2 = pd.concat(
            [df2, pd.DataFrame(0, index=df2.index, columns=range(df2.shape[1], max_cols))],
            axis=1
        )

    return df1, df2


def generate_df(train_file_path, 
                val_file_path=None, 
                test_file_path=None, 
                usem=False):
    """
    Generate dataframes for training, validation, and testing, ensuring that all datasets have the same sequence length.
    
    Args:
        train_file_path (str): Path to the training data file.
        val_file_path (str, optional): Path to the validation data file. If not provided, only the training and testing datasets will be generated.
        test_file_path (str): Path to the test data file.
        usem (bool): Whether to handle the one-hot matrix for M and C in a different way.
    
    Returns:
        tuple: Returns (train_df, val_df, test_df) if validation data is provided, otherwise returns (train_df, test_df).
    """
    # Load the training data
    train_df = convert_sequence(train_file_path, usem)

    # If validation data path exists, load the validation data
    val_df = None
    if val_file_path:
        val_df = convert_sequence(val_file_path, usem)

    # Load the test data
    test_df = convert_sequence(test_file_path, usem)

    # Ensure the training and test datasets have the same sequence length
    train_df, test_df = expand_df(train_df, test_df)

    # If validation data exists, ensure it has the same sequence length as training and test datasets
    if val_df is not None:
        train_df, val_df = expand_df(train_df, val_df)
        val_df, test_df = expand_df(val_df, test_df)

    return train_df, val_df, test_df


def display_attributes(cls):
    """
    A decorator to modify the __str__ method of the class to display its attributes.
    """
    def __str__(self):
        attrs = '\n'.join(f"############ {k} ############\n{v}\n" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}(\n{attrs})"
    cls.__str__ = __str__
    return cls


@display_attributes
class ModelRes():
    def __init__(self, name, trues, scores, reverse_label=0):
        self.name = name
        self.reverse_label = reverse_label
        self.trues = abs(self.reverse_label - trues)
        self.scores = abs(self.reverse_label - scores)
        # ROC values
        self.fpr, self.tpr, self.roc_thresholds = roc_curve(self.trues, self.scores)
        self.roc_auc = auc(self.fpr, self.tpr)
        self.j_scores = self.tpr - self.fpr
        self.best_index = np.argmax(self.j_scores)
        self.best_threshold_j = self.roc_thresholds[self.best_index]
        # PRC values
        self.precision, self.recall, self.pr_thresholds = precision_recall_curve(y_true=self.trues, probas_pred=self.scores, pos_label=1)
        self.pr_auc = average_precision_score(self.trues, self.scores)
        self.f1_scores = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        self.best_index = np.argmax(self.f1_scores)
        self.best_threshold_f1 = self.pr_thresholds[self.best_index]
        

@click.command()
@click.option('--train', required=True, type=click.Path(exists=True), help='Training set')
@click.option('--validation', required=False, default=None, type=click.Path(exists=True), help='Validation set')
@click.option('--test', required=True, type=click.Path(), help='Test set')
@click.option('--target', required=True, type=click.STRING, help='Target label')
@click.option('--background', required=True, type=click.STRING, help='Background label')
@click.option('--usem', required=True, type=click.BOOL, help='Set true to address M and C differently in the one-hot matrix')
@click.option('--output', required=True, type=click.Path(), help='Directory to save outputs')
def main(train, validation, test, target, background, usem, output):
    os.makedirs(output, exist_ok=True)
    
    # Generate feature dataframe from original MQ format file
    train_features, validation_features, test_features = generate_df(train, validation, test, usem)

    # Set target(1) and background(0) label
    for df in [train_features, validation_features, test_features]:
        if df is not None:
            df['label'] = df['label'].map({background: 0, target: 1})
    
    # Train step
    if validation_features is None:
        predictor = TabularPredictor(label='label', eval_metric='roc_auc', path=output).fit(
            train_data=train_features, 
            presets='best_quality')
    else:
        predictor = TabularPredictor(label='label', eval_metric='roc_auc', path=output).fit(
            train_data=train_features, 
            tuning_data=validation_features,
            use_bag_holdout=True,
            presets='best_quality')

    # Test step
    performance = predictor.evaluate(test_features)

    # Output model performance
    performance_file = os.path.join(output, 'performance.txt')
    with open(os.path.join(output, 'performance.txt'), 'w') as f:
        f.write("Model Performance:\n")
        for key, value in performance.items():
            f.write(f"{key}: {value}\n")
        f.write("\nLeaderboard:\n")
        f.write(predictor.leaderboard(test_features).to_string())

    print(f"Model performance and leaderboard have been saved to '{performance_file}'.")
    
    # Output predictions of test set
    test_proba = predictor.predict_proba(test_features)

    pred_output = pd.concat([test_features['label'], test_proba[test_proba.columns[0]], test_proba[test_proba.columns[1]]], axis=1)
    pred_output.columns = ['label', f'pred_{test_proba.columns[0]}', f'pred_{test_proba.columns[1]}']
    pred_output.to_csv(os.path.join(output, 'test_res.csv'), sep='\t', index=False)
    
    print(ModelRes(name='ag', trues=pred_output['label'].astype(int), scores=pred_output['pred_1']))

if __name__ == '__main__':
    main()

