import pandas as pd
import numpy as np
import os
from autogluon.tabular import TabularPredictor
import click


# Convert DNA sequence to one-hot matrix
def dna_to_onehot(seq, usem):
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
    default = [0] * len(next(iter(mapping.values())))
    onehot_encoded = np.array([mapping.get(base, default) for base in seq], dtype=np.int8)
    return onehot_encoded.flatten()
    

# Calculate alpha value of each sequence, -1 by default for reads without methylation information
def calculate_alpha(seq):
    meth_count = seq.count('M')
    unmeth_count = seq.count('C')
    total_count = meth_count + unmeth_count
    return meth_count / total_count if total_count > 0 else -1
    

def convert_sequence(file_path, usem=True):
    df = pd.read_csv(file_path, sep='\t', names=['chr', 'start', 'end', 'seq', 'tag', 'label'])

    df['alpha'] = df['seq'].apply(calculate_alpha)
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


def generate_df(train_file_path, test_file_path, usem):
    # Load the dataset and convert the sequence to one-hot encoding
    train_df = convert_sequence(train_file_path, usem)
    test_df = convert_sequence(test_file_path, usem)

    # Ensure the training and test sets have the same sequence length
    train_df, test_df = expand_df(train_df, test_df)
    
    return train_df, test_df


@click.command()
@click.option('--dataset', required=True, type=click.STRING, help='Path to the dataset')
def main(dataset):
    train = f"../data/benchmark/{dataset}/all_train_cov1.bed"
    test = f"../data/benchmark/{dataset}/all_test_cov1.bed"
    usem = True
    output = f"../results/benchmark/{dataset}"
    
    train_features, test_features = generate_df(train, test, usem)
    predictor = TabularPredictor.load(output)
    # Output predictions of test set
    test_proba = predictor.predict_proba(test_features)
    label_0, label_1 = test_proba.columns
    pred_output = pd.concat([test_features['label'], test_proba[test_proba.columns[0]], test_proba[test_proba.columns[1]]], axis=1)
    pred_output.columns = ['label', f'pred_{label_0}', f'pred_{label_1}']
    pred_output.to_csv(os.path.join(output, 'test_res.csv'), sep='\t', index=False)

if __name__ == '__main__':
    main()

