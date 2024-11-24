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
@click.option('--train', required=True, type=click.Path(exists=True), help='Training set')
@click.option('--test', required=True, type=click.Path(), help='Test set')
@click.option('--usem', required=True, type=click.BOOL, help='Set true to address M and C differently in the one-hot matrix')
@click.option('--output', required=True, type=click.Path(), help='Directory to save outputs')
def main(train, test, usem, output):
    os.makedirs(output, exist_ok=True)
    
    # Generate feature dataframe from original MQ format file
    train_features, test_features = generate_df(train, test, usem)
    
    predictor = TabularPredictor(label='label', eval_metric='acc', path=output).fit(train_data=train_features, presets='best_quality')
    
    performance = predictor.evaluate(test_features)

    # Log model performance
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
    pred_output.columns = ['label', 'pred_0', 'pred_1']
    pred_output.to_csv(os.path.join(output, 'test_res.csv'), sep='\t', index=False)
    

if __name__ == '__main__':
    main()

