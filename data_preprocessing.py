import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

def num_to_cat(df, column, num_bins):
    """
    Discretize a numerical column into specified number of bins.
    """
    bin_range = (df[column].min(), df[column].max())
    bin_edges = np.linspace(bin_range[0], bin_range[1], num_bins + 1)
    df[column] = pd.cut(df[column], bins=bin_edges, labels=range(1, num_bins + 1), include_lowest=True)
    return df[column]

def encode_X(X):
    """
    Apply ordinal encoding to features X.
    """
    enc = OrdinalEncoder()
    X_encoded = enc.fit_transform(X)
    return X_encoded.astype('int64')

def encode_y(y):
    """
    Apply label encoding to the target variable y.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded.astype('int64')

def load_and_preprocess_data(dataset_dir, dataset_name, dataset_header, feature_file=None, label_file=None):
    """
    Load and preprocess the dataset.
    """
    if dataset_dir is not None:
        if dataset_name == 'adult':
            df = pd.read_csv(dataset_dir, header=None)
            # Balance dataset by labels
            lower_bound = df.groupby(df.iloc[:, -1]).size().min()
            df = df.groupby(df.iloc[:, -1]).sample(lower_bound)
        else:
            if dataset_header is not None:
                df = pd.read_csv(dataset_dir)
            else:
                df = pd.read_csv(dataset_dir, header=None)

        df.dropna(inplace=True)

        for column in df.columns:
            if df[column].dtype != 'object':
                num_bins = 15 if df[column].nunique() > 10 else df[column].nunique()
                df[column] = num_to_cat(df, column, num_bins)

    elif feature_file and label_file:
        df = pd.read_csv(feature_file, header=None)
        df['label'] = pd.read_csv(label_file, header=None)
        df = df.sample(n=1000)  # Assuming you want to sample 1000 rows

    else:
        raise ValueError("Either dataset directory or feature/label files must be provided.")

    return df

# Additional preprocessing functions can be added here
