import pandas as pd
import numpy as np
import copy
import model_verification_utils as utils
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns


def get_distribution(X, y):
    """
    Returns the distribution of unique values and their counts for each feature in X,
    grouped by unique values of y.

    :param X: A 2D NumPy array representing the feature set.
    :param y: A 1D NumPy array representing the labels.
    :return: A dictionary where keys are unique labels from y, and values are dictionaries
             representing the distribution of each feature within that label.
    """
    unique_labels = np.unique(y)
    distribution = {label: {} for label in unique_labels}

    for label in unique_labels:
        indices = np.where(y == label)
        X_sub = X[indices]

        for i in range(X_sub.shape[1]):
            column = X_sub[:, i]
            unique, counts = np.unique(column, return_counts=True)
            distribution[label][f'Feature {i}'] = dict(zip(unique, counts))

    return distribution


def modify_distribution(X, y, fraction):
    unique_labels = np.unique(y)
    distribution = {label: {} for label in unique_labels}

    for label in unique_labels:
        indices = np.where(y == label)
        X_sub = X[indices]

        for i in range(X_sub.shape[1]):
            column = X_sub[:, i]
            unique, counts = np.unique(column, return_counts=True)

            # Modify the counts by shifting a fraction to the next neighbor
            for j in range(len(counts) - 1):  # Assuming the neighbor is the next value
                shift_amount = int(np.floor(counts[j] * fraction))
                counts[j] -= shift_amount
                counts[j + 1] += shift_amount

            distribution[label][f'Feature {i}'] = dict(zip(unique, counts))

    return distribution


def create_synthetic_dataset(modified_dist, original_y):
    synthetic_X = []
    synthetic_y = []

    # Calculate the number of samples for each label based on the original distribution
    labels, label_counts = np.unique(original_y, return_counts=True)
    label_distribution = dict(zip(labels, label_counts))

    for label in labels:
        # Keep the same number of instances
        num_samples_label = label_distribution[label]
        for _ in range(num_samples_label):
            sample = []
            for feature in modified_dist[label]:
                # Generate a feature value based on its distribution
                feature_values = list(modified_dist[label][feature].keys())
                feature_counts = list(modified_dist[label][feature].values())
                feature_value = np.random.choice(feature_values, p=feature_counts/np.sum(feature_counts))
                sample.append(feature_value)

            synthetic_X.append(sample)
            synthetic_y.append(label)

    # Shuffle the dataset
    synthetic_X, synthetic_y = shuffle(synthetic_X, synthetic_y)

    return np.array(synthetic_X), np.array(synthetic_y)


def create_D_prime(X, y, fraction):
    modified_dist = modify_distribution(X, y, fraction,)
    D_prime, y_prime = create_synthetic_dataset(modified_dist, y)
    return D_prime, y_prime


def create_low_acc_ds(X_train, y_train, shift_max, X_test, y_test, args, repeat):
    """
    Create datasets by applying shifts.
    """
    ori_dist = get_distribution(X_train, y_train)

    datasets = {}  # Dictionary to store datasets
    acc_arr = []
    for shift_percent in np.arange(start=0, stop=shift_max, step=0.1):
        shift_key = int(10 * shift_percent)  # Convert shift_percent to integer key

        if shift_key == 0:
            D, y = X_train, y_train
        else:
            D, y = create_D_prime(X_train, y_train, shift_percent)

        if repeat == True:
            for i in range(20):
                M_prime = utils.train_model(D, y, args)
                acc_arr.append([round(shift_percent, 2), M_prime.score(X_test, y_test)])
        else:
            M_prime = utils.train_model(D, y, args)
            acc_arr.append([round(shift_percent, 2), M_prime.score(X_test, y_test)])
        datasets[shift_key] = {'D': D, 'y': y}

    # Creating DataFrame
    df = pd.DataFrame(acc_arr, columns=['Shift Factor', 'Accuracy'])

    # Plotting with Seaborn
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Shift Factor', y='Accuracy', data=df, color='lightblue')

    # Calculate and plot the mean values
    means = df.groupby('Shift Factor')['Accuracy'].mean()
    sns.pointplot(x=means.index, y=means.values, color='darkblue', scale=0.5, join=False)

    plt.title('Boxplot of Model Accuracy for Different Shift Factors')
    plt.grid(True)
    plt.show()

    # Optionally, print the mean values
    print("Average Accuracies by Shift Factor:")
    print(means)

    return datasets, ori_dist

