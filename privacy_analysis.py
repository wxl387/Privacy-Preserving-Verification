# privacy_analysis.py
import numpy as np
import pandas as pd
from scipy.spatial import distance

#
# def flip_data(val, dist, label, attr):
#     val_new = val
#     count = 0
#     while(val_new == val and count < 100):
#         count += 1
#         unique_val = dist[label][attr][0]
#         counts = dist[label][attr][1]
#         prob = counts / sum(counts)
#         val_new = np.random.choice(unique_val, size=1, p=prob)
#     return val_new
#
#
# def add_noise(X, y, eps, dist):
#     X_new = X.copy()
#     for j in range(X.shape[1]):
#         unique_val = np.unique(X[:, j])
#         p = np.exp(eps) / (len(unique_val) - 1 + np.exp(eps))
#
#         for i in range(X.shape[0]):
#             rand_val = np.random.uniform(0, 1)
#             if rand_val > p:
#                 X_new[i, j] = flip_data(X_new[i, j], dist, int(y[i]), j)
#     return X_new


def create_D_hat(X, y, eps, distribution):
    D_hat = apply_randomized_response(X, y, eps, distribution)
    return D_hat


def apply_randomized_response(X, y, eps, distribution):
    X_randomized = X.copy()
    num_features = X.shape[1]
    num_samples = X.shape[0]

    for j in range(num_features):
        for i in range(num_samples):
            current_label = y[i]
            # Get the distribution for the current feature and label
            feature_distribution = distribution[current_label][f'Feature {j}']
            unique_values, counts = zip(*feature_distribution.items())
            total_counts = sum(counts)
            probabilities = [count / total_counts for count in counts]

            # Calculate flip probability based on epsilon
            flip_probability = np.exp(eps) / (len(unique_values) - 1 + np.exp(eps))

            # Flip the value with the given probability
            if np.random.rand() > flip_probability:
                X_randomized[i, j] = np.random.choice(unique_values, p=probabilities)

    return X_randomized


def data_sampling(X_train, y_train, X_test, y_test):
    DF_train = pd.DataFrame(X_train).copy()
    DF_train['label'] = y_train
    DF_test = pd.DataFrame(X_test).copy()
    DF_test['label'] = y_test

    case = DF_train.sample(frac=0.1)
    # not_case = DF[~DF.index.isin(case.index)]
    # control = not_case.sample(n=int(case.shape[0]/2))
    control = DF_test.sample(n=int(case.shape[0]/4))
    target_users = case.sample(n=control.shape[0])
    return case.to_numpy(), control.to_numpy(), target_users.to_numpy()


def hamming_distance(arr_1, arr_2):
    # Extracting features excluding the last column (label)
    features_1 = arr_1[:, :-1]
    features_2 = arr_2[:, :-1]

    # Initialize an array to store minimum distances
    min_dists = np.zeros(features_1.shape[0])

    # Calculate hamming distances
    for i, feature_set_1 in enumerate(features_1):
        # Compute hamming distance between feature_set_1 and each row in features_2
        dists = np.array([distance.hamming(feature_set_1, feature_set_2) for feature_set_2 in features_2])
        # Store the minimum distance
        min_dists[i] = dists.min()

    # Create DataFrame from the result
    df_hamin_dist = pd.DataFrame(min_dists, columns=['distance'])

    return df_hamin_dist


def power_cal(case, control, target_users):
    dist_control = hamming_distance(control, case)
    threshold = dist_control.distance.quantile(q=0.05)
    dist_target = hamming_distance(target_users, case)
    count = dist_target[dist_target.distance < threshold].shape[0]
    return count / dist_target.shape[0]


def privacy_eval(D, y, eps, true_dist, X_test, y_test):
    power_arr = []
    for _ in range(5):
        case, control, target_users = data_sampling(D, y, X_test, y_test)
        target_users_new = apply_randomized_response(target_users[:, :-1], target_users[:, -1], eps, true_dist)
        power = power_cal(case[:, :-1], control[:, :-1], target_users_new)
        power_arr.append(power)
    return np.mean(power_arr)
