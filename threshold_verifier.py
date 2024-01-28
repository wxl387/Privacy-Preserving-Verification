import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def find_kde_peak_and_height(kde_values, x_range):
    peaks, properties = find_peaks(kde_values, height=0)
    if len(peaks) > 0:
        peak_index = np.argmax(kde_values[peaks])  # Index of the peak with the highest KDE value
        peak = x_range[peaks[peak_index]]  # Peak position
        height = properties['peak_heights'][peak_index]  # Peak height
        return peak, height
    else:
        return None, None


def find_intersections(kde1, kde2, x_range):
    intersections = []
    for x in x_range:
        if np.isclose(kde1(x), kde2(x), atol=1e-5):
            intersections.append(x)
    return intersections


def calculate_false_rates(intersection, kde1, kde2, x_range):
    # Calculate areas under the curves to the left and right of the intersection
    area1_left = kde1.integrate_box_1d(x_range[0], intersection)
    area2_left = kde2.integrate_box_1d(x_range[0], intersection)
    area1_right = kde1.integrate_box_1d(intersection, x_range[-1])
    area2_right = kde2.integrate_box_1d(intersection, x_range[-1])

    # False positive rate is area under kde2 to the left of the intersection
    # False negative rate is area under kde1 to the right of the intersection
    false_positive_rate = area2_left
    false_negative_rate = area1_right
    return false_positive_rate + false_negative_rate


def optimize_threshold(kde1, kde2, x_range):
    intersections = find_intersections(kde1, kde2, x_range)
    if not intersections:
        return None

    # Calculate false rates for each intersection and choose the best one
    best_threshold = min(intersections, key=lambda x: calculate_false_rates(x, kde1, kde2, x_range))
    return best_threshold


def calculate_threshold(kde1, kde2, x_range):
    kde1_values = kde1(x_range)
    kde2_values = kde2(x_range)

    peak1, height1 = find_kde_peak_and_height(kde1_values, x_range)
    peak2, height2 = find_kde_peak_and_height(kde2_values, x_range)

    if peak1 is not None and peak2 is not None:
        if height1 > 2 * height2 or height2 > 2 * height1:
            return np.mean([peak1, peak2])  # Return only the mean value
        else:
            intersections = find_intersections(kde1, kde2, x_range)
            if intersections:
                if len(intersections) == 1:
                    return intersections[0]  # Return the single intersection
                else:
                    return optimize_threshold(kde1, kde2, x_range)  # Return the balanced threshold
            else:
                print("no intersection found")
                return None
    else:
        if peak2 is not None:
            print("peak1 not found")
        elif peak1 is not None:
            print("peak2 not found")
        else:
            print("both peak not found")

        peak1 = custom_peak_finder(kde1_values, x_range, label=0)  # Replace label=0 with actual label
        print("Peak for label=0:", peak1)

        return np.mean([peak1, peak2])


def kde_difference(x, kde1, kde2):
    """
    Compute the difference between two KDE functions at a given point.
    """
    return kde1(x) - kde2(x)


def find_intersections(kde1, kde2, x_range):
    intersections = []
    for x in x_range:
        if np.isclose(kde1(x), kde2(x), atol=1e-5):
            intersections.append(x)
    return intersections


def custom_peak_finder(kde_values, x_range, label):
    if label == 0:
        # Custom logic for label=0
        # Assuming peak is around 0.0, find max value in a small range
        near_zero_indices = np.where(x_range <= 0.05)  # Adjust range as necessary
        peak_index = np.argmax(kde_values[near_zero_indices])
        peak = x_range[near_zero_indices][peak_index]
        return peak
    else:
        # Default peak finding for other labels
        peaks, _ = find_peaks(kde_values, prominence=0.01)  # Adjust parameters as needed
        if peaks.size > 0:
            peak_index = np.argmax(kde_values[peaks])
            return x_range[peaks[peak_index]]
        else:
            return None


def plot_kde_and_find_peaks(data, bw_method='silverman', x_range=np.linspace(0, 1, 10000)):
    kde = gaussian_kde(data, bw_method=bw_method)
    kde_values = kde(x_range)
    plt.plot(x_range, kde_values, label='KDE')

    # Trying to find peaks
    peaks, properties = find_peaks(kde_values, height=0, prominence=0.01)  # Adjust parameters as needed
    if len(peaks) > 0:
        plt.plot(x_range[peaks], kde_values[peaks], "x", label='Peaks')
    else:
        print("No peaks found")

    plt.legend()
    plt.show()


def Thresh_Verifier_training(dists):
    # Flatten the distance data
    thresh_dists = []
    for iteration in dists:
        for case in iteration:
            for batch in case:
                distances = batch[:, 0].reshape(-1, 1)
                labels = batch[:, 1]
                scaled_batch = np.hstack((distances, labels.reshape(-1, 1)))
                average_per_batch = np.mean(scaled_batch, axis=0)
                thresh_dists.append(average_per_batch)

    # Convert to NumPy array
    scaled_averaged_dists_array = np.array(thresh_dists)

    # Separate the scaled distances and labels
    scaled_distances = scaled_averaged_dists_array[:, 0]
    labels = scaled_averaged_dists_array[:, 1]

    # Finding unique labels
    unique_labels = np.unique(labels)

    # Create KDEs
    kdes = {}
    for label in unique_labels:
        distances_for_label = scaled_distances[labels == label]
#         plot_kde_and_find_peaks(distances_for_label)
        kdes[label] = gaussian_kde(distances_for_label)

    # Define a range for searching peaks in KDE
    x_range = np.linspace(0, 1, 10000)

    # Find optimal separation points
    thresholds = {}
    for label in unique_labels:
        if label > 0:
            threshold = calculate_threshold(kdes[0], kdes[label], x_range)
            if threshold is not None:
                thresholds[(0, label)] = threshold

    print(thresholds)

    # Convert thresholds to an array for convenience
    threshold_array = np.array([value for key, value in thresholds.items()])

    # Plot KDEs and Thresholds
    plt.figure(figsize=(10, 6))
    for label in kdes:
        plt.plot(x_range, kdes[label](x_range), label=f'Label {label}')

    for i, ((label1, label2), threshold) in enumerate(thresholds.items()):
        plt.axvline(x=threshold, color='k', linestyle='--', lw=1)

        # Adjust vertical position for each label
        vertical_position = plt.ylim()[1] - (i * 0.05 * plt.ylim()[1])

        # Annotation
        plt.annotate(f'Threshold {label1}-{label2}', xy=(threshold, vertical_position),
                     xytext=(threshold, vertical_position + 0.05),
                     arrowprops=dict(facecolor='black', arrowstyle="->"),
                     ha='right')

    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('KDEs and Thresholds')
    plt.legend()
    plt.show()

    return threshold_array


def Thresh_Verifier_testing(dist, thresholds, spec_labels, test_frac):
    # Convert dist to a NumPy array if it's not already
    dist_array = np.array(dist)

    # Calculate the average for both distance and label within each batch
    avg_batch_data = np.mean(dist_array, axis=2)

    # Flatten the averaged data for further processing
    avg_batch_data_flat = avg_batch_data.reshape(-1, 2)

    # Specified labels to filter
    specified_labels = np.array(spec_labels)

    # Extract instances that have labels in specified_labels
    extracted_data = avg_batch_data_flat[np.isin(avg_batch_data_flat[:, -1], specified_labels)]

    # Number of samples to draw from each label group based on the fraction
    num_samples = int(len(extracted_data) * test_frac)

    # Sampling indices
    sampled_indices = np.random.choice(range(len(extracted_data)), num_samples, replace=False)
    sampled_data = extracted_data[sampled_indices]

    # Separate the scaled distances and labels
    dist_X_avg = sampled_data[:, 0]
    dist_y_avg = sampled_data[:, 1]

    # Extract the relevant subset of thresholds based on spec_labels
    relevant_thresholds = [thresholds[label - 1] for label in spec_labels if label != 0]

    thresh_pred = []
    for val in dist_X_avg:
        # Iterate over the relevant thresholds to find the corresponding label
        for index, threshold in enumerate(relevant_thresholds):
            if val <= threshold:
                predicted_label = spec_labels[index]
                break
        else:
            predicted_label = spec_labels[-1]

        thresh_pred.append(predicted_label)

#     # Plotting the frequency plot for testing distances with color based on true label
#     plt.figure(figsize=(10, 6))
#     for label in specified_labels:
#         label_dists = dist_X_avg[dist_y_avg == label]
#         plt.hist(label_dists, bins=30, alpha=0.7, label=f'Label {label}')

#     for threshold in relevant_thresholds:
#         plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold {threshold:.2f}')

#     plt.xlabel('Distance')
#     plt.ylabel('Frequency')
#     plt.title('Frequency Plot of Testing Distances by Label')
#     plt.legend()
#     plt.show()

    return thresh_pred, dist_y_avg
