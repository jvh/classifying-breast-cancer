#########################################################################
# Contains 'helper' functions which are common across other classifiers.#
#                                                                       #
# Author: Jonathan Harper                                               #
#########################################################################

import pandas as pd
import numpy as np

# The features, without the unecessary features + classification
FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
               'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
               'standard_error_radius', 'standard_error_texture', 'standard_error_perimeter', 'standard_error_area',
               'standard_error_smoothness', 'standard_error_compactness', 'standard_error_concavity',
               'standard_error_concave_points', 'standard_error_symmetry', 'standard_error_fractal_dimension',
               'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
               'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry',
               'worst_fractal_dimension'
            ]

def calculate_accuracy(predictions_data, data_class):
    """
    Calculates the accuracy between the predictions and the actual classification

    :param predictions_data: The latest predictions, based on the latest weights, made
    :param data_class: The genuine classification of the patient
    :return: The accuracy percentage
    """
    rounded_predictions = np.round(predictions_data)

    rounded_predictions_list = rounded_predictions.tolist()
    class_list = data_class.values.tolist()

    num_same = 0
    for i in range(len(class_list)):
        if class_list[i] == rounded_predictions_list[i]:
            num_same += 1

    accuracy = num_same / len(class_list)
    return accuracy


def remove_highly_correlated_features(data_wo_class):
    """
    This method removes any highly correlated features, removing features with a correlation of >= CORRELATION_THRESHOLD

    :param data_wo_class: This is the data with the 'class'/'diagnosis' removed
    """
    # Create correlation matrix
    corr_matrix = data_wo_class.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    for col in to_drop:
        data_wo_class.drop(col, axis=1, inplace=True)

    """ Show feature correlation in graph format """
    # sb.heatmap(data_without_class.corr())
    # plt.show()


def read_csv_and_prep():
    """
    Reads the .csv file and preps it for further processing

    :return data_formatted: This is the data_without_class stacked with the intercept_template
    :return data_without_class: The data file without the classification
    :return data_class_only: The classification in a vector
    :return data_file: The data including the classification
    """
    all_col_names = ['id', 'diagnosis'] + FEATURES
    data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
        header=None,
        names=all_col_names)

    # Replace 'B' and 'M' with 0 and 1 respectively
    data['diagnosis'].replace('B', 0, inplace=True)
    data['diagnosis'].replace('M', 1, inplace=True)

    data_file = data.drop('id', axis=1)

    # The existing data without the 'Class' column
    data_without_class = data_file.drop('diagnosis', axis=1)
    data_class_only = data_file['diagnosis']

    remove_highly_correlated_features(data_without_class)

    # Creating a vector essentially which is populated with all 1's. This serves as a placeholder to allow for the dot
    # product of 1 and the intercept value
    intercept_template = np.ones((data_without_class.shape[0], 1))
    # Stack together the template and data
    data_formatted = np.hstack((intercept_template, data_without_class))

    return data_formatted, data_without_class, data_class_only, data_file