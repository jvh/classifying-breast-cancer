#########################################################################
# k-nearest neighbour technique for classifying breast cancer.          #
#                                                                       #
# Author: Jonathan Harper                                               #
#########################################################################

import numpy as np
import helper_functions as func

# Maximum number of closest-proximity neighbours to consider
MAX_NUM_NEIGHBOURS = 3
# This represents the percentage split of the data into training/testing. 0.7 represents a 70:30 split of training and
# testing data
TRAINING_SPLIT = 0.8


def load_data():
    """
    Loads the data and splits it into training_data and unseen_data (for testing)
    :return training_data: The data which is used to train the model (the data we 'know')
    :return unseen_data: The data which hasn't been seen by the model (we're predicting the classification)
    """
    _, _, _, data = func.read_csv_and_prep()

    # Splitting the dataframe up into train/test sets (and randomly selecting values for it)
    from sklearn.model_selection import train_test_split
    training_data, unseen_data = train_test_split(data, test_size=1-TRAINING_SPLIT)

    return training_data, unseen_data


def get_neighbours(training_data, unseen_data_row):
    """
    Finding the closest MAX_NUM_NEIGHBOURS from the unseen_data_row point

    :param training_data: The 'known' and existing data
    :param unseen_data_row: The unseen data point that we are attempting to predict classification for
    :return: the MAX_NUM_NEIGHBOURS closest neighbours
    """
    from copy import deepcopy
    training_with_distances = deepcopy(training_data)

    # Working out the euclidean distances between the unseen data point and all of the existing training data points
    training_with_distances['distance'] = training_with_distances[func.FEATURES].\
        sub(np.array(unseen_data_row)).pow(2).sum(1).pow(0.5)

    # Sorting the dataframe based on the distances in ascending order
    training_with_distances.sort_values(by='distance', inplace=True)
    # Selecting the closest MAX_NUM_NEIGHBOURS
    neighbours = training_with_distances.head(n=MAX_NUM_NEIGHBOURS)

    return neighbours


def predict_classification(neighbours):
    """
    Given the closest neighbours selected, pick the majority classification for the unseen data point
    :param neighbours: The closest MAX_NUM_NEIGHBOURS neighbours in proximity
    :return: The predicted classification (str)
    """
    # Counting the number of instances of each classification within the dataframe
    number_of_class_instances = neighbours.groupby(['diagnosis'], sort=False).size().reset_index(name='count')
    # Sort by descending
    number_of_class_instances.sort_values(by='count', inplace=True, ascending=False)
    # Returns most frequent classification
    predicted_classification = number_of_class_instances.head(n=1).values.tolist()[0][0]
    return predicted_classification


if __name__ == '__main__':
    data_training, data_unseen = load_data()

    print('training_data length: {}'.format(len(data_training.index)))
    print('unseen_data length: {}'.format(len(data_unseen.index)))

    # The number of correct predictions
    num_correct = 0

    for index, row in data_unseen.iterrows():
        # Gets the neighbours for that particular unseen record
        records_neighbours = get_neighbours(data_training, row.tolist()[:-1])
        classification_prediction = predict_classification(records_neighbours)
        actual_classification = row.tolist()[0]
        # If predicted classification is the same as the actual
        if actual_classification == classification_prediction:
            num_correct += 1

    accuracy = num_correct / len(data_unseen.index)
    print("Accuracy: {}".format(accuracy))
