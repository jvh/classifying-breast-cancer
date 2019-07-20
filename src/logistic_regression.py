########################################################################################
# Attempts to classify breast cancer given a set of features using logistic regression.#
#                                                                                      #
# Author: Jonathan Harper                                                              #
########################################################################################

import numpy as np
import helper_functions as func

# Number of 'steps' to be taken to train model
EPOCH_COUNT = 1000000
# Limitation to the variance of the weights after each step
LEARNING_RATE = 0.01
# This describes the maximum correlation before features are considered highly correlated
CORRELATED_THRESHOLD = 0.95


def logistic_function(data, coefficients_list):
    """
    This is used to predict the binary classification given the linear combination of the independent variables by
    plotting onto a logistic/sigmoid function

    :param data: The data itself
    :param coefficients_list: The coefficients (otherwise known as weights) of the independent vars
    :return: The logistic function mapping
    """
    # Dot product between file and coefficients. The first value of the coefficient is the intercept, this is multiplied
    # against the intercept_template defined above to give intercept*1=intercept
    values = np.dot(data, coefficients_list)

    return 1 / (1 + np.exp(-values))


if __name__ == '__main__':
    data_formatted, data_without_class, data_class_only, _ = func.read_csv_and_prep()

    # Initialise coefficients (weights) to 0
    coefficients = np.zeros(data_formatted.shape[1])

    predictions = None

    for count in range(EPOCH_COUNT):
        if count > 0 and count % 10000 == 0:
            print("Count: ", count)

        predictions = logistic_function(data_formatted, coefficients)

        # Calculating deviation between real and predicted
        error = data_class_only.T - predictions
        # The gradient function is to be reduced, as we approach 0 we approach the least error (as the gradient tends
        # towards 0, our function shows that our results are becoming more accurate)
        gradient = np.dot(data_formatted.T, error)
        # Adjusting coefficients for more accurate result
        coefficients += gradient * LEARNING_RATE

    print("Coefficient values {}".format(coefficients))
    print("Accuracy: {}".format(func.calculate_accuracy(predictions, data_class_only)))
