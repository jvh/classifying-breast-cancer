**NOTE**: This was done as a personal project

# Classifying Breast Cancer Using Supervised Machine Learning Techniques

In this project, I attempt to classify if a tumour present in breast tissue (i.e. record of data containing a feature set of that tumour) is likely to be malignant or benign. For this, I have implemented 2 supervised learning techniques: **_k_-nearest neighbours** and **logistic regression**. 

## Usage Instructions

These commands should both be ran in the **top-level** directory of the repository.

### Prerequisites

Install [python3](https://www.python.org/download/releases/3.0/).

Run command `pip install -r src/requirements.txt` in order to install package requirements.

### Commands

* _k_-nearest_neighbours: `python src/k_nearest_neighbour.py`.
* logistic regression classification: `python src/logistic_regression.py`.

## Dataset

I am using a [dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/) (`wdbc.data`) provided by the University of Wisconsin. The data has been cleaned to remove irrelevant (e.g. ID) and otherwise highly-correlated features which would adversely affect the results of the classification. It should be noted that the dataset itself isn't huge (~600 records) and this project would likely benefit from a larger dataset for training purposes. Additionally, data wasn't cleaned for any anomalous entries, this should be considered for larger datasets without known sources. I recommend using [OpenRefine](http://openrefine.org/) for this step should you wish to take it.

## Logistic Regression

Essentially, uses the sigmoid function, otherwise known as the logistic function, to make binary predictions based on the linear combinations of independent variables onto a range between 0-1. In this case, the decision bound is located at 0.5, that is when a prediction of 0.6 is made, it corresponds to a final prediction of 1 (the cancer is malignant).

Logistic classification had an accuracy of ~97%. 

### Parameters

* Epoch count (the number of iterations taken to train the model): 1,000,000
* Learning rate (limitation to the variance of the weights after each step): 0.01
* Correlated threshold (which describes the maximum correlation before features are considered highly correlated): 0.95

## _k_-Nearest-Neighbour

kNN is a non-parametric, lazy learning algorithm, based on feature similarity. Non-parametric meaning that kNN does not make any assumptions on the underlying data distribution. Lazy refers to the fact the kNN does not use the training data points to do any generalisation. Meaning there is no explicit training phase, which makes training very fast, but can be computationally expensive over larger training datasets. When a prediction is required for a unseen data instance, the kNN algorithm will search through the training dataset for the k most similar instances. kNN involves a similarity measure, where the distance between two data instances is calculated and used to find the k most similar neighbours. As out dataset consisted of real-valued data, the Euclidean distance was used.

The kNN classifier was significantly less accurate that logistic classification. I recorded an accuracy of approximately 63%. I believe this is significantly affected by the relatively small training set.

### Parameters

* Maximum number of neighbouts \[for each point]: 4
* A ratio of 80/20 for train/test split of the dataset was used. This corresponds to 455 records for the training data and 114 records of unseen data.
