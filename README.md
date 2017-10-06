# Description
This project performs CIFAR-10 classification with scikit-learn and Tensorflow.
It demonstrates a few different approaches to images classification problem.

# Usage
_cifar.py_ is a script version running the whole analysis.

_cifar.ipynb_ has similar content and was used to develop the solution present in _cifar.py_.

To run the files above, use Anaconda environment described in _environment.yml_.

# Results
A shallow model (using HOG features and SVM classifier) achieved an accuracy of 48%.
It is unsatisfactory, so an Inception-v3 model was used to extract visual features from CIFAR10 images.
The features embedding into 2-dimensional space is presented in file _cnn_codes.png_.
It was created in two steps: first, by reducing the dimensionality to 10 dimensions using PCA;
then, by t-SNE to further reduce it to 2 dimensions.


With these features, several classifiers have been trained on a limited number of training examples (1000).
In each grid, the best parameters were chosen basing on a validation set score.

1. SVM

    Parameters grid:
    - kernel: linear, rbf
    - tolerance: 0.001, 0.01
    - error term penalty (C): 0.1, 1, 10

    Best parameters: rbf kernel, tolerance 0.001, penalty 10.
    Test score: 86%

2. Random Forest Classifier

    Parameters grid:
    - number of decision trees: 10, 100, 1000

    Best parameter: 1000.
    Test score: 83%


Afterwards, target estimators for 1. and 2. were trained on a larger dataset (10000), with the best hyperparameters.
The classifiers achieved accuracy of **89.2%** and **84.6**, respectively.


3. Besides, a simple neural network was trained with no parameters tuning and only one layer with 30 neurons.
Trained on a dataset of size 10000, it achieved accuracy of 86.4% on a test score.

