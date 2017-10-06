# coding: utf-8

# Copyright by Mikołaj Błaż, 2017

import pickle
import os
import tarfile
import urllib.request
from math import ceil
from shutil import rmtree
from tempfile import mkdtemp
from time import time

import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import color

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

import tensorflow as tf


np.random.seed(42)

BATCH_SIZE = 10000
BATCH_CNT = 5
TRAIN_HEIGHT = 50000  # = BATCH_SIZE * BATCH_CNT = sum([batch_dicts[0][b'data'].shape[i] for i in range(5)])

CLASS_CNT = 10  # = np.unique(test_labels)
CLASS_SAMPLE_SIZE = 10  # class images sample size

MODEL_DIR = 'inception/'
MODEL_FILE = 'classify_image_graph_def.pb'
NUM_FEATURES = 2048  # length of bottleneck layer

DATA_DIR = 'dataset/'
EXTRACTED_DATA_DIR = 'cifar-10-batches-py/'


###############################################################################
# 1. Prepare dataset

# Dataset CIFAR-10
# downloaded from: https://www.cs.toronto.edu/~kriz/cifar.html
# introduced in: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def download_extract_if_necessary(dest_dir, data_url, expected_tarfile, expected_extracted_file):
    """
    Download (if necessary) and extract (if necessary) a file
    in tar.gz format (if necessary) to a given directory.
    
    Both arguments 'expected_*' help to avoid unnecessary download or extraction.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    dest_filename = os.path.join(dest_dir, expected_tarfile)
    if not os.path.exists(dest_filename):
        print('Downloading data from %s...' % data_url)
        dest_filename, _ = urllib.request.urlretrieve(data_url, dest_filename)
        print('Download finished')
    
    if not os.path.exists(os.path.join(dest_dir, expected_extracted_file)):
        print('Extracting archive...')
        with tarfile.open(dest_filename, "r:gz") as tar:
            tar.extractall(dest_dir)
    
    print('Extracted file(s) ready in directory: %s' % dest_dir)


def prepare_data():
    download_extract_if_necessary(
        dest_dir=DATA_DIR,
        data_url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        expected_tarfile='cifar-10-python.tar.gz',
        expected_extracted_file=EXTRACTED_DATA_DIR
    )

    batch_train_dicts = [
        unpickle(os.path.join(DATA_DIR, EXTRACTED_DATA_DIR, ('data_batch_%d' % i)))
        for i in range(1, 6)
    ]
    batch_test_dict = unpickle(os.path.join(DATA_DIR, EXTRACTED_DATA_DIR, 'test_batch'))

    # Join training set batches together, for simplified processing
    train_width = batch_train_dicts[0][b'data'].shape[1]
    X_train = np.empty((TRAIN_HEIGHT, train_width), dtype='uint8')
    y_train = np.empty(TRAIN_HEIGHT, dtype='uint8')
    for i in range(0, BATCH_CNT):
        X_train[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)] = batch_train_dicts[i][b'data']
        y_train[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)] = batch_train_dicts[i][b'labels']

    X_test = batch_test_dict[b'data']
    y_test = np.asarray(batch_test_dict[b'labels'], dtype='uint8')

    print('Full training set size: %d' % len(X_train))
    print('Full test set size: %d' % len(X_test))

    return X_train, y_train, X_test, y_test


###############################################################################
# 2. Plot images

# Convert images to RGB format
def cifar_to_rgb_dataset(imgs):
    """
    Change format from CIFAR-like to matplotlib-like of all given images 
    
    :param imgs_cifar: an array of images represented by list of 3072 consecutive pixel values:
        first all red, then green, then blue; row-wise
    :return: an array of shape (..., 32, 32, 3), with values of type 'float32'
    """
    img_3d = np.reshape(imgs, (-1, 3, 32, 32))
    img_rgb = np.transpose(img_3d, (0, 2, 3, 1))
    # scale values to [0, 1] interval:
    return np.asarray(img_rgb, dtype='float32') / 255.


def plot_sample_images(X, y, sample_size=10):
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for cls in range(CLASS_CNT):
        # select class images
        X_class = X[y == cls]
        # choose 10 random images and convert to RGB format
        rnd_indices = np.random.choice(len(X_class), sample_size, replace=False)
        X_cls = cifar_to_rgb_dataset(X_class[rnd_indices])
        # plot them
        for x, img in enumerate(X_cls):
            fig.add_subplot(CLASS_CNT, sample_size, cls * sample_size + x + 1)
            plt.imshow(img)
            plt.axis('off')

    plt.savefig('sample.png')
    plt.show()


###############################################################################
# 3. Shallow classifier

# Extract HOG features
def rgb_to_hog(img):
    img_gray = color.rgb2gray(img)
    return hog(img_gray, block_norm='L2-Hys', visualise=False)


def rgb_to_hog_dataset(imgs):
    """ Calculate HOG for all images in dataset """
    result = list()
    for img in imgs:
        result.append(rgb_to_hog(img))
    return np.asarray(result, dtype='float32')


# Extract HOG features, starting from CIFAR format
def cifar_to_hog(imgs):
    return rgb_to_hog_dataset(cifar_to_rgb_dataset(imgs))

hogs = FunctionTransformer(cifar_to_hog)


# Save and load models from disk
def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    return joblib.load(filename)


def choose_random_subset(X, y, subset_size):
    indices = np.random.permutation(len(X))[:subset_size]
    return X[indices], y[indices]


def evaluate_model(clf, X, y):
    score = clf.score(X, y)
    print('Score: %f' % score)


def train_model(clf, X, y, subset_size=None, clf_file=None):
    """
    Train 'clf' on X, y and test on X_test, y_test.

    :param clf: sklearn classifier
    :param subset_size: if not None, the size of (randomly chosen) subset of X, y to use for training
    :param clf_file: load model from this file, or save model there after training
     """

    if clf_file is not None and os.path.exists(clf_file):
        return load_model(clf_file)

    if subset_size is not None:
        X, y = choose_random_subset(X, y, subset_size)

    print('Fitting on %d images...' % (len(X),))
    start = time()
    clf.fit(X, y)
    end = time()

    print('Fitting done.')
    print('Time elapsed: %0.03fs' % (end - start,))

    if clf_file is not None:
        if not os.path.exists('models/'):
            os.makedirs('models/')
        save_model(clf, clf_file)

    return clf


# Estimator fitting
def train_test_grid_search_model(clf, X, y, X_test, y_test, subset_size=None, clf_file=None):
    """
    Train GridSearch 'clf' on X, y and test on X_test, y_test.

    :param clf: GridSearch classifier
    :param subset_size: if not None, the size of (randomly chosen) subset of X, y to use for training
    :param clf_file: load model from this file, or save model there after training
     """

    clf = train_model(clf, X, y, subset_size, clf_file)

    # Grid search summary
    print('Best parameters set found on training set:')
    print(clf.best_params_)
    print()
    print('Best score:')
    print(clf.best_score_)
    print()
    print('Grid scores on training set:')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print('Score on test set:')
    print(clf.score(X_test, y_test))

    return clf


def train_shallow_classifier(X, y, X_test, y_test, subset_size):
    print()
    print('Train HoG + SVM')
    cachedir = mkdtemp()

    pipe = Pipeline([('hog', hogs), ('norm', StandardScaler()), ('svc', SVC())], memory=cachedir)

    grid_params = [{'svc__kernel': ['rbf'], 'svc__C': [0.1, 1., 10.]},
                   {'svc__kernel': ['linear'], 'svc__C': [1.]}]

    clf = GridSearchCV(pipe, grid_params, cv=3, n_jobs=-1)

    train_test_grid_search_model(clf, X, y, X_test, y_test, subset_size, clf_file='models/shallow.pkl')

    rmtree(cachedir)
    return clf


###############################################################################
# 4. Visual features

# Inception v3 model, trained on ImageNet data
# source: http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz


# Load the model graph (with pretrained weights)
def create_graph():
    """ Creates a graph from saved GraphDef file. """
    with tf.gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


# Methods for visual features extraction from CIFAR10 images with Inception model

def extract_cnn_codes(rgb_img, sess, bottleneck):
    """
    :param rgb_img: an array (?, ?, 3) with pixels in range [0,255] (*NOT* [0,1]).
    :return: an array (2048,) of calculated CNN codes
    """
    cnn_codes = sess.run(bottleneck, {'DecodeJpeg:0': rgb_img})
    return np.squeeze(cnn_codes)


def cifar_to_cnn_codes_dataset(imgs, sess):
    """
    :param imgs: an array (?, 3072) of type 'float32' with pixels in range [0,1]
    :return: an array (?, 2048) of calculated CNN codes
    """
    bottleneck = sess.graph.get_tensor_by_name('pool_3:0')  # visual features tensor

    rgb_imgs = cifar_to_rgb_dataset(imgs) * 255  # rescale to values in range [0,255]
    result = list()
    for img in rgb_imgs:
        result.append(extract_cnn_codes(img, sess, bottleneck))
    return np.asarray(result, dtype='float32')


def load_or_compute_cnn_codes(X, codes_file, sess):
    """
    Computes CNN codes for dataset X and saves them to file 'codes_file'.
    If file already exists, use it.
    
    :param X: an array of images in CIFAR format
    :param codes_file: filepath (String)
    :return: an array (?, 2048) of calculated CNN codes
    """
    codes_loaded = False
    if os.path.exists(codes_file):
        # Load codes from file
        try:
            X_codes = np.load(codes_file)
            # check if dataset matches codes length
            # (weak condition of data integrity, but it's enough here)
            if len(X) == len(X_codes):
                codes_loaded = True
                print('CNN codes loaded successfully from file %s' % codes_file)
            else:
                print('Invalid codes present in file, replacing with new ones...')
            
        except (IOError, ValueError):
            print('Error during codes loading')
    
    if not codes_loaded:
        # Compute codes
        print('Start computing CNN codes...')
        start = time()
        X_codes = cifar_to_cnn_codes_dataset(X, sess)
        end = time()

        print('Computing done.')
        print('  Time elapsed: %0.02fs.' % (end - start,))
        print('  Average time: %0.02fs/image' % ((end - start) / len(X)))
        
        # Save codes to file
        np.save(codes_file, X_codes)
        print('Codes saved succesfully to file %s' % codes_file)
        print()
        
    return X_codes


# Compute CNN codes in batches (to save checkpoints)
def compute_codes_in_batches(X, b_size, filename_pattern, sess):
    """
    Compute CNN codes for images X, in batches of size 'b_size'.
    The purpose of this method is checkpointing each batch on disk.
    Subsequent batches will be saved to files according to 'filename_pattern',
    which is a string with one integer to fill (batch_index).
    """

    X_codes = np.empty((len(X), NUM_FEATURES), dtype='float32')

    for i in range(ceil(len(X) / float(b_size))):
        X_codes[i * b_size: (i + 1) * b_size] = load_or_compute_cnn_codes(
                                                    X[i * b_size: (i + 1) * b_size],
                                                    filename_pattern % i,
                                                    sess
                                                )
    print('All codes computed!')
    return X_codes


def compute_cnn_codes(X, X_test, X_subset_size=None):
    """
    :param X_subset_size: here, subset size is not chosen randomly, but first 'X_subset_size' are taken
    """
    if X_subset_size is not None:
        X = X[:X_subset_size]

    # Download inception model
    download_extract_if_necessary(
        dest_dir=MODEL_DIR,
        data_url='http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz',
        expected_tarfile='inception-2015-12-05.tgz',
        expected_extracted_file=MODEL_FILE
    )

    if not os.path.exists('codes/'):
        os.makedirs('codes/')

    create_graph()
    with tf.Session() as sess:
        X_codes = compute_codes_in_batches(X, 1000, 'codes/codes_train_%d.npy', sess)
        X_codes_test = compute_codes_in_batches(X_test, 100, 'codes/codes_test_%d.npy', sess)

    return X_codes, X_codes_test


###############################################################################
# 5. Plot CNN codes in 2 dimensions
def normalize_data(X, X_test):
    normalizer = StandardScaler()
    X_norm = normalizer.fit_transform(X)
    X_norm_test = normalizer.transform(X_test)
    return X_norm, X_norm_test


# Reduce dimensionality using first PCA, then t-SNE
def visualize_cnn_codes(X_codes, y, n_codes=1000):
    """
    :param n_codes: how man codes to plot
    """
    X_vis, y_vis = choose_random_subset(X_codes, y, n_codes)

    reduction = make_pipeline(PCA(10), TSNE(2))
    X_vis_2d = reduction.fit_transform(X_vis)

    # Plot
    y_normalized = y_vis.astype('float32') / 9.  # scale to range [0,1]
    colors = plt.cm.rainbow(y_normalized)
    plt.scatter(X_vis_2d[:, 0], X_vis_2d[:, 1], c=colors, alpha=0.2)
    plt.savefig('cnn_codes.png')
    plt.show()


###############################################################################
# 6-7. Train SVM model on top of CNN codes

def train_svm_on_codes(X_codes, y, X_codes_test, y_test, subset_size):
    print()
    print('Train SVM')

    grid_params = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1., 10.], 'tol': [0.001, 0.01]}
    clf = GridSearchCV(SVC(), grid_params, cv=3, n_jobs=-1)
    clf = train_test_grid_search_model(clf, X_codes, y, X_codes_test, y_test, subset_size, 'models/svm.pkl')
    return clf


def train_best_svm_on_codes(X_codes, y, X_codes_test, y_test, subset_size, best_params):
    print()
    print('Train best SVM')

    clf = SVC()
    clf.set_params(**best_params)
    clf = train_model(clf, X_codes, y, subset_size, 'models/svm_best.pkl')
    evaluate_model(clf, X_codes_test, y_test)
    return clf


###############################################################################
# 8. Further improvements

# Random forest
def train_random_forest(X_codes, y, X_codes_test, y_test, subset_size):
    print()
    print('Train RandomForest')

    grid_params = {'n_estimators': [10, 100, 1000]}
    clf = GridSearchCV(RandomForestClassifier(), param_grid=grid_params)

    clf = train_test_grid_search_model(clf, X_codes, y, X_codes_test, y_test, subset_size, 'models/rf.pkl')
    return clf


def train_best_random_forest(X_codes, y, X_codes_test, y_test, subset_size, best_params):
    print()
    print('Train RandomForest')

    clf = RandomForestClassifier()
    clf.set_params(**best_params)
    clf = train_model(clf, X_codes, y, subset_size, 'models/rf_best.pkl')
    evaluate_model(clf, X_codes_test, y_test)
    return clf


# Neural network
def train_nn(X_codes, y, X_codes_test, y_test, subset_size):
    """ Train a simple neural network. """
    print()
    print('Train Neural Network')

    X_codes, y = choose_random_subset(X_codes, y, subset_size)

    clf = tf.estimator.DNNClassifier(
        feature_columns=[tf.feature_column.numeric_column("x", shape=[NUM_FEATURES])],
        hidden_units=[30],
        n_classes=10,
        model_dir='models/nn'
    )

    # Training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_codes},
        y=y.astype('int32'),
        num_epochs=None,
        batch_size=1000,
        shuffle=False
    )

    clf.train(input_fn=train_input_fn, max_steps=2000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_codes_test},
        y=y_test.astype('int32'),
        num_epochs=1,
        shuffle=False
    )

    print()
    accuracy_score = clf.evaluate(input_fn=test_input_fn)["accuracy"]
    print('Accuracy: %f' % accuracy_score)
    return clf



# Voting ensemble
def vote(ys):
    """
    :param ys: list of predictions (numpy array) for different classifiers
    :return: prediction after voting
    """
    y_len = len(ys[0])
    img_ids = np.arange(y_len)

    ys_votes = np.zeros((y_len, CLASS_CNT), dtype='int32')
    for y in ys:
        # each y gives one vote for each image
        ys_votes[img_ids, np.asarray(y, dtype='int32')] += 1

    # Retrieve classes with highest scores
    argmaxes = np.argmax(ys_votes, axis=1)
    maxes = np.max(ys_votes, axis=1)

    # final decision: if at least two votes were given for the best class, choose it
    # otherwise, pick the strongest (first) classifier
    return np.where(maxes >= 2, argmaxes, ys[0])


def test_voting_ensemble(clf_svm, clf_rf, clf_nn, X_codes_test, y_test):
    nn_test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_codes_test},
        num_epochs=1,
        shuffle=False
    )

    y_pred = [
        clf_svm.predict(X_codes_test),
        clf_rf.predict(X_codes_test),
        [pred['class_ids'][0] for pred in list(clf_nn.predict(nn_test_input_fn))]
    ]

    y = vote(y_pred)

    print()
    print('Voting score on test set:')
    print(accuracy_score(y, y_test))


def main():
    X, y, X_test, y_test = prepare_data()
    plot_sample_images(X, y)

    clf_sh = train_shallow_classifier(X, y, X_test, y_test, 1000)

    X_codes, X_codes_test = compute_cnn_codes(X, X_test)
    X_codes, X_codes_test = normalize_data(X_codes, X_codes_test)
    visualize_cnn_codes(X_codes, y, 1000)

    del X   # free memory no longer needed

    clf_svm = train_svm_on_codes(X_codes, y, X_codes_test, y_test, 1000)
    clf_rf = train_random_forest(X_codes, y, X_codes_test, y_test, 1000)

    clf_nn = train_nn(X_codes, y, X_codes_test, y_test, 10000)
    clf_svm_best = train_best_svm_on_codes(X_codes, y, X_codes_test, y_test, 10000, clf_svm.best_params_)
    clf_rf_best = train_best_random_forest(X_codes, y, X_codes_test, y_test, 10000, clf_rf.best_params_)

    test_voting_ensemble(clf_svm_best, clf_rf_best, clf_nn, X_codes_test, y_test)


if __name__ == '__main__':
    main()
