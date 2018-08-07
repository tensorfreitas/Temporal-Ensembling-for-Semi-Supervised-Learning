import os
import sys
import urllib

from scipy.io import loadmat
import numpy as np

import tensorflow as tf

from tfrecord_loader import TfrecordLoader


class SvnhLoader:
    """ If not in the disk, download the SVNH dataset and prepares the dataset ready
        to be loaded for the Semi-supervised learning task in tensorflow,
    """

    # Constant attributes
    _NUM_TOTAL_SAMPLES = 99289
    _TRAIN_URL = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    _TEST_URL = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    _IMAGE_SIZE = [32, 32, 3]
    _NUM_CLASSES = 10

    def __init__(self, dataset_path, num_train_samples, num_validation_samples,
                 num_labeled_samples, random_seed=666):
        """ Init

        Arguments:
            dataset_path {string} -- the path to save the data
            num_train_samples {int} -- number of samples to use in training set (the sum of
                                       labeld + unlabeled train samples)
            num_validation_samples {int} -- number of samples to use in validation set
            num_labeled_samples {int} -- number of labeled samples to use
            random_seed {int} -- seed to use
        """

        self._dataset_path = dataset_path
        self._num_train_samples = num_train_samples
        self._num_test_samples = self._NUM_TOTAL_SAMPLES - self._num_train_samples
        self._num_validation_samples = num_validation_samples
        self._num_labeled_samples = num_labeled_samples
        self._num_unlabeled_train_samples = num_train_samples - \
            num_validation_samples - num_labeled_samples
        self._random_seed = random_seed

    def __normalize_and_prepare_dataset(self, mat_dataset):
        """ Receives a mat dataset and normalized the data accordingly to the 
           described in the original paper (std normalization)

        Arguments:
            mat_dataset {dict} -- mat dict (scipy.io.loadmat) dataset directly loaded 
                                  from the url mat

        Returns:
            [np.ndarray] -- Images normalized and flattened (num_images x (32*32*3))
            [np.ndarray] -- Correspondent labels
        """

        # Convert data to numpy array
        X = mat_dataset['X'].astype(np.float64)

        # Convert it to zero mean and unit variance
        X -= np.mean(X, axis=(1, 2, 3), keepdims=True)
        X /= (np.mean(X ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5)

        # Original dataset comes with wrong order in the dimensions
        X = X.transpose((3, 0, 1, 2))

        X = X.reshape([X.shape[0], -1])
        y = mat_dataset['y'].flatten().astype(np.int32)
        # 0 is label 10
        y[y == 10] = 0

        return X, y

    def __download_and_extract_dataset(self):
        """ Downloads the dataset and saves it in the _dataset_path. 
            Data is saved as a .mat file (as given by the original
            dataset). The mat file is then loaded and std normalized.

        Returns:
            [np.array] -- normalized train images
            [np.array] -- train labels
            [np.array] -- normalized test images
            [np.array] -- test labels
        """

        filepath_train = self._dataset_path + '/train_32x32.mat'
        filepath_test = self._dataset_path + '/test_32x32.mat'

        def download_progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        # Download dataset
        urllib.request.urlretrieve(
            self._TRAIN_URL, filepath_train, download_progress)
        urllib.request.urlretrieve(
            self._TEST_URL, filepath_test, download_progress)

        print('\n')

        # Load resultant mat files
        train_data = loadmat(filepath_train)
        test_data = loadmat(filepath_test)

        # Normalize between 0 and 1
        train_X, train_y = self.__normalize_and_prepare_dataset(train_data)
        test_X, test_y = self.__normalize_and_prepare_dataset(test_data)

        # Remove mat files
        os.remove(filepath_train)
        os.remove(filepath_test)

        return train_X, train_y, test_X, test_y

    def __generate_tfrecord(self, images, labels, filename):
        """ Receives a set of images and labels and converts them into
            tensorflow tfrecords file saving them in the dataset path
            given with the desired filename.

        Arguments:
            images {np.array} -- images for this dataset
            labels {np.array} -- labels for this dataset
            filename {filename} -- filename for this dataset
        """

        # If we are taking care of unlabeled data
        if labels == []:
            pass
        elif images.shape[0] != labels.shape[0]:
            raise ValueError("Images size %d does not match label size %d." %
                             (images.shape[0], labels.shape[0]))

        print('Writing', filename)

        writer = tf.python_io.TFRecordWriter(filename)

        # Write each image for the tfrecords file
        for index in range(images.shape[0]):
            image = images[index].tolist()

            # If unlabeled dataset label is -1
            if labels == []:
                current_label = -1
            else:
                current_label = int(labels[index])

            # Image index is needed to keep track of the temporal ensembling past predictions 
            # without loosing the shuffle batches
            sample = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[32])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[32])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[current_label])),
                'image': tf.train.Feature(float_list=tf.train.FloatList(value=image)),
                'image_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))}))
            writer.write(sample.SerializeToString())
        writer.close()

    def download_images_and_generate_tf_record(self):
        """ Main function of the class that allows generating and saving the tfrecords
            for labeled train, unlabeled train, validation and test datasets.
        """

        # Create folder if needed
        if not os.path.exists(self._dataset_path):
            os.makedirs(self._dataset_path)
        else:  # Dataset already loaded
            return

        # Download and process dataset
        train_X, train_y, test_X, test_y = self.__download_and_extract_dataset()

        # Use the seed provided
        rng = np.random.RandomState(self._random_seed)

        # I know I could initalize to zeros to avoid the appends, but it's only
        # done once, so let me have it
        labeled_train_X = np.empty(shape=(0, 32*32*3))
        labeled_train_y = []
        unlabeled_train_X = np.empty(shape=(0, 32*32*3))
        validation_X = np.empty(shape=(0, 32*32*3))
        validation_y = []

        # Randomly shuffle the dataset, and have balanced labeled and validation
        # datasets (avoid having and unbalenced train set that could hurt the results)
        for label in range(10):
            label_mask = (train_y == label)
            current_label_X = train_X[label_mask]
            current_label_y = train_y[label_mask]
            current_label_X, current_label_y = rng.permutation(
                current_label_X), rng.permutation(current_label_y)
            # Take care of the labeled train set
            labeled_train_X = np.append(labeled_train_X, current_label_X[:int(
                self._num_labeled_samples/self._NUM_CLASSES), :], axis=0)
            labeled_train_y = np.append(labeled_train_y, current_label_y[:int(
                self._num_labeled_samples/self._NUM_CLASSES)])
            current_label_X = current_label_X[int(
                self._num_labeled_samples/self._NUM_CLASSES):, :]
            current_label_y = current_label_y[int(
                self._num_labeled_samples/self._NUM_CLASSES):]
            # Now let's take care of validation
            validation_X = np.append(validation_X, current_label_X[:int(
                self._num_validation_samples/self._NUM_CLASSES)], axis=0)
            validation_y = np.append(validation_y, current_label_y[:int(
                self._num_validation_samples/self._NUM_CLASSES)])
            current_label_X = current_label_X[int(
                self._num_validation_samples/self._NUM_CLASSES):, :]
            current_label_y = current_label_y[int(
                self._num_validation_samples/self._NUM_CLASSES):]
            # The rest goes to Unlabeled train
            unlabeled_train_X = np.append(
                unlabeled_train_X, current_label_X, axis=0)

        # Print final set shapes
        print("Labeled train shape: ", labeled_train_X.shape)
        print("Unlabeled train shape: ", unlabeled_train_X.shape)
        print("Validation shape: ", validation_X.shape)
        print("Test shape: ", test_X.shape)

        # Write tfrecords to disk
        self.__generate_tfrecord(labeled_train_X, labeled_train_y, os.path.join(
            self._dataset_path, 'labeled_train.tfrecords'))

        self.__generate_tfrecord(unlabeled_train_X, [], os.path.join(
            self._dataset_path, 'unlabeled_train.tfrecords'))

        self.__generate_tfrecord(validation_X, validation_y, os.path.join(
            self._dataset_path, 'validation_set.tfrecords'))

        self.__generate_tfrecord(test_X, test_y, os.path.join(
            self._dataset_path, 'test_set.tfrecords'))

    def load_dataset(self, batch_size, epochs, fraction_of_labeled_per_batch=1.0,
                     fraction_of_unlabeled_per_batch=1.0, shuffle=True):
        """ Calls the TfrecordLoader to load the generated 
           tfrecords file.

        Arguments:
            batch_size {int} -- desired batch size
            epochs {int} -- number of epochs for train
            fraction_of_labeled_per_batch {float} -- if 1.0 use full batch_size for labeled set, if not
                                                     use a batch size of batch_size * fraction_of_labeled_per_batch
                                                     for labeled set.
            fraction_of_unlabeled_per_batch {float} -- if 1.0 use full batch_size for unlabeled set, if not
                                                       use a batch size of batch_size * fraction_of_unlabeled_per_batch
                                                       for unlabeled set.
            shuffle {bool} -- shuffle the dataset (set it to false for temporal ensembling)

        Returns:
            {tf.data.Iterator} -- iterator for a specific tfrecords file
        """

        tfrecord_loader = TfrecordLoader(
            './data', batch_size, epochs, self._IMAGE_SIZE, self._NUM_CLASSES,
            fraction_of_labeled_per_batch, fraction_of_unlabeled_per_batch, shuffle)
        return tfrecord_loader.load_dataset()
