import math
import numpy as np 
import tensorflow as tf

# Edited files with weight normalization and mean only batch normalization
import weight_norm_layers.Conv2D
import weight_norm_layers.Dense


def temporal_ensembling_loss(X_train_labeled, y_train_labeled, X_train_unlabeled, model, unsupervised_weight, ensembling_targets):
    """ Gets the loss for the temporal ensembling model

    Arguments:
        X_train_labeled {tensor} -- labeled samples
        y_train_labeled {tensor} -- labeled train labels
        X_train_unlabeled {tensor} -- unlabeled samples 
        model {tf.keras.Model} -- temporal ensembling model
        unsupervised_weight {float} -- weight of the unsupervised loss
        ensembling_targets {np.array} --  ensembling targets

    Returns:
        {tensor} -- predictions for the ensembles
        {tensor} -- loss value
    """

    z_labeled = model(X_train_labeled)
    z_unlabeled = model(X_train_unlabeled)

    current_predictions = tf.concat([z_labeled, z_unlabeled], 0)

    return current_predictions, tf.losses.softmax_cross_entropy(
        y_train_labeled, z_labeled) + unsupervised_weight * (
            tf.losses.mean_squared_error(ensembling_targets, current_predictions))


def temporal_ensembling_gradients(X_train_labeled, y_train_labeled, X_train_unlabeled, model, unsupervised_weight, ensembling_targets):
    """ Gets the gradients for the temporal ensembling model

    Arguments:
        X_train_labeled {tensor} -- labeled samples
        y_train_labeled {tensor} -- labeled train labels
        X_train_unlabeled {tensor} -- unlabeled samples 
        model {tf.keras.Model} -- temporal ensembling model
        unsupervised_weight {float} -- weight of the unsupervised loss
        ensembling_targets {np.array} --  ensembling targets

    Returns:
        {tensor} -- predictions for the ensembles
        {tensor} -- loss value
        {tensor} -- gradients for each model variables
    """

    with tf.GradientTape() as tape:
        ensemble_precitions, loss_value = temporal_ensembling_loss(X_train_labeled, y_train_labeled, X_train_unlabeled,
                                                                   model, unsupervised_weight, ensembling_targets)

    return ensemble_precitions, loss_value, tape.gradient(loss_value, model.variables)


def pi_model_loss(X_train_labeled, y_train_labeled, X_train_unlabeled,
                  pi_model, unsupervised_weight):
    """ Gets the Loss Value for SSL Pi Model

    Arguments:
        X_train_labeled {tensor} -- train images
        y_train_labeled {tensor} -- train labels
        X_train_unlabeled {tensor} -- unlabeled train images
        pi_model {tf.keras.Model} -- model to be trained
        unsupervised_weight {float} -- weight

    Returns:
        {tensor} -- loss value
    """
    z_labeled = pi_model(X_train_labeled)
    z_labeled_i = pi_model(X_train_labeled)

    z_unlabeled = pi_model(X_train_unlabeled)
    z_unlabeled_i = pi_model(X_train_unlabeled)

    # Loss = supervised loss + unsup loss of labeled sample + unsup loss unlabeled sample
    return tf.losses.softmax_cross_entropy(
        y_train_labeled, z_labeled) + unsupervised_weight * (
            tf.losses.mean_squared_error(z_labeled, z_labeled_i) +
            tf.losses.mean_squared_error(z_unlabeled, z_unlabeled_i))


def pi_model_gradients(X_train_labeled, y_train_labeled, X_train_unlabeled,
                       pi_model, unsupervised_weight):
    """ Returns the loss and the gradients for eager Pi Model

    Arguments:
        X_train_labeled {tensor} -- train images
        y_train_labeled {tensor} -- train labels
        X_train_unlabeled {tensor} -- unlabeled train images
        pi_model {tf.keras.Model} -- model to be trained
        unsupervised_weight {float} -- weight

    Returns:
        {tensor} -- loss value
        {tensor} -- gradients for each model variables
    """
    with tf.GradientTape() as tape:
        loss_value = pi_model_loss(X_train_labeled, y_train_labeled, X_train_unlabeled,
                                   pi_model, unsupervised_weight)
    return loss_value, tape.gradient(loss_value, pi_model.variables)


def ramp_up_function(epoch, epoch_with_max_rampup=80):
    """ Ramps the value of the weight and learning rate according to the epoch
        according to the paper

    Arguments:
        {int} epoch
        {int} epoch where the rampup function gets its maximum value

    Returns:
        {float} -- rampup value
    """

    if epoch < epoch_with_max_rampup:
        p = max(0.0, float(epoch)) / float(epoch_with_max_rampup)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0


def ramp_down_function(epoch, num_epochs):
    """ Ramps down the value of the learning rate and adam's beta
        in the last 50 epochs according to the paper

    Arguments:
        {int} current epoch
        {int} total epochs to train

    Returns:
        {float} -- rampup value
    """
    epoch_with_max_rampdown = 50

    if epoch >= (num_epochs - epoch_with_max_rampdown):
        ep = (epoch - (num_epochs - epoch_with_max_rampdown)) * 0.5
        return math.exp(-(ep * ep) / epoch_with_max_rampdown)
    else:
        return 1.0


class PiModel(tf.keras.Model):
    """ Class for defining eager compatible tfrecords file

        I did not use tfe.Network since it will be depracated in the
        future by tensorflow.
    """

    def __init__(self):
        """ Init

            Set all the layers that need to be tracked in the process of
            gradients descent (pooling and dropout for example dont need
            to be stored)
        """

        super(PiModel, self).__init__()
        self._conv1a = weight_norm_layers.Conv2D.Conv2D(filters=128, kernel_size=[3, 3],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv1b = weight_norm_layers.Conv2D.Conv2D(filters=128, kernel_size=[3, 3],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv1c = weight_norm_layers.Conv2D.Conv2D(filters=128, kernel_size=[3, 3],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._pool1 = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=2, padding="same")
        self._dropout1 = tf.keras.layers.Dropout(0.5)

        self._conv2a = weight_norm_layers.Conv2D.Conv2D(filters=256, kernel_size=[3, 3],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv2b = weight_norm_layers.Conv2D.Conv2D(filters=256, kernel_size=[3, 3],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv2c = weight_norm_layers.Conv2D.Conv2D(filters=256, kernel_size=[3, 3],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._pool2 = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=2, padding="same")
        self._dropout2 = tf.keras.layers.Dropout(0.5)

        self._conv3a = weight_norm_layers.Conv2D.Conv2D(filters=512, kernel_size=[3, 3],
                                                        padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv3b = weight_norm_layers.Conv2D.Conv2D(filters=256, kernel_size=[1, 1],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)
        self._conv3c = weight_norm_layers.Conv2D.Conv2D(filters=128, kernel_size=[1, 1],
                                                        padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                                        bias_initializer=tf.keras.initializers.constant(
                                                            0.1),
                                                        weight_norm=True, mean_only_batch_norm=True)

        self._dense = weight_norm_layers.Dense.Dense(units=10, activation=tf.nn.softmax,
                                                     kernel_initializer=tf.keras.initializers.he_uniform(),
                                                     bias_initializer=tf.keras.initializers.constant(
                                                         0.1),
                                                     weight_norm=True, mean_only_batch_norm=True)

    def __aditive_gaussian_noise(self, input, std):
        """ Function to add additive zero mean noise as described in the paper

        Arguments:
            input {tensor} -- image
            std {int} -- std to use in the random_normal

        Returns:
            {tensor} -- image with added noise
        """

        noise = tf.random_normal(shape=tf.shape(
            input), mean=0.0, stddev=std, dtype=tf.float32)
        return input + noise

    def __apply_image_augmentation(self, image):
        """ Applies random transformation to the image

        Arguments:
            image {tensor} -- image

        Returns:
            {tensor} -- transformed image
        """

        random_shifts = np.random.randint(-2, 2, image.numpy().shape[0])
        random_transformations = tf.contrib.image.translations_to_projective_transforms(
            random_shifts)
        image = tf.contrib.image.transform(image, random_transformations, 'NEAREST',
                                           output_shape=tf.convert_to_tensor(image.numpy().shape[1:3], dtype=np.int32))
        return image

    def call(self, input, training=True):
        """ Function that allows running a tensor through the pi model

        Arguments:
            input {[tensor]} -- batch of images
            training {bool} -- if true applies augmentaton and additive noise

        Returns:
            [tensor] -- predictions
        """

        if training:
            h = self.__aditive_gaussian_noise(input, 0.15)
            h = self.__apply_image_augmentation(h)
        else:
            h = input

        h = self._conv1a(h, training)
        h = self._conv1b(h, training)
        h = self._conv1c(h, training)
        h = self._pool1(h)
        h = self._dropout1(h, training=training)

        h = self._conv2a(h, training)
        h = self._conv2b(h, training)
        h = self._conv2c(h, training)
        h = self._pool2(h)
        h = self._dropout2(h, training=training)

        h = self._conv3a(h, training)
        h = self._conv3b(h, training)
        h = self._conv3c(h, training)

        # Average Pooling
        h = tf.reduce_mean(h, reduction_indices=[1, 2])
        return self._dense(h, training)
