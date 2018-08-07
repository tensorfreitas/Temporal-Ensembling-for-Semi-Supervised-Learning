import tensorflow as tf

class TfrecordLoader:
    """ Class that allows loading a tfrecords file and generating an iterator
        for SSL training
    """

    def __init__(self, dataset_path, batch_size, epochs, image_size, num_classes, 
    fraction_of_labeled_per_batch=1.0, fraction_of_unlabeled_per_batch=1.0, shuffle=True):
        """ Init
        
        Arguments:
            dataset_path {string} -- path to the tfrecord files
            batch_size {int} -- batch size to use
            epochs {int} -- number of epochs
            image_size {list} -- image size [rows, cols, num_channels]
            num_classes {int} -- number of different classes
            fraction_of_labeled_per_batch {float} -- if 1.0 use full batch_size for labeled set, if not
                                                     use a batch size of batch_size * fraction_of_labeled_per_batch
                                                     for labeled set.
            fraction_of_unlabeled_per_batch {float} -- if 1.0 use full batch_size for unlabeled set, if not
                                                       use a batch size of batch_size * fraction_of_unlabeled_per_batch
                                                       for unlabeled set.
            shuffle {bool} -- shuffle the dataset (set it to false for temporal ensembling)
        """
        assert (fraction_of_labeled_per_batch <=1.0 and fraction_of_labeled_per_batch > 0),"Fraction should be between 0 and 1"
        self._dataset_path = dataset_path
        self._labeled_tfrecord_path = dataset_path + '/labeled_train.tfrecords'
        self._unlabeled_tfrecord_path = dataset_path + '/unlabeled_train.tfrecords'
        self._validation_tfrecord_path = dataset_path + '/validation_set.tfrecords'
        self._test_tfrecord_path = dataset_path + '/test_set.tfrecords'
        self._batch_size = batch_size
        self._epochs = epochs
        self._image_size = image_size
        self._num_classes = num_classes
        self._fraction_of_labeled_per_batch = fraction_of_labeled_per_batch
        self._fraction_of_unlabeled_per_batch = fraction_of_unlabeled_per_batch
        self._shuffle = shuffle

    def load_dataset(self):
        """ Main function to load the tfrecords files and create the iterators
        
        Returns:
            {tf.data.Iterator} -- iterator for labeled train set
            {tf.data.Iterator} -- iterator for unlabeled train set
            {tf.data.Iterator} -- iterator for validation set
            {tf.data.Iterator} -- iterator for test set
        """


        def __tfrecord_parser(sample):
            """ Helper parser
            """
            # Image index is needed to keep track of the temporal ensembling past predictions 
            # without loosing the shuffle batches
            keys_to_features = {
                'image': tf.FixedLenFeature(
                    [self._image_size[0]*self._image_size[1]*self._image_size[2]], tf.float32),
                'label': tf.FixedLenFeature([], tf.int64),
                'image_index': tf.FixedLenFeature([], tf.int64)
            }
            parsed_features = tf.parse_single_example(sample, keys_to_features)
            image = tf.reshape(parsed_features['image'], self._image_size)
            label = tf.one_hot(tf.cast(parsed_features['label'], tf.int64), self._num_classes)
            return image, label, tf.cast(parsed_features['image_index'], tf.int64)
        
        labeled_train_dataset = tf.data.TFRecordDataset([self._labeled_tfrecord_path])
        if self._shuffle:
            labeled_train_dataset = labeled_train_dataset.shuffle(10000, seed=None, reshuffle_each_iteration=True)
            
        labeled_train_dataset = labeled_train_dataset.repeat(self._epochs*1000)
        labeled_train_dataset = labeled_train_dataset.map(__tfrecord_parser)
        if self._fraction_of_labeled_per_batch == 1.0:
            labeled_train_dataset = labeled_train_dataset.batch(self._batch_size)
        else:
            labeled_train_dataset = labeled_train_dataset.batch(
                round(self._batch_size*self._fraction_of_labeled_per_batch))
        
        train_labeled_iterator = labeled_train_dataset.make_one_shot_iterator()
        

        unlabeled_train_dataset = tf.data.TFRecordDataset([self._unlabeled_tfrecord_path])
        if self._shuffle:
            unlabeled_train_dataset = unlabeled_train_dataset.shuffle(10000)

        unlabeled_train_dataset = unlabeled_train_dataset.repeat(self._epochs)
        unlabeled_train_dataset = unlabeled_train_dataset.map(__tfrecord_parser)
        if self._fraction_of_labeled_per_batch == 1.0:
            unlabeled_train_dataset = unlabeled_train_dataset.batch(self._batch_size)
        else:
            unlabeled_train_dataset = unlabeled_train_dataset.batch(
                round(self._batch_size*self._fraction_of_unlabeled_per_batch))

        train_unlabeled_iterator = unlabeled_train_dataset.make_one_shot_iterator()

        validation_dataset = tf.data.TFRecordDataset([self._validation_tfrecord_path])
        if self._shuffle:
            validation_dataset = validation_dataset.shuffle(10000)
        validation_dataset = validation_dataset.repeat(self._epochs)
        validation_dataset = validation_dataset.map(__tfrecord_parser)
        validation_dataset = validation_dataset.batch(self._batch_size)
        validation_iterator = validation_dataset.make_one_shot_iterator()

        test_dataset = tf.data.TFRecordDataset([self._test_tfrecord_path])
        if self._shuffle:
            test_dataset = test_dataset.shuffle(10000)
        test_dataset = test_dataset.repeat(self._epochs)
        test_dataset = test_dataset.map(__tfrecord_parser)
        test_dataset = test_dataset.batch(self._batch_size)
        test_iterator = test_dataset.make_one_shot_iterator()

        return train_labeled_iterator, train_unlabeled_iterator, validation_iterator, test_iterator