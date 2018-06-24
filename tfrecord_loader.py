import tensorflow as tf

class TfrecordLoader:
    """ Class that allows loading a tfrecords file and generating an iterator
        for SSL training
    """

    def __init__(self, dataset_path, batch_size, epochs, image_size, num_classes):
        """ Init
        
        Arguments:
            dataset_path {string} -- path to the tfrecord files
            batch_size {int} -- batch size to use
            epochs {int} -- number of epochs
            image_size {list} -- image size [rows, cols, num_channels]
            num_classes {int} -- number of different classes
        """

        self._dataset_path = dataset_path
        self._labeled_tfrecord_path = dataset_path + '/labeled_train.tfrecords'
        self._unlabeled_tfrecord_path = dataset_path + '/unlabeled_train.tfrecords'
        self._validation_tfrecord_path = dataset_path + '/validation_set.tfrecords'
        self._test_tfrecord_path = dataset_path + '/test_set.tfrecords'
        self._batch_size = batch_size
        self._epochs = epochs
        self._image_size = image_size
        self._num_classes = num_classes

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

            keys_to_features = {
                'image': tf.FixedLenFeature(
                    [self._image_size[0]*self._image_size[1]*self._image_size[2]], tf.float32),
                'label': tf.FixedLenFeature([], tf.int64),
            }
            parsed_features = tf.parse_single_example(sample, keys_to_features)
            image = tf.reshape(parsed_features['image'], self._image_size)
            label = tf.one_hot(tf.cast(parsed_features['label'], tf.int64), self._num_classes)
            return image, label
            
        labeled_train_dataset = tf.data.TFRecordDataset([self._labeled_tfrecord_path])
        labeled_train_dataset = labeled_train_dataset.shuffle(10000)
        labeled_train_dataset = labeled_train_dataset.repeat(self._epochs)
        labeled_train_dataset = labeled_train_dataset.map(__tfrecord_parser)
        labeled_train_dataset = labeled_train_dataset.batch(self._batch_size)
        train_labeled_iterator = labeled_train_dataset.make_one_shot_iterator()

        unlabeled_train_dataset = tf.data.TFRecordDataset([self._unlabeled_tfrecord_path])
        unlabeled_train_dataset = unlabeled_train_dataset.shuffle(10000)
        unlabeled_train_dataset = unlabeled_train_dataset.repeat(self._epochs)
        unlabeled_train_dataset = unlabeled_train_dataset.map(__tfrecord_parser)
        unlabeled_train_dataset = unlabeled_train_dataset.batch(self._batch_size)
        train_unlabeled_iterator = unlabeled_train_dataset  .make_one_shot_iterator()

        validation_dataset = tf.data.TFRecordDataset([self._validation_tfrecord_path])
        validation_dataset = validation_dataset.shuffle(10000)
        validation_dataset = validation_dataset.repeat(self._epochs)
        validation_dataset = validation_dataset.map(__tfrecord_parser)
        validation_dataset = validation_dataset.batch(self._batch_size)
        validation_iterator = validation_dataset.make_one_shot_iterator()

        test_dataset = tf.data.TFRecordDataset([self._test_tfrecord_path])
        test_dataset = test_dataset.shuffle(10000)
        test_dataset = test_dataset.repeat(self._epochs)
        test_dataset = test_dataset.map(__tfrecord_parser)
        test_dataset = test_dataset.batch(self._batch_size)
        test_iterator = test_dataset.make_one_shot_iterator()

        return train_labeled_iterator, train_unlabeled_iterator, validation_iterator, test_iterator