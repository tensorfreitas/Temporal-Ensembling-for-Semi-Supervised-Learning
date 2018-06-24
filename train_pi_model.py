import math
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Enable Eager Execution
tf.enable_eager_execution()

from svnh_loader import SvnhLoader
from tfrecord_loader import TfrecordLoader

from pi_model import PiModel, pi_model_loss, pi_model_gradients, ramp_up_function, ramp_down_function


def main():
    # Constants variables
    NUM_TRAIN_SAMPLES = 73257
    NUM_TEST_SAMPLES = 26032

    # Editable variables
    num_labeled_samples = 1000
    num_validation_samples = 200
    batch_size = 50
    epochs = 300
    max_learning_rate = 0.001
    initial_beta1 = 0.9
    final_beta1 = 0.5
    checkpoint_directory = './checkpoints/PiModel'
    tensorboard_logs_directory = './logs/PiModel'

    # Assign it as tfe.variable since we will change it across epochs
    learning_rate = tfe.Variable(max_learning_rate)
    beta_1 = tfe.Variable(initial_beta1)

    # Download and Save Dataset in Tfrecords
    loader = SvnhLoader('./data', NUM_TRAIN_SAMPLES,
                        num_validation_samples, num_labeled_samples)
    loader.download_images_and_generate_tf_record()

    # Generate data loaders
    train_labeled_iterator, train_unlabeled_iterator, validation_iterator, test_iterator = loader.load_dataset(
        batch_size, epochs)

    batches_per_epoch = int(num_labeled_samples/batch_size)
    batches_per_epoch_val = int(num_validation_samples / batch_size)

    model = PiModel()
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta_1, beta2=0.990)
    max_unsupervised_weight = 100 * num_labeled_samples / \
        (NUM_TRAIN_SAMPLES - num_validation_samples)

    best_val_accuracy = 0
    global_step = tf.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer(tensorboard_logs_directory)
    writer.set_as_default()

    for epoch in range(epochs):

        rampdown_value = ramp_down_function(epoch, epochs)
        rampup_value = ramp_up_function(epoch)

        if epoch == 0:
            unsupervised_weight = 0
        else:
            unsupervised_weight = max_unsupervised_weight * \
                rampup_value

        learning_rate.assign(rampup_value * rampdown_value * max_learning_rate)
        beta_1.assign(rampdown_value * initial_beta1 +
                      (1.0 - rampdown_value) * final_beta1)

        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
        epoch_loss_avg_val = tfe.metrics.Mean()
        epoch_accuracy_val = tfe.metrics.Accuracy()
        for batch_nr in range(batches_per_epoch):
            X_labeled_train, y_labeled_train = train_labeled_iterator.get_next()
            X_unlabeled_train, _ = train_unlabeled_iterator.get_next()

            loss_val, grads = pi_model_gradients(X_labeled_train, y_labeled_train, X_unlabeled_train,
                                                 model, unsupervised_weight)
            optimizer.apply_gradients(zip(grads, model.variables),
                                      global_step=global_step)

            epoch_loss_avg(loss_val)
            epoch_accuracy(
                tf.argmax(model(X_labeled_train), 1), tf.argmax(y_labeled_train, 1))
            if (batch_nr == batches_per_epoch - 1):
                for batch_val_nr in range(batches_per_epoch_val):
                    X_val, y_val = validation_iterator.get_next()
                    y_val_predictions = model(X_val, training=False)

                    epoch_loss_avg_val(tf.losses.softmax_cross_entropy(
                        y_val, y_val_predictions))
                    epoch_accuracy_val(
                        tf.argmax(y_val_predictions, 1), tf.argmax(y_val, 1))

        print("Epoch {:03d}/{:03d}: Train Loss: {:9.7f}, Train Accuracy: {:02.6%}, Validation Loss: {:9.7f}, "
              "Validation Accuracy: {:02.6%}, lr={:.9f}, unsupervised weight={:5.3f}, beta1={:.9f}".format(epoch+1,
                                                                                                           epochs,
                                                                                                           epoch_loss_avg.result(),
                                                                                                           epoch_accuracy.result(),
                                                                                                           epoch_loss_avg_val.result(),
                                                                                                           epoch_accuracy_val.result(),
                                                                                                           learning_rate.numpy(),
                                                                                                           unsupervised_weight,
                                                                                                           beta_1.numpy()))

        # If the accuracy of validation improves save a checkpoint Best 85%
        if best_val_accuracy < epoch_accuracy_val.result():
            best_val_accuracy = epoch_accuracy_val.result()
            checkpoint = tfe.Checkpoint(optimizer=optimizer,
                                        model=model,
                                        optimizer_step=global_step)
            checkpoint.save(file_prefix=checkpoint_directory)

        # Record summaries
        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            tf.contrib.summary.scalar('Train Loss', epoch_loss_avg.result())
            tf.contrib.summary.scalar(
                'Train Accuracy', epoch_accuracy.result())
            tf.contrib.summary.scalar(
                'Validation Loss', epoch_loss_avg_val.result())
            tf.contrib.summary.scalar(
                'Validation Accuracy', epoch_accuracy_val.result())
            tf.contrib.summary.scalar(
                'Unsupervised Weight', unsupervised_weight)
            tf.contrib.summary.scalar('Learning Rate', learning_rate.numpy())
            tf.contrib.summary.scalar('Ramp Up Function', rampup_value)
            tf.contrib.summary.scalar('Ramp Down Function', rampdown_value)
            

    print('\nTrain Ended! Best Validation accuracy = {}\n'.format(best_val_accuracy))

    # Load the best model
    root = tfe.Checkpoint(optimizer=optimizer,
                          model=model,
                          optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # Evaluate on the final test set
    num_test_batches = math.ceil(NUM_TEST_SAMPLES/batch_size)
    test_accuracy = tfe.metrics.Accuracy()
    for test_batch in range(num_test_batches):
        X_test, y_test = test_iterator.get_next()
        y_test_predictions = model(X_test, training=False)
        test_accuracy(tf.argmax(y_test_predictions, 1), tf.argmax(y_test, 1))

    print("Final Test Accuracy: {:.6%}".format(test_accuracy.result()))


if __name__ == "__main__":
    main()
