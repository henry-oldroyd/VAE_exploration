# using https://www.tensorflow.org/tutorials/generative/cvae

import tensorflow as tf
import numpy as np


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


def preprocess_images(images):
    print(images.shape)

    # Reshape images to (28, 28, 1) and normalize
    # images = images.reshape((images.shape[0], 28, 28, 1))
    images = images / 255
    return images


train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)


train_labels = tf.one_hot(train_labels, 10)
test_labels = tf.one_hot(test_labels, 10)

train_size = 60000
batch_size = 32
test_size = 10000


train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    .shuffle(train_size)
    .batch(batch_size)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    .shuffle(test_size)
    .batch(batch_size)
)
