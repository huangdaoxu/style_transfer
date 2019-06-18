import tensorflow as tf
import numpy as np


MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3])


def load_image(filename, width=256, height=256):
    image_raw_data = tf.gfile.FastGFile(filename, 'rb').read()
    image = tf.image.decode_jpeg(image_raw_data, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, size=(width, height))
    image *= 255.0
    return image


def get_iterator(path, batch_size, num_epochs, width=256, height=256):
    path = tf.convert_to_tensor(path, dtype=tf.string)
    images_queue = tf.train.string_input_producer(path, num_epochs=num_epochs)

    reader = tf.WholeFileReader()
    _, value = reader.read(images_queue)

    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(value, channels=3), tf.float32)
    image = tf.image.resize_images(image, size=(width, height))
    image *= 255.0
    image_batch = tf.train.batch([image], batch_size=batch_size)
    return image_batch

