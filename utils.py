import tensorflow as tf


MEAN_VALUES = [123.68, 116.779, 103.939]


def load_single_picture(filename, width=256, height=256):
    image_raw_data = tf.gfile.FastGFile(filename, 'rb').read()
    image = tf.image.decode_jpeg(image_raw_data, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, size=(width, height))
    #image = _mean_image_subtraction(image)
    return image


def get_iterator(path, batch_size, num_epochs, width=256, height=256):
    path = tf.convert_to_tensor(path, dtype=tf.string)
    images_queue = tf.train.string_input_producer(path, num_epochs=num_epochs)

    reader = tf.WholeFileReader()
    _, value = reader.read(images_queue)

    imgs = tf.image.convert_image_dtype(tf.image.decode_jpeg(value, channels=3), tf.float32)
    imgs = tf.image.resize_images(imgs, size=(width, height))
    #imgs = _mean_image_subtraction(imgs)

    image_batch = tf.train.batch([imgs], batch_size=batch_size)
    return image_batch


def _mean_image_subtraction(image, means=MEAN_VALUES):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, 2)
