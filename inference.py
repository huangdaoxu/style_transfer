import os

import tensorflow as tf

from utils import load_image
from matplotlib import pyplot as plt

flags = tf.app.flags
############################
#    hyper parameters      #
############################
flags.DEFINE_string("test_image", "./test/sunshine_boy.jpeg", "test image path")
flags.DEFINE_string("model_path", "./trained_model/starry/model.ckpt-49000", "meta path")
flags.DEFINE_string("saved_path", "./test/inference_starry.jpg", "transfromed image path")

FLAGS = tf.app.flags.FLAGS

def main(_):
    test_image = load_image(FLAGS.test_image)
    with tf.Session() as sess:
        test_image = sess.run(test_image)
        saver = tf.train.import_meta_graph(FLAGS.model_path + ".meta")
        saver.restore(sess, FLAGS.model_path)
        graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name("input:0")
        y = graph.get_operation_by_name("transfer/Slice").outputs[0]
        y = tf.cast(y, tf.uint8)
        img = sess.run(y, feed_dict={inputs: [test_image]})
        image_encode = tf.image.resize_images(img[0], (256, 256), method=1)
        image_encode = tf.image.convert_image_dtype(image_encode, dtype=tf.uint8)
        image_encode = tf.image.encode_jpeg(image_encode)
        image_saver = tf.gfile.FastGFile(FLAGS.saved_path, "w+")
        image_saver.write(sess.run(image_encode))
        image_saver.close()


if __name__ == '__main__':
    tf.app.run()
