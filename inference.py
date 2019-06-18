import tensorflow as tf

from utils import load_image

flags = tf.app.flags
############################
#    hyper parameters      #
############################
flags.DEFINE_string("test_image", "./test/sunshine_boy.jpeg", "test image path")
flags.DEFINE_string("model_path", "./trained_model/landscape/model.ckpt-15000", "meta path")
flags.DEFINE_string("saved_path", "./test/inference_landscape.jpg", "transfromed image path")

FLAGS = tf.app.flags.FLAGS

# def main(_):
#     test_image = load_image(FLAGS.test_image)
#     with tf.Session() as sess:
#         test_image = sess.run(test_image)
#         saver = tf.train.import_meta_graph(FLAGS.model_path + ".meta")
#         saver.restore(sess, FLAGS.model_path)
#         graph = tf.get_default_graph()
#         inputs = graph.get_tensor_by_name("input:0")
#         y = graph.get_operation_by_name("transfer/Slice").outputs[0]
#         y = tf.cast(y, tf.uint8)
#         img = sess.run(y, feed_dict={inputs: [test_image]})
#         image_encode = tf.image.resize_images(img[0], (256, 256), method=1)
#         image_encode = tf.image.convert_image_dtype(image_encode, dtype=tf.uint8)
#         image_encode = tf.image.encode_jpeg(image_encode)
#         image_saver = tf.gfile.FastGFile(FLAGS.saved_path, "w+")
#         image_saver.write(sess.run(image_encode))
#         image_saver.close()



import numpy as np
from os.path import join
from scipy.misc import imread, imresize, imsave

from model import transfer_net, transfer_arg_scope
from utils import MEAN_VALUES
from tensorflow.contrib import slim

def main(_):
    image_raw_data = tf.gfile.FastGFile(FLAGS.test_image, 'rb').read()
    image = tf.image.decode_jpeg(image_raw_data, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image *= 255.0
    with tf.Session() as sess:
        image = sess.run(image)
    t_shape = image.shape

    with tf.Graph().as_default():
        test_image = tf.placeholder(tf.float32, [None, t_shape[0], t_shape[1], t_shape[2]])
        with slim.arg_scope(transfer_arg_scope()):
            generated_image, _ = transfer_net(test_image - MEAN_VALUES, reuse=False)
        squeezed_generated_image = tf.squeeze(generated_image, [0])

        restorer = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            restorer.restore(sess, FLAGS.model_path)
            styled_image = sess.run(squeezed_generated_image, feed_dict={test_image: [image]})
            imsave(FLAGS.saved_path, np.squeeze(styled_image))


if __name__ == '__main__':
    tf.app.run()
