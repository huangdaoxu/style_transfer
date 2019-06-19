import tensorflow as tf

from tensorflow.contrib import slim

from utils import load_image, MEAN_VALUES
from model import transfer_arg_scope, transfer_net

flags = tf.app.flags
############################
#    hyper parameters      #
############################
flags.DEFINE_string("test_image", "./test/sunshine_boy.jpeg", "test image path")
flags.DEFINE_string("model_path", "./trained_model/starry/model.ckpt-49000", "meta path")
flags.DEFINE_string("saved_path", "./test/inference_starry.jpg", "transfromed image path")

FLAGS = tf.app.flags.FLAGS

def main(_):
    test_image = load_image(FLAGS.test_image, resize=False)

    sess = tf.Session()
    test_image = sess.run(test_image)
    shape = test_image.shape

    # build new input with trained parameters and weights
    inputs = tf.placeholder(tf.float32, [None, shape[0], shape[1], shape[2]])
    with slim.arg_scope(transfer_arg_scope()):
        generated_image, _ = transfer_net(inputs - MEAN_VALUES, reuse=False)
    generated_image = tf.squeeze(generated_image, [0])

    # restore trained parameters and weights
    restorer = tf.train.Saver()
    restorer.restore(sess, FLAGS.model_path)

    # save the transformed image
    generated_image = sess.run(generated_image, feed_dict={inputs: [test_image]})
    generated_image = tf.image.encode_jpeg(generated_image)
    image_saver = tf.gfile.FastGFile(FLAGS.saved_path, "w+")
    image_saver.write(sess.run(generated_image))
    image_saver.close()


if __name__ == '__main__':
    tf.app.run()
