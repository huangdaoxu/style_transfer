import glob

import tensorflow as tf
from tensorflow.contrib import slim

from model import build_model
from utils import get_iterator, load_image

flags = tf.app.flags
############################
#    hyper parameters      #
############################
flags.DEFINE_integer("epoch", 50, "number of training epoch")
flags.DEFINE_integer("batch_size", 4, "batch size")
flags.DEFINE_float('learning_rate', 0.002, "learning rate")
flags.DEFINE_float('content_loss_weight', 1.0, "content loss weight")
flags.DEFINE_float('style_loss_weight', 100.0, "style loss weight")
flags.DEFINE_string("summary_path", "./tensorboard/wave/", "tensorboard file path")
flags.DEFINE_string("vgg_path", "./vgg_16.ckpt", "pre-trained vgg file path")
flags.DEFINE_string("model_path", "./trained_model/wave/model.ckpt", "model path")
flags.DEFINE_string("style_image_path", "./style_images/wave.jpg", "style image path")
flags.DEFINE_string("train_dataset", "/home/hdx/data/coco/val2017/*.jpg", "dataset for fit")

FLAGS = tf.app.flags.FLAGS

def main(_):
    # create input tensor
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name="input")
    style = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name="style")

    # init image data
    style_image = load_image(FLAGS.style_image_path)
    iterator = get_iterator(
        glob.glob(FLAGS.train_dataset),
        FLAGS.batch_size, FLAGS.epoch)

    #build transfer model
    optimizer, trans, total_loss, content_loss, style_loss = \
        build_model(inputs, style, FLAGS.learning_rate,
                    FLAGS.content_loss_weight, FLAGS.style_loss_weight)

    with tf.Session() as sess:
        # load pre-trained parameters
        vgg_vars = slim.get_variables_to_restore(include=['vgg_16'])
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.vgg_path,
                                                             vgg_vars,
                                                             ignore_missing_vars=True)
        variable_restore_op(sess)

        # get trainable parameters
        variables_to_save = slim.get_variables_to_restore(include=['transfer'])
        saver = tf.train.Saver(variables_to_save)

        all_var = tf.global_variables()
        init_var = [v for v in all_var if 'vgg_16' not in v.name]
        init = tf.variables_initializer(var_list=init_var)
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        style_image = sess.run(style_image)

        # config visualization parameters
        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.scalar('losses/content_loss', content_loss)
        tf.summary.scalar('losses/style_loss', style_loss)
        tf.summary.image('transformed', trans)
        tf.summary.image('origin', inputs)
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(FLAGS.summary_path, sess.graph)

        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=sess, coord=coord)
        counter = 0
        try:
            while not coord.should_stop():
                images = sess.run(iterator)
                feed_dict = {inputs: images,
                             style: [style_image for _ in range(images.shape[0])]}
                sess.run([optimizer], feed_dict=feed_dict)
                counter += 1

                if counter % 10 == 0:
                    result = sess.run(summary, feed_dict=feed_dict)
                    # update summary
                    writer.add_summary(result, counter)

                if counter % 1000 == 0:
                    # save model parameters
                    saver.save(sess, FLAGS.model_path, global_step=counter)

        except tf.errors.OutOfRangeError:
            coord.request_stop()
            coord.join(thread)

        writer.close()


if __name__ == '__main__':
    tf.app.run()



