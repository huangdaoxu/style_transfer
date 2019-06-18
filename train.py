import glob

import tensorflow as tf
from tensorflow.contrib import slim

from model import build_model
from utils import get_iterator, load_single_picture
from utils import MEAN_VALUES

style_pic = load_single_picture("/home/hdx/data/coco/landscape.jpg")

epoch = 50
batch_size = 4

inputs = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name="input")
style = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name="style")

iterator = get_iterator(glob.glob("/home/hdx/data/coco/val2017/*.jpg"), batch_size, epoch)

optimizer, trans, total_loss, content_loss, style_loss = build_model(inputs, style)

tf.summary.scalar('losses/total_loss', total_loss)
tf.summary.scalar('losses/content_loss', content_loss)
tf.summary.scalar('losses/style_loss', style_loss)
tf.summary.image('transformed', trans)
tf.summary.image('origin_without_mean', inputs - MEAN_VALUES)
tf.summary.image('origin', inputs)

summary = tf.summary.merge_all()

with tf.Session() as sess:
    # load pre-trained parameters
    vgg_vars = slim.get_variables_to_restore(include=['vgg_16'])
    variable_restore_op = slim.assign_from_checkpoint_fn("./vgg_16.ckpt",
                                                         vgg_vars,
                                                         ignore_missing_vars=True)
    variable_restore_op(sess)

    variables_to_save = slim.get_variables_to_restore(include=['transfer'])
    saver = tf.train.Saver(variables_to_save)

    all_var = tf.global_variables()
    init_var = [v for v in all_var if 'vgg_16' not in v.name]
    init = tf.variables_initializer(var_list=init_var)
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    style_pic = sess.run(style_pic)
    writer = tf.summary.FileWriter("./tensorboard/wave/", sess.graph)

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    counter = 0
    try:
        while not coord.should_stop():
            images = sess.run(iterator)
            feed_dict = {inputs: images,
                         style: [style_pic for _ in range(images.shape[0])]}
            sess.run([optimizer], feed_dict=feed_dict)
            counter += 1
            if counter % 10 == 0:
                result = sess.run(summary, feed_dict=feed_dict)
                print(counter)
                writer.add_summary(result, counter)

            if counter % 1000 == 0:
                # save model parameters
                saver.save(sess, ('trained_model/wave/' + 'model.ckpt'), global_step=counter)

    except tf.errors.OutOfRangeError:
        coord.request_stop()
        coord.join(thread)

    writer.close()

