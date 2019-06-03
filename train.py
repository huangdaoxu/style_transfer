import glob
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from model import build_model
from utils import get_iterator

style_pic = cv2.imread("/Users/hdx/Downloads/style1.jpg")
style_pic = cv2.resize(style_pic, (256, 256))

epoch = 5
current_epoch = 0
batch_size = 2

MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

inputs = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name="input")
style = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name="x")

iterator = get_iterator(glob.glob("/Users/hdx/code/python3/ocr/data/*.png"), 2, epoch)
optimizer, trans, total_loss = build_model(inputs, style)

tf.summary.scalar('losses/total_loss', total_loss)
tf.summary.image('transformed', trans)
tf.summary.image('origin', inputs)
tf.summary.image('style', style)

summary = tf.summary.merge_all()

# load pre-trained parameters
vgg_vars = slim.get_variables_to_restore(include=['vgg_16'])
variable_restore_op = slim.assign_from_checkpoint_fn("./vgg_16.ckpt",
                                                     vgg_vars,
                                                     ignore_missing_vars=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    writer = tf.summary.FileWriter("./tensorboard/", sess.graph)
    variable_restore_op(sess)

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    counter = 1
    try:
        while not coord.should_stop():
            images = sess.run(iterator)
            sess.run([optimizer], feed_dict={inputs: images, style: [style_pic for _ in range(images.shape[0])]})

    except tf.errors.OutOfRangeError:
        coord.request_stop()
        coord.join(thread)

    writer.close()

