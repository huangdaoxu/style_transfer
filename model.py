import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.nets import vgg
from tensorflow.python.ops import variable_scope

from utils import MEAN_VALUES


def vgg_16(inputs,
           scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      net = layers_lib.repeat(
          inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
      # net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
      # net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
      # net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
    return net, end_points


def transfer_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME') as arg_sc:
            return arg_sc


def transfer_net(inputs, name="transfer", reuse=True):
    inputs = tf.pad(inputs, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
    with tf.variable_scope(name, reuse=reuse) as vs:
        net = slim.conv2d(inputs, 32, [9, 9], stride=1, scope='conv1')
        net = tf.nn.relu(slim.instance_norm(net))
        net = slim.conv2d(net, 64, [3, 3], stride=2, scope='conv2')
        net = tf.nn.relu(slim.instance_norm(net))
        net = slim.conv2d(net, 128, [3, 3], stride=2, scope='conv3')
        net = tf.nn.relu(slim.instance_norm(net))

        net = residual(net, 128, "residual1")
        net = residual(net, 128, "residual2")
        net = residual(net, 128, "residual3")
        net = residual(net, 128, "residual4")
        net = residual(net, 128, "residual5")

        net = deconv2d(net, 64, [3, 3], 1, scale=2, scope="conv4")
        net = tf.nn.relu(slim.instance_norm(net))
        net = deconv2d(net, 32, [3, 3], 1, scale=2, scope="conv5")
        net = tf.nn.relu(slim.instance_norm(net))
        net = deconv2d(net, 3, [9, 9], 1, scale=1, scope="conv6")
        net = tf.nn.tanh(slim.instance_norm(net))
        net = (net + 1.0) / 2 * 255.0

        variables = tf.contrib.framework.get_variables(vs)

        height = net.get_shape()[1].value
        width = net.get_shape()[2].value
        net = tf.slice(net, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))
        # net = tf.image.crop_to_bounding_box(net, 10, 10, height - 20, width - 20)
        return net, variables


def deconv2d(inputs, filters, kernel_size, strides, scale, scope):
    height, width = inputs.get_shape()[1].value, inputs.get_shape()[2].value

    h0 = tf.image.resize_images(inputs, [height * scale, width * scale],
                                tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return slim.conv2d(h0, filters, kernel_size, stride=strides, scope=scope)


def residual(inputs, filters, name):
    with tf.variable_scope(name_or_scope=name):
        h0 = slim.conv2d(inputs, filters, kernel_size=[3, 3], stride=1)
        h0 = tf.nn.relu(h0)
        h0 = slim.conv2d(h0, filters, kernel_size=[3, 3], stride=1)
    return tf.add(inputs, h0)


def build_model(inputs, style):
    with slim.arg_scope(transfer_arg_scope()):
        trans, var = transfer_net(inputs - MEAN_VALUES, reuse=False)

    inputs = tf.concat([trans, inputs, style], axis=0)

    # with slim.arg_scope(vgg.vgg_arg_scope()):
    #     _, end_points = vgg_16(inputs - MEAN_VALUES)

    _, end_points = vgg.vgg_16(inputs - MEAN_VALUES, spatial_squeeze=False)

    f1 = end_points["vgg_16/conv1/conv1_2"]
    f2 = end_points["vgg_16/conv2/conv2_2"]
    f3 = end_points["vgg_16/conv3/conv3_3"]
    f4 = end_points["vgg_16/conv4/conv4_3"]

    trans_f3, inputs_f3, _ = tf.split(f3, 3, 0)
    content_loss = 1.0*(tf.nn.l2_loss(trans_f3 - inputs_f3) / tf.to_float(tf.size(trans_f3)))

    style_loss = 200*styleloss(f1, f2, f3, f4)

    total_loss = content_loss + style_loss

    optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss, var_list=var)

    return optimizer, trans, total_loss, content_loss, style_loss


def block_v1(inputs, filters, name):
    with tf.variable_scope(name_or_scope=name):
        shortcut = slim.conv2d(inputs, filters, kernel_size=[1, 1], stride=1, scope="conv1")

        inputs = slim.conv2d(inputs, filters, kernel_size=[1, 1], stride=1, scope="conv2_1")

        inputs = slim.conv2d(inputs, filters, kernel_size=[3, 3], stride=1, scope="conv2_2")

        inputs = slim.conv2d(inputs, filters, kernel_size=[1, 1], stride=1, scope="conv2_3")
        inputs += shortcut
        return inputs


def styleloss(f1, f2, f3, f4):
    gen_f, _, style_f = tf.split(f1, 3, 0)
    size = tf.size(gen_f)
    style_loss = tf.nn.l2_loss(gram_matrix(gen_f) - gram_matrix(style_f))*2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f2, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram_matrix(gen_f) - gram_matrix(style_f)) * 2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f3, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram_matrix(gen_f) - gram_matrix(style_f)) * 2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f4, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram_matrix(gen_f) - gram_matrix(style_f)) * 2 / tf.to_float(size)

    return style_loss


def gram_matrix(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams

