import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.nets import vgg
from tensorflow.python.ops import variable_scope

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


def transfer_net(inputs, name="transfer", reuse=True):
    with tf.variable_scope(name, reuse=reuse) as vs:
        net = layers_lib.repeat(inputs, 1, layers.conv2d, 32, [3, 3], scope='conv1')
        net = layers_lib.repeat(net, 2, layers.conv2d, 64, [3, 3], scope='conv2')
        net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv3')

        net = block_v1(net, 128, "residual1")
        net = block_v1(net, 128, "residual2")
        net = block_v1(net, 128, "residual3")
        net = block_v1(net, 128, "residual4")

        net = layers_lib.repeat(net, 2, layers.conv2d, 64, [3, 3], scope='conv4')
        net = layers_lib.repeat(net, 2, layers.conv2d, 32, [3, 3], scope='conv5')
        net = layers_lib.repeat(net, 1, layers.conv2d, 3, [3, 3], scope='conv6',
                                activation_fn=tf.nn.tanh)

        variables = tf.contrib.framework.get_variables(vs)

        return net, variables


def build_model(inputs, style):
    trans, var = transfer_net(inputs, reuse=False)

    inputs = tf.concat([trans, inputs, style], axis=0)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, end_points = vgg_16(inputs)

    f1 = end_points["vgg_16/conv1/conv1_2"]
    f2 = end_points["vgg_16/conv2/conv2_2"]
    f3 = end_points["vgg_16/conv3/conv3_3"]
    f4 = end_points["vgg_16/conv4/conv4_3"]

    trans_f3, inputs_f3, _ = tf.split(f3, 3, 0)
    content_loss = tf.nn.l2_loss(trans_f3 - inputs_f3) / tf.to_float(tf.size(trans_f3))

    style_loss = styleloss(f1, f2, f3, f4)

    total_loss = 0.25*content_loss + 0.75*style_loss

    optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss, var_list=var)

    return optimizer, trans, total_loss


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


def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)