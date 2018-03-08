""" Example ResNet50 with one Non-Local block. """
__author__ = "Lucas Beyer"
__license__ = "MIT"
__email__ = "lucasb.eyer.be@gmail.com"


import tensorflow as tf

from nets.nonlocal_resnet_utils import nonlocal_dot
from nets.resnet_utils import conv2d_same
from nets.resnet_v1 import bottleneck, resnet_arg_scope
slim = tf.contrib.slim

_RGB_MEAN = [123.68, 116.78, 103.94]


def endpoints(image, is_training):
  """ Send `image` through a ResNet50 with a non-local block at stage 3.

  Use like so:

      import nonlocal_resnet_v1_50_nl3 as model
      endpoints, body_prefix = model.endpoints(images, is_training=True)

      # BEFORE DEFINING THE OPTIMIZER:

      model_variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)

      # IF LOADING PRE-TRAINED WEIGHTS:

      saver = tf.train.Saver(model_variables)
      saver.restore(sess, args.initial_checkpoint)
  """

  if image.get_shape().ndims != 4:
    raise ValueError('Input must be of size [batch, height, width, 3]')

  image = image - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1,1,1,3))

  with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
    with tf.variable_scope('resnet_v1_50', values=[image]) as sc:
      end_points_collection = sc.name + '_end_points'
      with slim.arg_scope([slim.conv2d, bottleneck],
                          outputs_collections=end_points_collection):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
          net = image
          net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

          # NOTE: base_depth is that inside the bottleneck. i/o is 4x that.
          with tf.variable_scope('block1', values=[net]) as sc_block:
            with tf.variable_scope('unit_1', values=[net]):
              net = bottleneck(net, depth=4*64, depth_bottleneck=64, stride=1)
            with tf.variable_scope('unit_2', values=[net]):
              net = bottleneck(net, depth=4*64, depth_bottleneck=64, stride=1)
            with tf.variable_scope('unit_3', values=[net]):
              net = bottleneck(net, depth=4*64, depth_bottleneck=64, stride=2)

          with tf.variable_scope('block2', values=[net]) as sc_block:
            with tf.variable_scope('unit_1', values=[net]):
              net = bottleneck(net, depth=4*128, depth_bottleneck=128, stride=1)
            with tf.variable_scope('unit_2', values=[net]):
              net = bottleneck(net, depth=4*128, depth_bottleneck=128, stride=1)
            with tf.variable_scope('unit_3', values=[net]):
              net = bottleneck(net, depth=4*128, depth_bottleneck=128, stride=1)
            with tf.variable_scope('unit_4', values=[net]):
              net = bottleneck(net, depth=4*128, depth_bottleneck=128, stride=2)

          with tf.variable_scope('block3', values=[net]) as sc_block:
            with tf.variable_scope('unit_1', values=[net]):
              net = bottleneck(net, depth=4*256, depth_bottleneck=256, stride=1)
            with tf.variable_scope('unit_2', values=[net]):
              net = bottleneck(net, depth=4*256, depth_bottleneck=256, stride=1)
            with tf.variable_scope('unit_3', values=[net]):
              net = bottleneck(net, depth=4*256, depth_bottleneck=256, stride=1)
            with tf.variable_scope('unit_4', values=[net]):
              net = bottleneck(net, depth=4*256, depth_bottleneck=256, stride=1)
            with tf.variable_scope('unit_5', values=[net]):
              net = bottleneck(net, depth=4*256, depth_bottleneck=256, stride=1)
            with tf.variable_scope('unit_6', values=[net]):
              net = bottleneck(net, depth=4*256, depth_bottleneck=256, stride=2)

          net = nonlocal_dot(net, depth=512, embed=True, softmax=True, maxpool=2, scope='nonlocal3')

          with tf.variable_scope('block4', values=[net]) as sc_block:
            with tf.variable_scope('unit_1', values=[net]):
              net = bottleneck(net, depth=4*512, depth_bottleneck=512, stride=1)
            with tf.variable_scope('unit_2', values=[net]):
              net = bottleneck(net, depth=4*512, depth_bottleneck=512, stride=1)
            with tf.variable_scope('unit_3', values=[net]):
              net = bottleneck(net, depth=4*512, depth_bottleneck=512, stride=2)

        # Global average pooling.
        net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=False)
        # Convert end_points_collection into a dictionary of end_points.
        endpts = slim.utils.convert_collection_to_dict(
            end_points_collection)
        endpts['model_output'] = endpts['global_pool'] = net

    # The following is necessary to skip trying to load pre-trained non-local blocks.
    return endpts, 'resnet_v1_50/(?!nonlocal)'
