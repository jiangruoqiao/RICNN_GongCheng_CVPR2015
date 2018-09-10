import math
import tensorflow as tf
from tensorflow.contrib import layers

regularizer = layers.l2_regularizer(0.0005)

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
beta = beta, bias = bias, name = name)

def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],strides = [1, strideX, strideY, 1], padding = padding, name = name)

def conv2d(x, strideX, strideY, W):
  return tf.nn.conv2d(x, W, strides=[1, strideX, strideY, 1], padding='SAME')

def weights_biases(kernel_shape, bias_shape):
  in_variables = 1
  for index in range(len(kernel_shape) - 1):
    in_variables *= kernel_shape[index]
  stdv = 1.0 / math.sqrt(in_variables)
  with tf.variable_scope('wei', regularizer = regularizer):
      weights = tf.get_variable(
          'weights',
          kernel_shape,
          initializer=tf.random_uniform_initializer(-stdv, stdv))
  with tf.variable_scope('bias'):
      biases = tf.get_variable(
          'biases',
          bias_shape,
          initializer=tf.random_uniform_initializer(-stdv, stdv))
  return weights, biases

def conv_relu(input, strideX, strideY, kernel_shape, bias_shape):
  weights, biases = weights_biases(kernel_shape, bias_shape)
  return tf.nn.relu(conv2d(input, strideX, strideY, weights) + biases)

def fc_relu(input, kernel_shape, bias_shape):
  weights, biases = weights_biases(kernel_shape, bias_shape)
  return tf.nn.relu(tf.matmul(input, weights) + biases)

def fc(input, kernel_shape, bias_shape):
  weights, biases = weights_biases(kernel_shape, bias_shape)
  return tf.matmul(input, weights) + biases

# x should already be reshaped as a 32x32x1 image
def single_branch(x, keep_prob, number_of_fc_features):
  with tf.variable_scope('conv1'):
    conv1 = conv_relu(x, 4, 4, [11, 11, 1, 96], [96])
    lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
    pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

  with tf.variable_scope('conv2'):
    conv2 = conv_relu(pool1, 1, 1, [5, 5, 96, 256], [256])
    lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
    pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

  with tf.variable_scope('conv3'):
    conv3 = conv_relu(pool2, 1, 1, [3, 3, 256, 384],[384])

  with tf.variable_scope('conv4'):
    conv4 = conv_relu(conv3, 1, 1, [3, 3, 384, 384],[384])

  with tf.variable_scope('conv5'):
    conv5 = conv_relu(conv4, 1, 1, [3, 3, 384, 256],[256])
    pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")
    flattened_size = 6*6*256
    flattened = tf.reshape(pool5, [-1, flattened_size])

  with tf.variable_scope('fc6'):
    fc6 = fc_relu(flattened, [6*6*256, 4096], [4096])
    drop1 = tf.nn.dropout(fc6, keep_prob)

  with tf.variable_scope('fc7'):
    fc7 = fc_relu(drop1, [4096, 4096], [4096])
    drop2 = tf.nn.dropout(fc7, keep_prob)

  with tf.variable_scope('fca'):
    fca = fc_relu(drop2, [4096, 10], [10])
  return fca

def define_model(x,
                 keep_prob,
                 number_of_classes,
                 number_of_filters,
                 number_of_fc_features):
  splitted = tf.unstack(x, axis=4)
  K = len(splitted)
  branches = []
  with tf.variable_scope('branches') as scope:
    raw_feature = single_branch(splitted[0],keep_prob,number_of_fc_features)
    scope.reuse_variables()
    for index, tensor_slice in enumerate(splitted):
      if (index == K - 1):
          break
      branches.append(single_branch(splitted[index+1],
                                    keep_prob,
                                    number_of_fc_features))
      scope.reuse_variables()
    concatenated = tf.stack(branches, axis=2)
    ti_pooled = tf.reduce_mean(concatenated, reduction_indices=[2])
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return ti_pooled, raw_feature, regularization_loss