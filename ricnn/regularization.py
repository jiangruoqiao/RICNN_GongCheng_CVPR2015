#ecoding:utf-8
import tensorflow as tf
def regu_constraint(raw_feature, aver_feature):
    constraint = tf.nn.l2_loss(raw_feature - aver_feature)
    return constraint