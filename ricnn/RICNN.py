#ecoding:utf-8
import DatasetLoader
import RICNNModel
import tensorflow as tf
import sys
import numpy as np
import regularization as re
import os
import trainLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

TRAIN_FILENAME = '/media/liuqi/Files/dataset/test_mnist_ricnn_raw_100.h5'
TEST_FILENAME = '/media/liuqi/Files/dataset/test_mnist_ricnn_raw.h5'
TRAIN_LABELS = '/media/liuqi/Files/dataset/rotate_100_simple.h5'
TEST_LABELS = '/home/liuqi/Desktop/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat'

LOADED_SIZE = 28
DESIRED_SIZE = 227
# model constants
NUMBER_OF_CLASSES = 10
NUMBER_OF_FILTERS = 40
NUMBER_OF_FC_FEATURES = 5120
NUMBER_OF_TRANSFORMATIONS = 8
# optimization constants
BATCH_SIZE = 64
TEST_CHUNK_SIZE = 100
ADAM_LEARNING_RATE = 1e-5
PRINTING_INTERVAL = 10
# set seeds
np.random.seed(100)
tf.set_random_seed(100)
x = tf.placeholder(tf.float32, shape=[None,
                                      DESIRED_SIZE,
                                      DESIRED_SIZE,
                                      1,
                                      NUMBER_OF_TRANSFORMATIONS])
y_gt = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES])
keep_prob = tf.placeholder(tf.float32)
logits, raw_feature, regularization_loss = RICNNModel.define_model(x,
                                 keep_prob,
                                 NUMBER_OF_CLASSES,
                                 NUMBER_OF_FILTERS,
                                 NUMBER_OF_FC_FEATURES)
with tf.name_scope('loss'):
    with tf.name_scope('re_loss'):
        re_loss = re.regu_constraint(raw_feature, logits)
    with tf.name_scope('sotfmax_loss'):
        sotfmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_gt))
    with tf.name_scope('total_loss'):
        total_loss = sotfmax_loss

train_step = tf.train.AdamOptimizer(ADAM_LEARNING_RATE).minimize(total_loss)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_gt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session = tf.Session()
session.run(tf.initialize_all_variables())
train_data_loader = trainLoader.DataLoader(TRAIN_FILENAME,
                                     TRAIN_LABELS,
                                     NUMBER_OF_CLASSES,
                                     NUMBER_OF_TRANSFORMATIONS,
                                     LOADED_SIZE,
                                     DESIRED_SIZE)
test_data_loader = DatasetLoader.DataLoader(TEST_FILENAME,
                                    TEST_LABELS,
                                    NUMBER_OF_CLASSES,
                                    NUMBER_OF_TRANSFORMATIONS,
                                    LOADED_SIZE,
                                    DESIRED_SIZE)
test_size = test_data_loader.all()[1].shape[0]
assert test_size % TEST_CHUNK_SIZE == 0
number_of_test_chunks = test_size / TEST_CHUNK_SIZE

while (True):
  batch = train_data_loader.next_batch(BATCH_SIZE) # next_batch from the loader
  txt_name = "accary_ricnn.txt"
  txt_file = file(txt_name, "a+")
  if (train_data_loader.is_new_epoch()):
    train_accuracy = session.run(accuracy, feed_dict={x : batch[0],
                                                      y_gt : batch[1],
                                                      keep_prob : 1.0})
    print_loss = session.run(re_loss,feed_dict={x : batch[0],
                                                      y_gt : batch[1],
                                                      keep_prob : 1.0})
    print_loss_1 = session.run(sotfmax_loss, feed_dict={x: batch[0],
                                                 y_gt: batch[1],
                                                 keep_prob: 1.0})
    print(print_loss)
    print(print_loss_1)
    train_context = "epochs:" + str(train_data_loader.get_completed_epochs()) + '\n'
    txt_file.write(train_context)
    loss_context = "softmax_loss:" + str(print_loss_1) + '\n'
    txt_file.write(loss_context)
    txt_file.close()
    print("completed_epochs %d, training accuracy %g" %
          (train_data_loader.get_completed_epochs(), train_accuracy))
    sys.stdout.flush()

    if (train_data_loader.get_completed_epochs() % PRINTING_INTERVAL == 0):
      sum = 0.0
      xt_name = "accary_ricnn.txt"
      txt_file = file(txt_name, "a+")
      for chunk_index in xrange(number_of_test_chunks):
        chunk = test_data_loader.next_batch(TEST_CHUNK_SIZE)
        sum += session.run(accuracy, feed_dict={x : chunk[0],
                                                y_gt : chunk[1],
                                                keep_prob : 1.0})
      test_accuracy = sum / number_of_test_chunks
      new_context = "testing accuracy: " + str(test_accuracy) + '\n'
      txt_file.write(new_context)
      txt_file.close()
      print("testing accuracy %g" % test_accuracy)
      sys.stdout.flush()
  session.run(train_step, feed_dict={x : batch[0],
                                     y_gt : batch[1],
                                     keep_prob : 0.5})
