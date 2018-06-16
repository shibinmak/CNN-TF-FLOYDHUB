import ImageData
import tensorflow as tf
from datetime import datetime
import numpy as np
import os

input_root = os.path.abspath('/input/')
output_root = os.path.abspath('/output/')
assert os.path.exists(output_root)
assert os.path.exists(input_root)
start = datetime.now()
np.random.seed(123)
tf.set_random_seed(123)

data = ImageData.train_test_split(image_size=128, test_size=0.3)


def weight_init(shape):
    init_weight= tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial_value=init_weight)


def bias_init(shape):
    init_bias= tf.constant(0.05, shape=shape)
    return tf.Variable(initial_value=init_bias)


def conv2d(input_image, kernel):
    """ convolution layer function,to be called in convolution layer
    input_image [batch,h,w,channels]
    filter [filter_height,filter_width,input_channel,output_channel]
    flattens filter to [fh*fw*ic,oc] , gives output [batch,out_height,out_width,fh*fw*ic]"""
    return tf.nn.conv2d(input=input_image, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')


def pooling(input_image):
    """expects a input 4d tensor [batch,height,width,channels]"""
    return tf.nn.max_pool(value=input_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolution_layer(input_image, shape):
    """z = wx+b ,z= (filter conv input_image) + bias
    followed by max pooling"""
    kernel = weight_init(shape=shape)
    bias = bias_init([shape[3]])
    """convolution"""
    layer = conv2d(input_image=input_image, kernel=kernel)+bias
    """pooling"""
    layer = pooling(layer)
    """activationfunction relu"""
    layer = tf.nn.relu(layer)

    return layer


def flatten_layer(layer):
    """number of feature maps will be h*w*c """
    layer_shape = layer.get_shape()
    number_of_feature_maps = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, (-1, number_of_feature_maps))
    return layer


def connected_layer(layer, output_size, relu=True):
    """ connects convolution layer and ann to form fully connected CNN"""
    input_size =layer.get_shape()[1:4].num_elements()
    weight = weight_init([input_size, output_size])
    bias = bias_init([output_size])
    layer= tf.matmul(layer, weight)+bias
    if relu:
        layer = tf.nn.relu(layer)

    return layer


#input image

x = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])

#labels


y_true = tf.placeholder(dtype=tf.float32, shape=[None, 3])
y_true_indexed = tf.argmax(y_true, axis=1)

convo1 = convolution_layer(input_image=x, shape=[3, 3, 3, 32])
convo2 = convolution_layer(input_image=convo1, shape=[3, 3, 32, 64])
convo3 = convolution_layer(input_image=convo2, shape=[3, 3, 64, 128])

flatten = flatten_layer(layer=convo3)

ann1 = connected_layer(flatten, 128)

ann2 = connected_layer(ann1, output_size=3, relu=False)

hold_prob = tf.placeholder(tf.float32)
annwithdropout = tf.nn.dropout(ann2, keep_prob=hold_prob)

y_pred = tf.nn.softmax(annwithdropout)
y_pred_indexed = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(cost)

prediction_rate = tf.equal(y_pred_indexed, y_true_indexed)
accuracy = tf.reduce_mean(tf.cast(prediction_rate, tf.float32))

saver = tf.train.Saver()
init= tf.global_variables_initializer()
dir_path = os.path.dirname(os.path.realpath('__file__'))
modelpath = os.path.join(output_root, 'model', 'model.ckpt')

total_iterations = 0


def model_train(iteration, batch_size):
    global total_iterations

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)

        for i in range(iteration):
            x_train, y_train, _, category_tarin = data.train.batch(batch_size=batch_size)
            x_test, y_test, _, category_test = data.test.batch(batch_size=batch_size)

            train_feed = {x: x_train, y_true: y_train, hold_prob: 0.5}
            test_feed = {x: x_test, y_true: y_test, hold_prob: 0.5}

            sess.run(train, feed_dict=train_feed)

            if i % int(data.train.total_images/batch_size) == 0:
                val_loss = sess.run(cost, feed_dict=test_feed)
                epoch = int(i/int(data.train.total_images/batch_size))
                accu = sess.run(accuracy, feed_dict=train_feed)
                val_accu = sess.run(accuracy, feed_dict=test_feed)
                status= "epoch: {}---Training Accuracy: {} validation Accuracy: {} validation loss:{}"
                print(status.format(epoch+1, accu, val_accu, val_loss))

            saver.save(sess, modelpath)
            total_iterations += iteration


model_train(iteration=10, batch_size=32)

stop = datetime.now()
print("total iteration : {}".format(total_iterations))
timetaken = stop-start

print("execution time: {}".format(timetaken))













