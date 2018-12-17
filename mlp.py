import tensorflow as tf
import numpy as np
import csv
from math import floor
import import_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h1, w_h2, w_o,keep_prob):
    h1 = tf.nn.relu(tf.matmul(X, w_h1)) # this is a basic mlp, think 2 stacked logistic regressions\
    h1 = tf.nn.dropout(h1,keep_prob)
    h = tf.nn.relu(tf.matmul(h1, w_h2))
    h1 = tf.nn.dropout(h1,keep_prob)
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


numberOfTrainingData = len(import_data.X)
a = floor(numberOfTrainingData * 0.7)
a = int(a)
trX = import_data.X[:a]
trY = import_data.Y[:a]
teX = import_data.X[a:]
teY = import_data.Y[a:]

size_h1 = tf.constant(20, dtype=tf.int32)
size_h2 = tf.constant(15, dtype=tf.int32)

X = tf.placeholder("float", [None, 22])
Y = tf.placeholder("float", [None, 2])
keep_prob = tf.placeholder(tf.float32)

w_h1 = init_weights([22, size_h1]) # create symbolic variables
w_h2 = init_weights([size_h1, size_h2])
w_o = init_weights([size_h2, 2])

py_x = model(X, w_h1, w_h2, w_o, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(10000):
        sess.run(train_op, feed_dict={X: trX, Y: trY, keep_prob: 0.75})

        if i % 1000 == 0:
             print(i, 'training_arruracy:', accuracy.eval({X: teX, Y: teY, keep_prob: 1}))
    print('final_accuracy:', accuracy.eval({X: teX, Y: teY, keep_prob: 1}))

    saver.save(sess,"mlp/session.ckpt")
