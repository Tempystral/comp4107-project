import tensorflow as tf
import numpy as np
import csv
from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp, think 2 stacked logistic regressions
    h = tf.nn.sigmoid(tf.matmul(h1, w_h2))
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

def onehot(type):
    result = []
    if type == 'van':
        result = np.asarray([1,0,0,0])
    elif type == 'bus':
        result = np.asarray([0,1,0,0])
    elif type == 'saab':
        result = np.asarray([0,0,1,0])
    elif type == 'opel':
        result = np.asarray([0,0,0,1])
    return result;

def load_data():
    datas = np.empty((1, 18))
    labels = np.empty((1, 4))
    with open('vehicle.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            int_data = map(int, line[:18])          # convert string to int
            temp = np.asarray(int_data)
            datas = np.vstack((datas,temp))

            labels = np.vstack((labels,onehot(line[18])))
    return datas[1:],labels[1:]

trX = load_data()[0][:600]
trY = load_data()[1][:600]
teX = load_data()[0][600:]
teY = load_data()[1][600:]

print(trY.shape)

size_h1 = tf.constant(50, dtype=tf.int32)
size_h2 = tf.constant(30, dtype=tf.int32)

X = tf.placeholder("float", [None, 18])
Y = tf.placeholder("float", [None, 4])

w_h1 = init_weights([18, size_h1]) # create symbolic variables
w_h2 = init_weights([size_h1, size_h2])
w_o = init_weights([size_h2, 4])

py_x = model(X, w_h1, w_h2, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(3):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        accuracy = np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX}))
        print(i, accuracy)

    saver.save(sess,"mlp/session.ckpt")
