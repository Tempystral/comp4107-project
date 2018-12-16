import tensorflow as tf
import numpy as np
import csv
import import_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp, think 2 stacked logistic regressions
    h = tf.nn.sigmoid(tf.matmul(h1, w_h2))
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

trX = import_data.X[:7000]
trY = import_data.Y[:7000]
teX = import_data.X[7000:]
teY = import_data.Y[7000:]

size_h1 = tf.constant(160, dtype=tf.int32)
size_h2 = tf.constant(83, dtype=tf.int32)

X = tf.placeholder("float", [None, 22])
Y = tf.placeholder("float", [None, 2])

w_h1 = init_weights([22, size_h1]) # create symbolic variables
w_h2 = init_weights([size_h1, size_h2])
w_o = init_weights([size_h2, 2])

py_x = model(X, w_h1, w_h2, w_o)

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
        sess.run(train_op, feed_dict={X: trX, Y: trY})

        if i % 1000 == 0:
             print(i, 'training_arruracy:', accuracy.eval({X: teX, Y: teY}))
    print('final_accuracy:', accuracy.eval({X: teX, Y: teY}))

    saver.save(sess,"mlp/session.ckpt")
