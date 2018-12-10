#import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data

imported_data = np.loadtxt("vehicle.csv", delimiter=",", dtype=str)
'''data = np.array([imported_data[0]])
for i in range(1, data.shape[0]):
    data = np.vstack((data, imported_data[i]), axis=0)'''

def reshape_input(input_data):
    output = np.array([[input_data[0]]])
    for i in range(1, input_data.shape[0]):
        output = np.vstack((output, np.array([[input_data[i]]])))
    return output

data = reshape_input(imported_data)

# To get a vector of labels: data.T[18].T
# To get a single case vector: data[n].T


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp, think 2 stacked logistic regressions
    h = tf.nn.sigmoid(tf.matmul(h1, w_h2))
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


size_h1 = tf.constant(625, dtype=tf.int32)
size_h2 = tf.constant(300, dtype=tf.int32)

with name_scope("Inputs") as scope:
    X = tf.placeholder("float", [None, 19])
    Y = tf.placeholder("float", [None, 10])

with name_scope("Weights") as scope:
    w_h1 = init_weights([784, size_h1]) # create symbolic variables
    w_h2 = init_weights([size_h1, size_h2])
    w_o = init_weights([size_h2, 10])

with name_scope("Model") as scope:
    py_x = model(X, w_h1, w_h2, w_o)

with name_scope("Functions") as scope:
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