logitsimport tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange

LEARNING_RATE = 0.05
EPOCHS = 15
INPUT_LAYER_SIZE = 784
HIDDEN_LAYER_SIZE = 625
NUM_CLASSES = 10

# initializes a weight using the shape provided
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

    train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    # The input and output matrixes are always placeholders
    # THESE VALUES WILL NOT BE EFFECTED BY BACKPROPAGATION ALGORITHM
    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])

    # THESE VALUES WILL BE EFFECTED BY BACKPROPAGATION AND INITIALIZED DIFFERENTLY
    hidden_weights = init_weights((INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE))
    output_weights = init_weights((HIDDEN_LAYER_SIZE, NUM_CLASSES))

    # Input layer weights gets matrix multiplied with the hidden layer
    hidden_layer_output = tf.nn.sigmoid(tf.matmul(X, hidden_weights))

    # LOGITS LAYER IS THE OUTPUT LAYER
    # hidden layer output gets matrix multiplied with the output weights
    logits = tf.matmul(hidden_layer_output, output_weights)

    # the cost or loss of the neural network is calculated, this loss is reduced by various optimizers
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))

    # this is the accuracy of the neural network used to calculate the accuracy later
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # optimizer is used to minimize the loss computed using the cross entropy function
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    prediction_optimizer = tf.argmax(logits_layer_output, 1)

    with tf.Session() as sess:
        # initializes all the global variables, since its written in c or c++, all the variables are initialized
        tf.global_variables_initializer().run()

        for epoch in trange(EPOCHS):
            counter = 0
            for start, end in zip(range(0, len(train_x), 128), range(128, len(train_x) + 1, 128)):
                counter += 1
                sess.run(optimizer, feed_dict = {X: train_x[start:end], Y: train_y[start: end]})
                if counter % 100 == 0:
                    # THIS IS HOW YOU CALCULATE THE LOSS OF A NEURAL NETWORK
                    # you need to pass in the loss to the sess.run to get the loss
                    l = sess.run(loss, feed_dict = {X: train_x[start:end], Y: train_y[start:end]})
                    print("Loss: {}".format(l))
                    # THIS IS HOW YOU CALCULATE THE ACCURACY
                    # a tuple will be returned, disregard the first element as it is used to write tensorboard summaries
                    a = sess.run([accuracy], feed_dict = {X: train_x[start:end], Y: train_y[start:end]})[0]
                    print("Accuracy: {}%".format(a * 100))
