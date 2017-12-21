import tensorflow as tf
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

    '''
        tf.argmax(input, axis=None, name=None, dimension=None)
        Returns the index with the largest value across axis of a tensor.

        input is a Tensor and axis describes which axis of the input Tensor to reduce across. For vectors, use axis = 0.

        For your specific case let's use two arrays and demonstrate this

        pred = np.array([[31, 23,  4, 24, 27, 34],
                        [18,  3, 25,  0,  6, 35],
                        [28, 14, 33, 22, 20,  8],
                        [13, 30, 21, 19,  7,  9],
                        [16,  1, 26, 32,  2, 29],
                        [17, 12,  5, 11, 10, 15]])

        y = np.array([[31, 23,  4, 24, 27, 34],
                        [18,  3, 25,  0,  6, 35],
                        [28, 14, 33, 22, 20,  8],
                        [13, 30, 21, 19,  7,  9],
                        [16,  1, 26, 32,  2, 29],
                        [17, 12,  5, 11, 10, 15]])
        Evaluating tf.argmax(pred, 1) gives a tensor whose evaluation will give array([5, 5, 2, 1, 3, 0])

        Evaluating tf.argmax(y, 1) gives a tensor whose evaluation will give array([5, 5, 2, 1, 3, 0])

        tf.equal(x, y, name=None) takes two tensors(x and y) as inputs and returns the truth value of (x == y) element-wise.
        Following our example, tf.equal(tf.argmax(pred, 1),tf.argmax(y, 1)) returns a tensor whose evaluation will givearray(1,1,1,1,1,1).

    '''
    # this is the accuracy of the neural network used to calculate the accuracy later
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # optimizer is used to minimize the loss computed using the cross entropy function
    optimizer = tf.train.AdamOptimizer().minimize(loss)


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
