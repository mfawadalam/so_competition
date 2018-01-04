from src.preprocess.converter import TrainData
from src.preprocess.read import JsonDataReader
import tensorflow as tf


def main():
    json_train = JsonDataReader("/Users/fawadalam/Documents/Kaggle/statoil_ccore/data/train.json").data
    numpy_converter = TrainData(json_train)

    X_train = numpy_converter.X_train[:,:10]
    Y_train = numpy_converter.Y_train
    X_test = numpy_converter.X_test[:,:10]
    Y_test = numpy_converter.Y_test

    print X_train.shape
    print Y_train.shape
    print Y_train.sum()
    print X_test.shape
    print Y_test.shape
    print Y_test.sum()

    print X_train.dtype
    print Y_train.dtype
    print X_test.dtype
    print Y_test.dtype
    sess = tf.InteractiveSession()
    X_train_ph = tf.placeholder(tf.float32,shape=[None,10])
    Y_train_ph = tf.placeholder(tf.float32,shape=[None,1])

    W1 = tf.Variable(tf.random_normal([10,1]))
    b1 = tf.Variable(tf.random_normal([1]))
    #
    # W2 = tf.Variable(tf.truncated_normal([200,50], stddev=0.1))
    # b2 = tf.Variable(tf.zeros([50]))
    #
    # W3 = tf.Variable(tf.truncated_normal([50,10], stddev=0.1))
    # b3 = tf.Variable(tf.zeros([10]))
    #
    # W4 = tf.Variable(tf.truncated_normal([10,1], stddev=0.1))
    # b4 = tf.Variable(tf.zeros([1]))

    A1 = tf.nn.sigmoid(tf.matmul(X_train_ph, W1, name="X_x_W1") + b1)
    # A2 = tf.nn.tanh(tf.add(tf.matmul(A1, W2, name="A1_x_W2"), b2, name="A1_x_W2b2"), name="relu2")
    # A3 = tf.nn.tanh(tf.add(tf.matmul(A2, W3, name="A2_x_W3"), b3, name="A2_x_W3b3"), name="relu3")
    # Y_ = tf.nn.softmax(tf.add(tf.matmul(A3, W4, name="A3_x_W4"), b4, name="A3_x_W4b4"), name="sigmoid4")

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_train_ph * tf.log(A1),axis=1))
    cross_entropy = -tf.reduce_sum(Y_train_ph*tf.log(tf.clip_by_value(A1,1e-10,1.0)))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    tf.global_variables_initializer().run()

    for i in range(1000):
        sess.run(train_step,{X_train_ph:X_train, Y_train_ph: Y_train})
        print(sess.run(cross_entropy, {X_train_ph:X_train, Y_train_ph: Y_train}),
              sess.run(cross_entropy, {X_train_ph:X_test, Y_train_ph: Y_test}))
        # print(sess.run(cross_entropy, {X_train_ph:X_test, Y_train_ph: Y_test}))

main()