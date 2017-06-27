import tensorflow as tf
import numpy as np
import cv2

tf.set_random_seed(777)
batch_size = 100
epochs = 200
learning_rate = 0.0001

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 9380])

            X_img = tf.reshape(self.X, [-1, 134, 70, 1])

            self.Y = tf.placeholder(tf.float32, [None, 2])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.5, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.5, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.5, training=self.training)

            conv4 = tf.layers.conv2d(inputs=dropout3, filters=256, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout4 = tf.layers.dropout(inputs=pool4,
                                         rate=0.5, training=self.training)

            print(dropout4)

            # Dense Layer with Relu
            flat = tf.reshape(dropout4, [-1, 256 * 9 * 5])

            dense5 = tf.layers.dense(inputs=flat,
                                     units=1000, activation=tf.nn.relu)
            dropout5 = tf.layers.dropout(inputs=dense5,
                                         rate=0.5, training=self.training)

            dense6 = tf.layers.dense(inputs=dropout5,
                                     units=1000, activation=tf.nn.relu)
            dropout6 = tf.layers.dropout(inputs=dense6,
                                         rate=0.5, training=self.training)

            self.logits = tf.layers.dense(inputs=dropout6, units=2)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={ self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "C:\\Users\\Ryu\\PycharmProjects\\vss\\modeltrain.ckpt")

test = []
test1 = cv2.imread("C:\\Users\\Ryu\\PycharmProjects\\vss\\FudanPed00026.png", 0)
test.append(test1)
test.append(cv2.resize(test1, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA))
test.append(cv2.resize(test1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
test.append(cv2.resize(test1, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA))
test.append(cv2.resize(test1, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA))

test = np.array(test)

original_image = cv2.imread("C:\\Users\\Ryu\\PycharmProjects\\vss\\FudanPed00026.png", cv2.IMREAD_COLOR)

rate = [1, 1.43, 2, 3.33, 10]

for i in range(len(test)):
    img = test[i]

    for colIndex in range(0, test[i].shape[0]-134, int(134 * 0.25)):
        for rowIndex in range(0, test[i].shape[1]-70, int(70 * 0.25)):
            crop_img = img[colIndex:colIndex + 134, rowIndex:rowIndex + 70]
            crop_img = np.reshape(crop_img, [1, 9380])

            if 1.0 == m1.get_accuracy(crop_img, [[1, 0]]):
                cv2.rectangle(original_image, (int((rowIndex+70) * rate[i]), int((colIndex+134) * rate[i])), (int(rowIndex * rate[i]), int(colIndex * rate[i])), (0, 0, 255), 2)

cv2.imshow('test',original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

