import tensorflow as tf
import numpy as np
import os
import cv2

tf.set_random_seed(777)
batch_size = 100
epochs = 200
learning_rate = 0.0001

train_images = []
tlabels = []

neg_path = os.getcwd() + "/neg/neg_train"
counter = 0
for filename in os.listdir(neg_path):
    image = cv2.imread(neg_path+"/"+filename,0)
    image = cv2.resize(image,(70, 134))
    train_images.append(image)
    tlabels.append(0)
    counter += 1

pos_path = os.getcwd() + "/pos/pos_train"

for filename in os.listdir(pos_path):
    image = cv2.imread(pos_path+"/"+filename,0)
    train_images.append(image)
    tlabels.append(1)
    counter += 1

train_images = np.array(train_images)
train_images = train_images.reshape(counter, 9380, )

tlabels = np.array(tlabels)
tlabels = tlabels.reshape(counter,1)

train_labels  = np.array(np.zeros(counter*2).reshape(counter,2))
for num in range(counter):
    train_labels[num][int(tlabels[num][0]) - 1] = 1

test_images = []
testlabels = []

neg_path = os.getcwd() + "/neg/neg_test"
counter = 0
for filename in os.listdir(neg_path):
    image = cv2.imread(neg_path+"/"+filename,0)
    image = cv2.resize(image,(70, 134))
    test_images.append(image)
    testlabels.append(0)
    counter += 1

pos_path = os.getcwd() + "/pos/pos_test"

for filename in os.listdir(pos_path):
    image = cv2.imread(pos_path+"/"+filename,0)
    test_images.append(image)
    testlabels.append(1)
    counter += 1

test_images = np.array(test_images)
test_images = test_images.reshape(counter, 9380, )

testlabels = np.array(testlabels)
testlabels = testlabels.reshape(counter,1)

test_labels  = np.array(np.zeros(counter*2).reshape(counter,2))
for num in range(counter):
    test_labels[num][int(testlabels[num][0]) - 1] = 1

train_images = train_images / 255.
test_images =  test_images / 255.

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
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

print('Learning Started!')

# train my model
for epoch in range(epochs):
    avg_cost = 0
    total_batch = int(len(train_labels) / batch_size)

    for i in range(total_batch):
        start = i * batch_size
        end = i * batch_size + batch_size
        batch_xs, batch_ys = train_images[start:end], train_labels[start:end]
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    acc = m1.get_accuracy(test_images, test_labels)

    print('Accuracy:', acc)

    save_path = saver.save(sess, "C:\\Users\\Ryu\\PycharmProjects\\vss\\modeltrain.ckpt")


