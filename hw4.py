import tensorflow as tf
import h5py
import numpy as np
import scipy.misc
import cv2

tf.set_random_seed(777)
flag = True
batch_size = 49
epochs = 2000
learning_rate = 0.0001
a = 0

def one_hot_vector(label):
    num_labels = label.size
    num_classes = 14
    index_offset = np.arange(num_labels) * num_classes
    label_one_hot = np.zeros((num_labels, num_classes))
    label_one_hot.flat[index_offset + label.ravel()] = 1

    return label_one_hot

with h5py.File('kalph_train.hf', 'r') as hf:
    #images = [cv2.GaussianBlur(image, (3, 3), 0) for image in hf['images']]
    trainimages1 = np.reshape(np.array(hf['images']), [-1, 52, 52]) #52
    trainlabels1 = np.array(hf['labels'])

num_imgs, rows, cols = trainimages1.shape

with h5py.File('kalph_train.hf', 'r') as hf:
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 350, 1)
    images = [cv2.warpAffine(img, M, (cols, rows)) for img in hf['images']]

    trainimages2 = np.reshape(np.array(images), [-1, 52, 52]) #52, 28
    trainlabels2 = np.array(hf['labels'])

with h5py.File('kalph_train.hf', 'r') as hf:
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 352, 1)
    images = [cv2.warpAffine(img, M, (cols, rows)) for img in hf['images']]

    trainimages3 = np.reshape(np.array(images), [-1, 52, 52]) #52, 28
    trainlabels3 = np.array(hf['labels'])

with h5py.File('kalph_train.hf', 'r') as hf:
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 354, 1)
    images = [cv2.warpAffine(img, M, (cols, rows)) for img in hf['images']]

    trainimages4 = np.reshape(np.array(images), [-1, 52, 52]) #52, 28
    trainlabels4 = np.array(hf['labels'])

with h5py.File('kalph_train.hf', 'r') as hf:
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 356, 1)
    images = [cv2.warpAffine(img, M, (cols, rows)) for img in hf['images']]

    trainimages5 = np.reshape(np.array(images), [-1, 52, 52]) #52, 28
    trainlabels5 = np.array(hf['labels'])

with h5py.File('kalph_train.hf', 'r') as hf:
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 358, 1)
    images = [cv2.warpAffine(img, M, (cols, rows)) for img in hf['images']]

    trainimages6 = np.reshape(np.array(images), [-1, 52, 52]) #52, 28
    trainlabels6 = np.array(hf['labels'])

trainimages = np.concatenate([trainimages1, trainimages2, trainimages3, trainimages4, trainimages5, trainimages6])
trainlabels = np.concatenate([trainlabels1, trainlabels2, trainlabels3, trainlabels4, trainlabels5, trainlabels6])


with h5py.File('kalph_test.hf', 'r') as hf:
    #images = [cv2.GaussianBlur(image, (3, 3), 0) for image in hf['images']]
    testimages = np.reshape(np.array(hf['images']), [-1, 52, 52])
    testlabels = np.array(hf['labels'])

trainlabels = one_hot_vector(trainlabels)
testlabels = one_hot_vector(testlabels)

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
            self.X = tf.placeholder(tf.float32, [None, 52, 52])

            X_img = tf.reshape(self.X, [-1, 52, 52, 1])

            self.Y = tf.placeholder(tf.float32, [None, 14])

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
            flat = tf.reshape(dropout4, [-1, 256 * 4 * 4])

            dense5 = tf.layers.dense(inputs=flat,
                                     units=1000, activation=tf.nn.relu)
            dropout5 = tf.layers.dropout(inputs=dense5,
                                         rate=0.5, training=self.training)

            dense6 = tf.layers.dense(inputs=dropout5,
                                     units=1000, activation=tf.nn.relu)
            dropout6 = tf.layers.dropout(inputs=dense6,
                                         rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 14 outputs
            self.logits = tf.layers.dense(inputs=dropout6, units=14)

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

print('Learning Started!')

# train my model
for epoch in range(epochs):
    avg_cost = 0
    total_batch = int(num_imgs / batch_size)

    for i in range(total_batch):
        start = i * batch_size
        end = i * batch_size + batch_size
        batch_xs, batch_ys = trainimages1[start:end], trainlabels[start:end]
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))


#print('Learning Finished!')

# Test model and check accuracy
    acc = m1.get_accuracy(testimages, testlabels)
    if acc > 0.97 and flag:

        for i in range(len(testimages)):
            ttt = m1.get_accuracy(testimages[i:i+1], testlabels[i:i+1])
            if ttt == 0.0:
                scipy.misc.imsave('C:/Users/Ryu/Pictures/temp/outfile%d.jpg' % i, testimages[i])

        flag = False

    if acc >= a:
        a = acc

    print('Accuracy:', acc)

print(a)