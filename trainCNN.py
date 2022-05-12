import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# 自动分配显存
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

feng_path = 'datasets/feng_jing_xuan'
gao_path = 'datasets/gao_kang_yue'
kuang_path = 'datasets/kuang_bin'
chen_path = 'datasets/chen_lei'
fu_path = 'datasets/fu_di'
other_faces_path = './other_faces'
size = 64

imgs = []
labs = []


def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right


# 读取图片并添加标签
def readData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top, bottom, left, right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)


def encodingLabel(labs):
    le = LabelEncoder()
    resultLabel = le.fit_transform(labs)
    mapping = dict(zip(le.classes_, range(0, len(le.classes_))))
    print(mapping)
    return resultLabel


readData(feng_path)
readData(gao_path)
readData(kuang_path)
readData(chen_path)
readData(fu_path)
readData(other_faces_path)

imgs = np.array(imgs)  # 转换成np格式
labs = encodingLabel(labs)  # 对标签进行编码
Labs = []
# 将编码数字改成one-hot
for lab in labs:
    if lab == 0:
        a = [1, 0, 0, 0, 0, 0]
    if lab == 1:
        a = [0, 1, 0, 0, 0, 0]
    if lab == 2:
        a = [0, 0, 1, 0, 0, 0]
    if lab == 3:
        a = [0, 0, 0, 1, 0, 0]
    if lab == 4:
        a = [0, 0, 0, 0, 1, 0]
    if lab == 5:
        a = [0, 0, 0, 0, 0, 1]
    Labs.append(a)

# 随机划分测试集与训练集
train_x, test_x, train_y, test_y = train_test_split(imgs, Labs, test_size=0.05, random_state=random.randint(0, 100))
# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# 将数据转换成小于1的数
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
# 每一个batch为100张图片
batch_size = 100
num_batch = len(train_x) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y = tf.placeholder(tf.float32, [None, 6])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)


def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dropout(x, keep):
    return tf.nn.dropout(x, keep)


def cnn():
    # 第一层
    W1 = weightVariable([3, 3, 3, 32])  # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([8 * 8 * 64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8 * 8 * 64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512, 6])
    bout = biasVariable([6])
    # out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out


def cnnTrain():
    out = cnn()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)), tf.float32))
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 初始化数据保存器
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        for r in range(1):  # 训练轮数
            print('Round', r + 1)
            for n in range(10):
                for i in range(num_batch):
                    batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                    batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                    # 开始训练数据
                    _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                                feed_dict={x: batch_x, y: batch_y, keep_prob_5: 0.5,
                                                           keep_prob_75: 0.75})
                    summary_writer.add_summary(summary, n * num_batch + i)

                    if (n * num_batch + i) % 100 == 0:
                        acc = accuracy.eval({x: test_x, y: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                        print('round: %s, batch: %s, accuracy: %s' % (str(r + 1), str(n), str(acc)))  # 输出每一个batch的准确度

        # 保存模型
        saver.save(sess, 'my_net/train_faces.model', global_step=n * num_batch + i)
        print('saved')


cnnTrain()
