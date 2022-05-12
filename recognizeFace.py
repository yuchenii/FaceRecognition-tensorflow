import os
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import cv2
import dlib
import sys

size = 64

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 6])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnn():
    # 第一层
    W1 = weightVariable([3,3,3,32])
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([8*16*32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,6])
    bout = biasVariable([6])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

output = cnn()
results = tf.argmax(output, 1)
   
saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, tf.train.latest_checkpoint('my_net'))
   
def recognize(image):
    res = sess.run(results, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})
    if res[0] == 0:
        return '陈磊'
    if res[0] == 1:
        return '冯靖轩'
    if res[0] == 2:
        return '付迪'
    if res[0] == 3:
        return '高康悦'
    if res[0] == 4:
        return '匡斌'
    else:
        return 'unkuown'
    # return str(res)

#使用dlib.frontal_face_detector作特征提取器
detector = dlib.get_frontal_face_detector()

#mapping = "{'./chen_lei': 0, './feng_jing_xuan': 1, './fu_di': 2, './gao_kang_yue': 3, './kuang_bin': 4, './other_faces': 5}"


path = input("若检测视频输入正确路径否则为摄像头检测：")
if os.path.exists(path):
    cam = cv2.VideoCapture(path)
    print('找到对应文件，开始准备检测')
else:
    cam = cv2.VideoCapture(0)
    print('摄像头检测')

while True:
    _, img = cam.read()
    cv2.imshow('image', img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    if not len(dets):
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1,x2:y2]
        face = cv2.resize(face, (size,size))
        print('%s' % recognize(face))

        cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
        cv2.imshow('image',img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)
  
sess.close() 
