import sys
import os
import cv2
import dlib

input_path = './input_img'
output_path = './other_facesss'
size = 64

# 判断路径是否存在
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 使用dlib自带的frontal_face_detector作为特征提取器
detector = dlib.get_frontal_face_detector()
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

index = 1
for (path, dirnames, filenames) in os.walk(input_path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('开始处理图片 %s' % index)
            img_path = path + '/' + filename
            # 读取图片
            img = cv2.imread(img_path)
            # 转换成灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用detector进行人脸检测
            dets = detector(gray_img, 1)

            # 使用enumerate 函数遍历序列中的元素以及它们的下标
            # 下标i即为人脸序号
            # left：人脸左边距离图片左边界的距离；right：人脸右边距离图片左边界的距离
            # top：人脸上边距离图片上边界的距离；bottom：人脸下边距离图片上边界的距离
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                face = img[x1:y1, x2:y2]
                # 调整图片大小
                face = cv2.resize(face, (size, size))
                cv2.imshow('image', face)
                # 保存图片
                cv2.imwrite(output_path + '/' + str(index) + '.jpg', face)
                index += 1

            key = cv2.waitKey(30) & 0xff
            # 按 ESC 键退出
            if key == 27:
                print('退出')
                break
# 关闭所有窗口
cv2.destroyAllWindows()