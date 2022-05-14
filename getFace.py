import cv2
import dlib
import os
import random

# 图片输出目录
output_dir = './datasets/feng_jing_xuan'
# 输出图片大小
size = 64

# 判断输出目录是否存在，不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 改变图片的亮度与对比度，丰富人脸数据集
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(img[j, i, c] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, c] = tmp
    return img


# 使用dlib自带的frontal_face_detector作为特征提取器
detector = dlib.get_frontal_face_detector()
# 打开摄像头 参数为输入流，可以为摄像头或视频文件
camera = cv2.VideoCapture('./001.mp4')
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

index = 1
while True:
    if (index <= 600):
        print('捕获图片 %s' % index)
        # 从摄像头读取照片
        success, img = camera.read()
        # 显示照片
        cv2.imshow('frame', img)
        # 转为灰度图片
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(gray_img, 1)

        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
            cv2.imshow('frame', img)

            face = img[x1:y1, x2:y2]
            # 改变图片的亮度与对比度
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            # 调整图片大小
            face = cv2.resize(face, (size, size))
            # 保存图片到指定目录
            cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
            index += 1
        key = cv2.waitKey(30) & 0xff
        # 按 ESC 键退出
        if key == 27:
            print('退出')
            break
    else:
        print('Finished!')
        break

# 结束后释放VideoCapture对象
camera.release()
cv2.destroyAllWindows()
