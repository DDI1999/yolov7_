import logging
import os
import time

import cv2
import torch

# x = torch.linspace(1, 180, steps=180).view(3, 15, 2, 2)  # 设置一个三维数组
# print(x)
# print(x.size())  # 查看数组的维数
#
# c = x.view(3, 3, 5, 2, 2)
# # b = x.permute(0, 1, 2)  # 不改变维度
# print(c)
# print(c.size())
# d = x.view(3, 3, 5, 2, 2).permute(0, 1, 3, 4, 2)
#
#
# # c = x.permute(0,2,1)             # 每一块的行与列进行交换，即每一块做转置行为
# print(d)
# print(d.size())
# e = x.view(3, 3, 5, 2, 2).permute(0, 1, 3, 4, 2).contiguous()
# # d = x.permute(1, 0, 2)  # 交换块和行
# print(e)
# print(e.size())
# x = torch.randn(3, 2)
# y = x.view(6, 1).contiguous()
# print("修改前：")
# print("x-", x)
# print("y-", y)
#
# print("\n修改后：")
# y[0, 0] = 11
# print("x-", x)
# print("y-", y)
import torch
import numpy as np
import cv2 as cv
import PIL.Image as Image
from detection import YOLO


class gen_frame():
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.count = 0  # count the number of pictures
        self.frame_interval = 1  # video frame count interval frequency
        self.frame_interval_count = 0
        self.save_path = 'E:/fangang/data/'
        c = 0
        for i in os.listdir(self.save_path):  # HK_01
            if int(i.split('_')[-1]) >= c:
                c = int(i.split('_')[-1]) + 1
        self.save_path = os.path.join(self.save_path, 'HK_{}'.format(c))
        os.mkdir(self.save_path)

    def save_image(self, num, image):
        image_path = os.path.join(self.save_path, '{}.PNG'.format(str(num)))
        cv2.imwrite(image_path, image)

    def gen_frame_30(self, frame_data):
        # store operation every time f frame
        if self.frame_interval_count % self.frame_interval == 0:
            self.save_image(self.count, frame_data)
            logging.info("num：" + str(self.count) + ", frame: " +
                         str(self.frame_interval_count))
            self.count += 1
        self.frame_interval_count += 1


# yolo = YOLO()
cap = cv.VideoCapture(0)
fps = 0
gen = gen_frame()
while (True):
    # 一帧一帧捕捉
    ret, frame = cap.read()
    # 我们对帧的操作在这里
    # img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # img = Image.fromarray(img)
    # ti = time.time()
    # r_image = yolo.detect_image(img, crop=False)
    # fps = np.round((fps + (1. / (time.time() - ti))) / 2, 0)
    # img = cv.cvtColor(np.array(r_image), cv.COLOR_RGB2BGR)
    # image = cv.putText(img, f'FPS: {fps}', (5, 50), cv.FONT_HERSHEY_SIMPLEX,
    #                    0.75, (0, 0, 255), 2)
    gen.gen_frame_30(frame)
    # 显示返回的每帧
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    # r_image.show()
# 当所有事完成，释放 VideoCapture 对象
cap.release()
cv.destroyAllWindows()
