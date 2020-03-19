# -*- coding:utf-8 -*-

import cv2


class UseCv:
    def __init__(self):
        self.path = 'test1.png'

    def cut(self):
        img = cv2.imread(self.path, flags=cv2.IMREAD_COLOR)
        bbox = cv2.selectROI(img, False)
        cut = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        cv2.imwrite('cut.jpg', cut)


if __name__ == '__main__':
    UseCv().cut()