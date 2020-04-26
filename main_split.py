# coding=utf-8
import PIL.Image
import os
import cv2 as cv
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from function import Deal
from function import Algo
from function.Algo import Point
from function.Algo import LevelSet

class UseCv:
    def __init__(self):
        self.path = './resource/test6.png'

    def cut(self):
        img = cv.imread(self.path, flags=cv.IMREAD_COLOR)
        bbox = cv.selectROI(img, False)
        cut = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        cv.imwrite('cut.jpg', cut)

def deal_pic():
    start = cv.getTickCount()

    src = cv.imread("cut.jpg")
    # cv.imshow("src", src)
    cv.imwrite('./result/original.png', src)
    print("调整后的图片大小： ", src.shape)

    src_ls = src.copy()
    src_final = src.copy()

    B, G, R = Deal.split(src)
    hsv = cv.cvtColor(src, cv.COLOR_RGB2HSV)
    H, S, V = Deal.split(hsv)
    img_v = V
    img_s = S

    hist = cv.calcHist([src], [0], None, [256], [0,256])
    plt.figure()
    plt.title("Histogram of img")
    plt.xlabel("values")
    plt.xlim([0, 256])
    plt.ylabel("numbers")
    plt.plot(hist)
    plt.savefig("./result/histogram.png")

    IterationNumber = 10
    ClusterNumber = 2

    cluster_centers_v = [Point() for _ in range(ClusterNumber)]
    cluster_centers_v = Algo.k_means_plus(img_v, cluster_centers_v)

    cluster_centers_s = [Point() for _ in range(ClusterNumber)]
    cluster_centers_s = Algo.k_means_plus(img_s, cluster_centers_s)

    k_means_img_v_temp, binary_inv_v = Algo.k_means(img_v, ClusterNumber, cluster_centers_v, IterationNumber)
    k_means_v_img = Deal.to_binary(k_means_img_v_temp, binary_inv_v)
    k_means_img_s_temp, binary_inv_s = Algo.k_means(img_s, ClusterNumber, cluster_centers_s, IterationNumber, 1)  # s值需要反转
    k_means_s_img = Deal.to_binary(k_means_img_s_temp, binary_inv_s)
    img = cv.bitwise_and(k_means_img_v_temp, k_means_img_s_temp)
    cv.imwrite("./result/cluster.png", img)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # 使用SE进行腐蚀和膨胀
    erode = cv.morphologyEx(k_means_v_img, cv.MORPH_ERODE, kernel)
    dilate = cv.morphologyEx(erode, cv.MORPH_DILATE, kernel)
    cv.imwrite("./result/dilate.png", dilate)
    cloneImage, contours, layout = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for c_i, contour in enumerate(contours):
        cv.drawContours(src, contours, c_i, (0, 255, 0), 1)
    cv.imwrite("./result/inital_contours.png", src)

    contours_final = []
    for ls_i in range(len(contours)):
        roi_temp, contour_temp, offset_x, offset_y = Deal.contours_to_roi(src_ls, contours[ls_i])
        ls = LevelSet(roi_temp)
        ls.initialize(1, roi_temp, contour_temp)
        ls_final = ls.evolution()
        erode2 = cv.morphologyEx(ls_final, cv.MORPH_ERODE, kernel)  # 腐蚀
        dilate2 = cv.morphologyEx(erode2, cv.MORPH_DILATE, kernel)  # 膨胀
        ret3, ls_one_2value = cv.threshold(dilate2, 0, 255, cv.THRESH_BINARY)
        ls_one_2value = cv.convertScaleAbs(ls_one_2value, cv.CV_8UC1)  # 转为CV_8UC1格式
        cloneImage2, contour_one, layout2 = cv.findContours(ls_one_2value, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        if len(contour_one) > 0:
            contour_convert = contour_one[0]
            for i in range(len(contour_convert)):
                contour_convert[i][0][0] = contour_convert[i][0][0] + offset_x
                contour_convert[i][0][1] = contour_convert[i][0][1] + offset_y
            contours_final.append(contour_convert)
        else:
            continue

    # 最后结果
    print("一共有" , len(contours_final) , "个轮廓")
    for c_i, contour_final in enumerate(contours_final):
        cv.drawContours(src_final, contours_final, c_i, (0, 255, 0), 1)
    cv.imwrite("./result/final.png", src_final)
    end = cv.getTickCount()
    total_time = (end - start)/cv.getTickFrequency()
    print("所用时间：%s 秒" % total_time)

    img1 = cv.imread("./result/original.png")
    img2 = cv.imread("./result/cluster.png")
    img3 = cv.imread("./result/dilate.png")
    img4 = cv.imread("./result/inital_contours.png")
    img5 = cv.imread("./result/final.png")
    imgs = np.hstack([img1, img2, img3, img4, img5])
    cv.imshow("original-cluster-dilate-inital_contours-final", imgs)
    plt.show("./result/histogram.png")

def imgreshape():
    infile = 'cut.jpg'
    outfile = 'cut.jpg'
    im = PIL.Image.open(infile)
    (x,y) = im.size
    print("原始图像大小：",im.size)
    x_s = 160
    y_s = int(y * x_s / x)
    out = im.resize((x_s,y_s),PIL.Image.ANTIALIAS)
    out.save(outfile)

if __name__ == '__main__':
    UseCv().cut()
    imgreshape()
    deal_pic()
