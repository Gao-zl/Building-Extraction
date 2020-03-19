# coding=utf-8
import os
import cv2 as cv
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from function import Deal
from function import Algo
from function.Algo import Point
from function.Algo import LevelSet

# 图像预处理---------------------------------
start = cv.getTickCount()

# 读取并显示图片
# 以下为修改内容之一：版本v1
# src = cv.imread("./resource/amap2.jpg")
# cv.imshow("src", src)
# print("原图大小： ", src.shape)

# 读取不显示图片
# 修改：不显示图片，直接保存最后再一次性显示，减少等待时间简化操作步骤
src = cv.imread("cut.jpg")
# cv.imshow("src", src)
cv.imwrite('./result/original.png', src)
print("原图大小： ", src.shape)

# 后续用到的备份：水平集和最终的图片载体
src_ls = src.copy()
src_final = src.copy()

# 提取色彩空间
B, G, R = Deal.split(src)
hsv = cv.cvtColor(src, cv.COLOR_RGB2HSV)
H, S, V = Deal.split(hsv)
img_v = V
img_s = S

# 初始图线提取------------------------------
# 计算直方图，后续再修改为不显示吧，这个图片的大小通道都不一样
hist = cv.calcHist([src], [0], None, [256], [0,256])
plt.figure()
plt.title("Histogram of img")
plt.xlabel("values")
plt.xlim([0, 256])
plt.ylabel("numbers")
plt.plot(hist)
plt.savefig("./result/histogram.png")

# 设定后续计算的参数
# 迭代参数
IterationNumber = 10
# 聚类数：如果前景背景的话就是两个聚类，可以多
ClusterNumber = 2

# k_mean_plus 计算质心：在Algo里面有此函数
# 先初始化质心，再把0带入寻找k_means的初始质心
cluster_centers_v = [Point() for _ in range(ClusterNumber)]
cluster_centers_v = Algo.k_means_plus(img_v, cluster_centers_v)

cluster_centers_s = [Point() for _ in range(ClusterNumber)]
cluster_centers_s = Algo.k_means_plus(img_s, cluster_centers_s)


# 重复聚类达到稳定
# V聚类
# 在每一个聚类中计算新的质心点
k_means_img_v_temp, binary_inv_v = Algo.k_means(img_v, ClusterNumber, cluster_centers_v, IterationNumber)
# 将像素值变为二值并且深色为黑
k_means_v_img = Deal.to_binary(k_means_img_v_temp, binary_inv_v)

# S聚类，与V同，但是需要反转
k_means_img_s_temp, binary_inv_s = Algo.k_means(img_s, ClusterNumber, cluster_centers_s, IterationNumber, 1)  # s值需要反转
k_means_s_img = Deal.to_binary(k_means_img_s_temp, binary_inv_s)

# # 显示聚类后的图片v1
# img = cv.bitwise_and(k_means_img_v_temp, k_means_img_s_temp)
# cv.imshow("test", img)
# cv.waitKey(0)

# 修改：到后面再显示聚类图片，以聚类图片命名
img = cv.bitwise_and(k_means_img_v_temp, k_means_img_s_temp)
cv.imwrite("./result/cluster.png", img)


# 使用矩形结构元素对前景图像进行形态学操作
# 计算连通的部分和对象数量，保存长宽为Xi, Yi
# 使用检测物体长宽平均值来计算结构元素SE的长宽值
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

# 使用SE进行腐蚀和膨胀
erode = cv.morphologyEx(k_means_v_img, cv.MORPH_ERODE, kernel)
dilate = cv.morphologyEx(erode, cv.MORPH_DILATE, kernel)

# 原版v1
# cv.imshow("dilate", dilate)
# cv.waitKey(0)

# 修改后续显示
cv.imwrite("./result/dilate.png", dilate)

# 应用边界检测提取初始线
# 提取边界并存在contours中，为另一个list
cloneImage, contours, layout = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# 此行代码有错误：cv.findCountours只能返回两个值：在opencv4以后，因此只能安装opencv4之前的版本，使用pip如下：
# pip install opencv-python==3.4.3.18
for c_i, contour in enumerate(contours):
    cv.drawContours(src, contours, c_i, (0, 255, 0), 1)
# cv.imshow("inital_contours", src)
# 修改先不显示
cv.imwrite("./result/inital_contours.png", src)

# 主动轮廓模型
contours_final = []
for ls_i in range(len(contours)):
    roi_temp, contour_temp, offset_x, offset_y = Deal.contours_to_roi(src_ls, contours[ls_i])

    # 初始化水平集
    ls = LevelSet(roi_temp)
    ls.initialize(1, roi_temp, contour_temp)

    # 水平集演变
    ls_final = ls.evolution()

    # 结构元素进行形态学操作
    erode2 = cv.morphologyEx(ls_final, cv.MORPH_ERODE, kernel)  # 腐蚀
    dilate2 = cv.morphologyEx(erode2, cv.MORPH_DILATE, kernel)  # 膨胀

    # 二值化
    ret3, ls_one_2value = cv.threshold(dilate2, 0, 255, cv.THRESH_BINARY)
    # 在每一个的二值化结果里再找轮廓并存储
    ls_one_2value = cv.convertScaleAbs(ls_one_2value, cv.CV_8UC1)  # 转为CV_8UC1格式
    cloneImage2, contour_one, layout2 = cv.findContours(ls_one_2value, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # 还原偏移值
    if len(contour_one) > 0:
        contour_convert = contour_one[0]
        for i in range(len(contour_convert)):
            # 挨个减去最左最上值
            contour_convert[i][0][0] = contour_convert[i][0][0] + offset_x
            contour_convert[i][0][1] = contour_convert[i][0][1] + offset_y
        contours_final.append(contour_convert)
    else:
        continue

# 最后结果
print("一共有" , len(contours_final) , "个轮廓")
for c_i, contour_final in enumerate(contours_final):
    cv.drawContours(src_final, contours_final, c_i, (0, 255, 0), 1)
# cv.imshow("final", src_final)
cv.imwrite("./result/final.png", src_final)
end = cv.getTickCount()

# 获取结束时间并计算总时间
total_time = (end - start)/cv.getTickFrequency()
print("所用时间：%s 秒" % total_time)

# 综合显示方法1：这种方法有一个缺点就是每一张图片都要一样的大小才行
# 而且不能显示图片的标题，因此想换成方法2来实现图片的显示，这次会包括plt的整合
# 2太麻烦，注释掉了，就用这个方法就好
# 修改的第二版本是改在标题上直接显示出这些图片的名字，反正够长
# 然后单独显示直方图。
# # 显示图片
img1 = cv.imread("./result/original.png")
img2 = cv.imread("./result/cluster.png")
img3 = cv.imread("./result/dilate.png")
img4 = cv.imread("./result/inital_contours.png")
img5 = cv.imread("./result/final.png")
imgs = np.hstack([img1, img2, img3, img4, img5])
cv.imshow("original-cluster-dilate-inital_contours-final", imgs)
plt.show("./result/histogram.png")

# 不需要，直接等plt.show()跳出来一起关掉就好
# cv.waitKey(0)

# # 显示方法2：
# def show_img2():
#     def im_show(path, img_num, title_num, pos):
#         gs = gridspec.GridSpec(5, 5)
#         ##### 为什么不行?
#         # 未解
#         # 直接用 "img" + str(i)这种命名方法在这里一直提示我出错了
#         # 不知道是不是编译器出错了
#         # 在colab上都可以的
#         if pos == 6:
#             plt.subplot(gs[1: , :])
#             plt.imshow(img_num)
#             # plt.title(title_num)
#             plt.xticks([])
#             plt.yticks([])
#         else:
#             plt.subplot(5, 5, pos)
#             plt.imshow(img_num)
#             plt.title(title_num)
#             plt.xticks([])
#             plt.yticks([])
#
#     img1 = plt.imread("./result/original.png")
#     img2 = cv.imread("./result/cluster.png")
#     img3 = cv.imread("./result/dilate.png")
#     img4 = plt.imread("./result/inital_contours.png")
#     img5 = plt.imread("./result/final.png")
#     img6 = plt.imread("./result/histogram.png")
#
#     im_show("./result/original.png", img1, "original.png", 1)
#     im_show("./result/cluster.png", img2, "cluster.png", 2)
#     im_show("./result/dilate.png", img3, "dilate.png", 3)
#     im_show("./result/inital_contours.png", img4, "inital_contours.png", 4)
#     im_show("./result/final.png", img5, "final.png", 5)
#     im_show("./result/histogram.png", img6, "histogram.png", 6)
#
#     plt.show()
#
# print("输入想看的图片模式\n"
#       "1.大图清晰无标题(建议方法):输入除0外任意字符选择此项\n"
#       "0.小图可操作：输入0以选择此项")
#
# choice = input()
# if choice != "0":
#     show_img1()
# else:
#     show_img2()

# 后续添加内容：对于大图像的分割以及数据分析