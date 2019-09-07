# coding=utf-8

import cv2 as cv

# 导入函数文件
from function import Deal
from function import Algo
from function.Algo import Point
from function.Algo import LevelSet

"""=================图像预操作==================="""
# 计算处理时间
start = cv.getTickCount()

'''---------------输入输出原图-----------------'''
src = cv.imread("./resources/amap2.jpg")
# src = cv.imread("./resources/test_map.jpg")
# src = cv.imread("./resources/test2.jpg")

src_ls = src.copy()  # 后面水平集要用
src_final = src.copy()  # 最终结果载体

cv.imshow("src", src)
print("原图大小：", src.shape)
# '''----------提取各个色彩空间-------------'''
B, G, R = Deal.split(src)  # RGB顺序相反
# 转为hsv
hsv = cv.cvtColor(src, cv.COLOR_RGB2HSV)
H, S, V = Deal.split(hsv)
# cv.imshow("V", V)
# cv.imshow("S", S)

# 转换到要处理的图像
img_v = V
img_s = S
# cv.imwrite("../NotPush/test_gray.jpg", V)

"""=================初始曲线提取===================="""

'''--------使用给定的单值图像表示并计算强度直方图----------'''
# hist = cv.calcHist([img], [0], None, [256], [0, 256])
#
# plt.figure()  # 新建一个图像
# plt.title("Histogram of img")  # 标题
# plt.xlabel("值")  # x轴名称与坐标范围
# plt.xlim([0, 256])
# plt.ylabel("数量")
#
# plt.plot(hist)  # 绘图
# plt.show()

'''------------------------设定参数------------------------'''
# 迭代参数
# 测试发现，10以后的结果都差不多，因此设定为10即可
IterationNumber = 10  # 经验值
# 聚类数
# 原来代码中写的是2个聚类
ClusterNumber = 2  # 分为前景和背景两个聚类

'''--------------使用k_means++算法求k个质心-------------'''
# 初始化质心，根据聚类数确定个数
cluster_centers_v = [Point() for _ in range(ClusterNumber)]
# 将0值的聚类中心代入函数寻找k_means++的初始质心
cluster_centers_v = Algo.k_means_plus(img_v, cluster_centers_v)

# # 初始化质心，根据聚类数确定个数
# cluster_centers_s = [Point() for _ in range(ClusterNumber)]
# # 将0值的聚类中心代入函数寻找k_means++的初始质心
# cluster_centers_s = Algo.k_means_plus(img_s, cluster_centers_s)

'''-------------------重复聚类步骤直到稳定-----------------'''
# 基于距离聚类各点

# V聚类
# 在每一个聚类中计算新的质心点
k_means_img_v_temp, binary_inv_v = Algo.k_means(img_v, ClusterNumber, cluster_centers_v, IterationNumber)
# 将像素值变为二值并且深色为黑
k_means_v_img = Deal.to_binary(k_means_img_v_temp, binary_inv_v)

# # S聚类
# # 在每一个聚类中计算新的质心点
# k_means_img_s_temp, binary_inv_s = Algo.k_means(img_s, ClusterNumber, cluster_centers_s, IterationNumber, 1)  # s值需要反转
# # 将像素值变为二值并且深色为黑
# k_means_s_img = Deal.to_binary(k_means_img_s_temp, binary_inv_s)

# cv.imshow("k_means_v", k_means_v_img)
# cv.imshow("k_means_s", k_means_s_img)

# # test
# img = cv.bitwise_and(k_means_s_img, k_means_v_img)
# cv.imshow("test", img)


'''--------使用矩形结构元素对前景图像进行形态学操作--------'''
# 计算连通部分或者对象的数量，并保存其长宽为Xi,Yi
# 使用检测物体的长宽平均值来计算结构元素SE的长宽值

# 原先的情况如下：
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

# 使用SE进行腐蚀和膨胀操作

erode = cv.morphologyEx(k_means_v_img, cv.MORPH_ERODE, kernel)  # 腐蚀
dilate = cv.morphologyEx(erode, cv.MORPH_DILATE, kernel)  # 膨胀

# cv.imshow("dilate", dilate)
# final = dilate.copy()  # 用final存储最终结果

'''-----------------应用边界检测提取初始曲线-----------------'''
# 提取边界并存储在contours中，为一个list
cloneImage, contours, layout = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# 绘制边界
for c_i, contour in enumerate(contours):
    cv.drawContours(src, contours, c_i, (0, 255, 0), 1)
cv.imshow("inital_contours", src)

"""=====================主动轮廓模型====================="""
'''--------引入初始轮廓并通过等式初始化水平集设置函数--------'''
# 存储最后的轮廓
contours_final = []
# 每一个轮廓进行主动轮廓模型
for ls_i in range(len(contours)):
    # 将轮廓曲线提取为ROI区域
    roi_temp, contour_temp, offset_x, offset_y = Deal.contours_to_roi(src_ls, contours[ls_i])

    '''----------------------初始化水平集----------------------'''
    # 初始化
    ls = LevelSet(roi_temp)
    ls.initialize(1, roi_temp, contour_temp)  # 迭代次数

    '''-----------------------水平集演变-----------------------'''
    # 水平集演化
    ls_final = ls.evolution()

    '''---------------使用结构元素进行形态学操作---------------'''
    # 形态学操作
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

"""==================最终结果=========================="""
# 绘制最终轮廓
print("一共有轮廓%d个" % len(contours_final))
for c_i, contour_final in enumerate(contours_final):
    cv.drawContours(src_final, contours_final, c_i, (0, 255, 0), 1)
cv.imshow("final", src_final)

"""================等待========================"""
# 获取结束时间并计算总时间
end = cv.getTickCount()
total_time = (end - start)/cv.getTickFrequency()
print("所用时间：%s 秒" % total_time)

cv.waitKey(0)
cv.destroyAllWindows()
