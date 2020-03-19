# Building-Extraction
For Innovative undertaking works.  
extraction the building  
# First Edition:
简单的功能实现：能够完成特定类型建筑物的提取 
v1.0版本：git tag 1.0可导出
v2.0版本：git tag 2.0
          2.0主要更新可手动绘制图像区域功能
# Requirements：
  python 3.7.1  
  opencv-python 3.4.4.19  
  其余环境使用Anaconda3自带的或者使用pip安装即可  
# How to run：
  python main.py  
  具体的图片选择直接在main中更改  
# current results：
1.最佳显示结果：
![image](https://github.com/Gao-zl/Building-Extraction/blob/master/result/result-amap2.png)  
2.协和学院的建筑物提取效果：
![image](https://github.com/Gao-zl/Building-Extraction/blob/master/result/result-xh.jpg)  
3.1某中学的效果：(聚类为2的情况)
![image](https://github.com/Gao-zl/Building-Extraction/blob/master/result/result-mid-school-2.jpg)  
3.2某中学的效果：(聚类为3的情况)
![image](https://github.com/Gao-zl/Building-Extraction/blob/master/result/result-mid-school-ClusterNumber-2.jpg)  
4.糟糕的运行结果：知明笃行那一片区域：
![image](https://github.com/Gao-zl/Building-Extraction/blob/master/result/result-bad-situation.jpg)  
# 后续打算补充项目内容：  
1.添加可以直接输入图片名字来得到最后的实验结果  
2.图片输出的时候会直接按原图大小来显示，因此更改后实现图片按规定大小输出  
3.图片的分辨率可以直接在预处理的时候进行改进，可以添加前一段时间的CNN算法实现的图片转高清的文件  
4.某些特殊的图片处理的不太到位如上图4中的结果，感觉可能是图片的颜色问题，后续看看能不能改进  
5.添加分析代码  
# license：
  #https://github.com/Gao-zl/Building-Extraction  
  大创项目建筑物提取
