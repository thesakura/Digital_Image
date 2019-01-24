from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform,data
from numpy import *
import  cv2
from scipy import misc
def kuang(res_end,img):#轮廓定位函数
    hh,contours, hierarchy = cv2.findContours(res_end,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#找出所有的轮廓
    for i in range(len(contours)):##对所有的轮廓进行筛选
        cnt = contours[i]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        area=cv2.contourArea(cnt)
        leng=(((box[0][1]-box[1][1])**2)+((box[0][0]-box[1][0])**2))**0.5#求长
        high=(((box[0][1]-box[3][1])**2)+((box[0][0]-box[3][0])**2))**0.5#求宽
        if area<2500:
            continue

        if (leng/high<5 and leng/high>1.5) or (leng/high>0.2 and leng/high<0.66):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 10)


    pic = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
    '''cv2.imshow('resize_pic',pic)
    cv2.waitKey(0)'''
    return pic#返回最后框选的结果
if __name__=="__main__":
    str0 = "License_plate/canon/IMG_0"  # IMG_0175.jpg
    str1 = "License_plate/hp_recorder/IMG_0"  # IMG_001.jpg
    str2 = "License_plate/iPhone_5s/IMG_0"  # IMG_0119.jpg
    for tt in range(175, 200):#canon数据集操作
        image = str0 + str(tt) + ".jpg"
        img = cv2.imread(image)#读取图像
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#转化为HSV空间
        H, S, V = cv2.split(HSV)
        LowerBlue = np.array([100, 100, 50])
        UpperBlue = np.array([130, 255, 255])

        mask = cv2.inRange(HSV, LowerBlue, UpperBlue)#进行蓝色的区分
        BlueThings = cv2.bitwise_and(img, img, mask=mask)
        LowerYellow = np.array([15, 150, 150])
        UpperYellow = np.array([30, 255, 255])
        mask = cv2.inRange(HSV, LowerYellow, UpperYellow)#进行黄色的区分
        YellowThings = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.bitwise_or(BlueThings, YellowThings)  # 黄色内容和蓝色内容取交集的结果
        # 转化为灰度图
        res_gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        retval, res_gray = cv2.threshold(res_gray, 20, 255, cv2.THRESH_BINARY)  # 二值化方法

        # 进行边缘检测
        tmp = str0 + str(tt) + ".jpg"
        img_color = cv2.imread(tmp)

        row, col, x = img_color.shape
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)#转化为灰度图
        # img_gray=grey_scale(img_gray)对比度拉伸

        img_gray = cv2.medianBlur(img_gray, 5)  # 中值滤波，去除椒盐噪声

        # gaussianResult = cv2.GaussianBlur(img_gray, (5, 5), 1.5)  # 高斯模糊
        # img_gray = cv2.equalizeHist(img_gray)  # 直方图均衡化

        x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=3)  # sobel算子
        y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1, ksize=3)
        #laplacian = cv2.Laplacian(img_gray, cv2.CV_16S)
        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)
        #laplacian = cv2.convertScaleAbs(laplacian)
        img_gray = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        '''plt.imshow(res_end, cmap='gray')
        plt.axis('off')
        plt.show()'''

        retval, img_gray = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)  # 二值化方法


        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        img_gray = cv2.dilate(img_gray, kernel, iterations=1)# 膨胀

        img_gray = cv2.erode(img_gray, kernel)#腐蚀
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)#开运算
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)

        # 边缘检测结果和颜色的结果取交集
        res_end = cv2.bitwise_and(img_gray, res_gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        res_end = cv2.dilate(res_end, kernel, iterations=1)#膨胀
        res_end=kuang(res_end,img)
        dire = "res/canon/" + str(tt) + ".jpg"
        cv2.imwrite(dire, res_end)



    for tt in range(1, 30):#hp_recorder数据集操作
        str1_tmp=str1
        if tt>=1 and tt<=9:
            str1_tmp=str1_tmp+"0"
        image = str1_tmp + str(tt) + ".jpg"
        img = cv2.imread(image)#读取图像
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#转化为HSV空间
        H, S, V = cv2.split(HSV)
        LowerBlue = np.array([100, 100, 50])
        UpperBlue = np.array([130, 255, 255])

        mask = cv2.inRange(HSV, LowerBlue, UpperBlue)#选择出蓝色区域
        BlueThings = cv2.bitwise_and(img, img, mask=mask)
        LowerYellow = np.array([15, 150, 150])
        UpperYellow = np.array([30, 255, 255])
        mask = cv2.inRange(HSV, LowerYellow, UpperYellow)#选择出黄色区域
        YellowThings = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.bitwise_or(BlueThings, YellowThings)  # 黄色内容和蓝色内容取交集的结果
        # 转化为灰度图
        res_gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        retval, res_gray = cv2.threshold(res_gray, 20, 255, cv2.THRESH_BINARY)  # 二值化方法

        # 进行边缘检测
        tmp = str1_tmp + str(tt) + ".jpg"
        img_color = cv2.imread(tmp)

        row, col, x = img_color.shape
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)#转化为灰度图
        # img_gray=grey_scale(img_gray)对比度拉伸

        img_gray = cv2.medianBlur(img_gray, 5)  # 中值滤波，去除椒盐噪声

        # gaussianResult = cv2.GaussianBlur(img_gray, (5, 5), 1.5)  # 高斯模糊
        # img_gray = cv2.equalizeHist(img_gray)  # 直方图均衡化

        x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=3)  # sobel算子
        y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1, ksize=3)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_16S)
        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)
        laplacian = cv2.convertScaleAbs(laplacian)
        img_gray = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


        retval, img_gray = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)  # 二值化方法


        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 膨胀
        img_gray = cv2.dilate(img_gray, kernel, iterations=1)

        img_gray = cv2.erode(img_gray, kernel)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)#开运算
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)

        # 边缘检测结果和颜色的结果取交集
        res_end = cv2.bitwise_and(img_gray, res_gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        res_end = cv2.dilate(res_end, kernel, iterations=1)#膨胀
        res_end = kuang(res_end, img)
        dire = "res/hp_recorder/" + str(tt) + ".jpg"
        cv2.imwrite(dire, res_end)

    for tt in range(119, 155):#iPhone_5s数据集操作
        image = str2 + str(tt) + ".jpg"
        img = cv2.imread(image)#读取图像
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#转化为HSV空间
        H, S, V = cv2.split(HSV)
        LowerBlue = np.array([100, 100, 50])
        UpperBlue = np.array([130, 255, 255])

        mask = cv2.inRange(HSV, LowerBlue, UpperBlue)#选择出蓝色区域
        BlueThings = cv2.bitwise_and(img, img, mask=mask)
        LowerYellow = np.array([15, 150, 150])
        UpperYellow = np.array([30, 255, 255])
        mask = cv2.inRange(HSV, LowerYellow, UpperYellow)#选择出黄色区域
        YellowThings = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.bitwise_or(BlueThings, YellowThings)  # 黄色内容和蓝色内容取交集的结果
        # 转化为灰度图
        res_gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        retval, res_gray = cv2.threshold(res_gray, 20, 255, cv2.THRESH_BINARY)  # 二值化方法

        # 进行边缘检测
        tmp = str2 + str(tt) + ".jpg"
        img_color = cv2.imread(tmp)

        row, col, x = img_color.shape
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        # img_gray=grey_scale(img_gray)对比度拉伸

        img_gray = cv2.medianBlur(img_gray, 5)  # 中值滤波，去除椒盐噪声

        # gaussianResult = cv2.GaussianBlur(img_gray, (5, 5), 1.5)  # 高斯模糊
        # img_gray = cv2.equalizeHist(img_gray)  # 直方图均衡化

        x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=3)  # sobel算子
        y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1, ksize=3)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_16S)
        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)
        laplacian = cv2.convertScaleAbs(laplacian)
        img_gray = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        retval, img_gray = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)  # 二值化方法

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 膨胀
        img_gray = cv2.dilate(img_gray, kernel, iterations=1)

        img_gray = cv2.erode(img_gray, kernel)#腐蚀
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)#开运算
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)

        # 边缘检测结果和颜色的结果取交集
        res_end = cv2.bitwise_and(img_gray, res_gray)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        res_end = cv2.morphologyEx(res_end, cv2.MORPH_CLOSE, kernel)#开运算
        res_end = cv2.dilate(res_end, kernel, iterations=1)#膨胀
        res_end = cv2.morphologyEx(res_end, cv2.MORPH_CLOSE, kernel)
        res_end = cv2.dilate(res_end, kernel, iterations=1)
        res_end = cv2.dilate(res_end, kernel, iterations=1)
        res_end = cv2.morphologyEx(res_end, cv2.MORPH_CLOSE, kernel)

        res_end = kuang(res_end, img)
        dire = "res/iPhone_5s/"+ str(tt) + ".jpg"
        cv2.imwrite(dire, res_end)