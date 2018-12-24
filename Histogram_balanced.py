from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
tmp=Image.open('1.bmp').convert('L')
tmp.save('gray.bmp')
img=np.array(tmp)  #打开图像并转化为数字矩阵
plt.figure('pokemon')
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()
img1=np.array(Image.open('gray.bmp'))
row,cols=img1.shape
count=0;
for i in range(row):
    for j in range(cols):
        if(img1[i][j]>200):
            img1[i][j]=0
            count=count+1
        else:
            img1[i][j] = 255
print(count/(cols*row))
plt.imshow(img1,cmap='gray')
plt.axis('off')
plt.show()
