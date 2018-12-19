from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import *

tmp=Image.open('digitalimage.png')
img=np.array(tmp)
img1=img
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()
a=zeros(256);
row,cols=img.shape
for i in range(row):
    for j in range(cols):
        a[img[i][j]]=a[img[i][j]]+1
b=zeros(256)
for i in range(256):
    b[i]=a[i]/(row*cols*1.0)
for i in range(256):
    if(i!=0):
        b[i]=b[i]+b[i-1];
b=np.round(255 * b)
plt.imshow(img1,cmap='gray')
plt.axis('off')
plt.show()
for i in range(row):
    for j in range(cols):
        img1[i][j]=b[img1[i][j]]
plt.imshow(img1,cmap='gray')
plt.axis('off')
plt.show()