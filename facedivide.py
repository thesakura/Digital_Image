import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
img_gray=Image.open("face1.jpg").convert('L')
img_color=Image.open("face1.jpg")


img=np.array(img_gray)
row,col=img.shape
img1=np.array(img_color)

for i in range(row):
    for j in range(col):
        if img1[i][j][0]<=150 and img1[i][j][1]<=150 and img1[i][j][2]>=70 and img1[i][j][0]<img1[i][j][2]:
            img1[i][j]=[240,240,240]

plt.imshow(img1,cmap='gray')
plt.axis('off')
plt.show()
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)




max=0
min=255
for i in range(row):
    for j in range(col):
        if img1[i][j]>max:
            max=img1[i][j]
        if img1[i][j]<min:
            min=img1[i][j]
T=[]
T.append(0)
min=int(min)
max=int(max)
T.append((min+max)/2)
print((max+min)//2,max,min)
while T[len(T)-1]!=T[len(T)-2]:
    count = 0
    count1 = 0
    a = 0
    b = 0
    for i in range(row):
        for j in range(col):
            if img1[i][j]>T[len(T)-1]:
                a=a+img1[i][j]
                count=count+1
            else:
                b=b+img1[i][j]
                count1=count1+1
            a=int(a)
            b=int(b)
    T.append((a//count+b//count1)//2)
t=T[len(T)-1]
print(T[len(T)-1])

for i in range(row):
    for j in range(col):
        if img1[i][j]>t:
            img1[i][j]=0
        '''else:
            img1[i][j]=255'''
kernel = np.ones((5,5),np.uint8)
img1 = cv2.dilate(img1,kernel,iterations = 1)

img1=cv2.bitwise_and(img,img1)

plt.imshow(img1,cmap='gray')
plt.axis('off')
plt.show()
