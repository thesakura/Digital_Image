from PIL import Image
import numpy as np

img=Image.open('1.jpg').convert('L')
img1=img.copy()
img2=img.copy()
img3=img.copy()
img4=img.copy()
row=img.size[0]
cols=img.size[1]
a1=list([[0,-1,0],[-1,4,-1],[0,-1,0]])
a2=list([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
a3=list([[1,-2,1],[-2,4,-2],[1,-2,1]])
a4=list([[0,-1,0],[-1,5,-1],[0,-1,0]])
for i in range(1,row-1):
    for j in range(1,cols-1):
        tmp1=128+img.getpixel((i,j))*a1[1][1]+img.getpixel((i-1,j))*a1[0][1]+img.getpixel((i+1,j))*a1[2][1]+img.getpixel((i,j-1))*a1[1][0]+img.getpixel((i,j+1))*a1[1][2]
        tmp2 = 128 + a2[1][1] * img.getpixel((i, j)) +a2[0][0]* img.getpixel((i - 1, j - 1)) +a2[0][1]* img.getpixel(
            (i - 1, j)) +a2[0][2]* img.getpixel((i - 1, j + 1)) +a2[1][0]* img.getpixel((i, j - 1)) +a2[1][2]* img.getpixel(
            (i, j + 1)) +a2[2][0]* img.getpixel((i + 1, j - 1)) +a2[2][1]* img.getpixel((i + 1, j)) +a2[2][2]* img.getpixel((i + 1, j + 1))
        tmp3 = 128 + a3[1][1] * img.getpixel((i, j)) + a3[0][0] * img.getpixel((i - 1, j - 1)) + a3[0][1] * img.getpixel(
            (i - 1, j)) + a3[0][2] * img.getpixel((i - 1, j + 1)) + a3[1][0] * img.getpixel((i, j - 1)) + a3[1][2] * img.getpixel(
            (i, j + 1)) + a3[2][0] * img.getpixel((i + 1, j - 1)) + a3[2][1] * img.getpixel((i + 1, j)) + a3[2][2] * img.getpixel((i + 1, j + 1))
        tmp4 = img.getpixel((i,j)) * a4[1][1] +img.getpixel((i-1,j))* a4[0][1] + img.getpixel((i+1,j)) * a4[2][1] + img.getpixel((i,j-1)) *  a4[1][0] + img.getpixel((i,j+1)) * a4[1][2]
        img1.putpixel((i,j),tmp1)
        img2.putpixel((i,j),tmp2)
        img3.putpixel((i,j),tmp3)
        img4.putpixel((i,j),tmp4)
img1.save('image1.bmp')
img2.save('image2.bmp')
img3.save('image3.bmp')
img4.save('image4.bmp')