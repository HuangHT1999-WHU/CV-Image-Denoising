import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2

img = cv2.imread('C:/Users/15103/Desktop/NoisedImages/Lena.jpg',1)

cv2.imshow("img", img)
plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

b_Noised, g_Noised, r_Noised = cv2.split(img);#分离出图片的B，R，G颜色通道
zeros = np.zeros(img.shape[:2],dtype="uint8");#创建与image相同大小的零矩阵
cv2.imshow("BLUE",cv2.merge([b_Noised,zeros,zeros]));#显示 （B，0，0）图像
cv2.imshow("GREEN",cv2.merge([zeros,g_Noised,zeros]));#显示（0，G，0）图像
cv2.imshow("RED",cv2.merge([zeros,zeros,r_Noised]));#显示（0，0，R）图像
k = cv2.waitKey()  #等待回应
if k == 27: #等待Esc退出
    cv2.destroyAllWindows()  #摧毁窗口

pca=PCA(0.865)

pca.fit(r_Noised)
r_low=pca.transform(r_Noised)
r_result=pca.inverse_transform(r_low).astype(np.int)
plt.subplot(222),plt.imshow(r_result,cmap = 'gray')
plt.title('r_result'), plt.xticks([]), plt.yticks([])

pca.fit(g_Noised)
g_low=pca.transform(g_Noised)
g_result=pca.inverse_transform(g_low).astype(np.int)
plt.subplot(223),plt.imshow(g_result,cmap = 'gray')
plt.title('g_result'), plt.xticks([]), plt.yticks([])

pca.fit(b_Noised)
b_low=pca.fit_transform(b_Noised)
b_result=pca.inverse_transform(b_low).astype(np.int)
plt.subplot(224),plt.imshow(b_result,cmap = 'gray')
plt.title('b_result'), plt.xticks([]), plt.yticks([])
plt.show()

img2 = cv2.merge([b_result, g_result, r_result])

"""
cv2.imshow('Merge', img2)
k = cv2.waitKey()  #等待回应
if k == 27: #等待Esc退出
    cv2.destroyAllWindows()  #摧毁窗口
"""

cv2.imwrite('C:/Users/15103/Desktop/Lena_de.jpg',img2)

import math

def PSNR(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

img_original = cv2.imread('C:/Users/15103/Desktop/OriginalImages/Lena.jpg',1)
print(PSNR(img2, img_original))