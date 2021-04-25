import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/15103/Desktop/NoisedImages/WuhanFighting.jpg',1)#读图
b_Noised, g_Noised, r_Noised = cv2.split(img);#分离出图片的B，R，G颜色通道


def Fourier(img):
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    #设置掩模为1的窗口大小x
    x=66
    mask[int(crow)-x:int(crow)+x, int(ccol)-x:int(ccol)+x] = 1
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    dst=np.zeros(img_back.shape,dtype=np.float32)
    cv2.normalize(img_back,dst=dst,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    dst=np.uint8(dst*255)
    return dst

b_result=Fourier(b_Noised)
g_result=Fourier(g_Noised)
r_result=Fourier(r_Noised)

plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(b_result, cmap = 'gray')
plt.title('b_result'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(g_result, cmap = 'gray')
plt.title('g_result'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(r_result, cmap = 'gray')
plt.title('r_result'), plt.xticks([]), plt.yticks([])
plt.show()

img2 = cv2.merge([b_result, g_result, r_result])
cv2.imwrite('C:/Users/15103/Desktop/WuhanFighting_de.jpg',img2)

import math

def PSNR(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

img_original = cv2.imread('C:/Users/15103/Desktop/OriginalImages/WuhanFighting.jpg',1)
print(PSNR(img2, img_original))

import math

def PSNR(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

img_original = cv2.imread('C:/Users/15103/Desktop/OriginalImages/Lena.jpg',1)
print(PSNR(img2, img_original))