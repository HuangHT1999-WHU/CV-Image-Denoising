import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('C:/Users/15103/Desktop/NoisedImages/Lena.jpg', 1)  # 读图
b_Noised, g_Noised, r_Noised = cv2.split(img);  # 分离出图片的B，R，G颜色通道

# cv2.imshow("img", img)
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

"""遍历图像每个像素的每个通道"""
height = img.shape[0]  # 将tuple中的元素取出，赋值给height，width，channels
width = img.shape[1]

b_result = np.zeros(b_Noised.shape, np.uint8)
g_result = np.zeros(g_Noised.shape, dtype=np.uint8)
r_result = np.zeros(r_Noised.shape, dtype=np.uint8)

for row in range(1, height - 1):  # 遍历每一行
    for col in range(1, width - 1):  # 遍历每一列
        b_result[row][col] = (int(b_Noised[row - 1][col - 1]) + int(b_Noised[row - 1][col]) + int(
            b_Noised[row - 1][col + 1]) + int(b_Noised[row][col - 1]) + int(b_Noised[row][col]) + int(
            b_Noised[row][col + 1]) + int(b_Noised[row + 1][col - 1]) + int(b_Noised[row + 1][col]) + int(
            b_Noised[row + 1][col + 1])) / 9
        g_result[row][col] = (int(g_Noised[row - 1][col - 1]) + int(g_Noised[row - 1][col]) + int(
            g_Noised[row - 1][col + 1]) + int(g_Noised[row][col - 1]) + int(g_Noised[row][col]) + int(
            g_Noised[row][col + 1]) + int(g_Noised[row + 1][col - 1]) + int(g_Noised[row + 1][col]) + int(
            g_Noised[row + 1][col + 1])) / 9
        r_result[row][col] = (int(r_Noised[row - 1][col - 1]) + int(r_Noised[row - 1][col]) + int(
            r_Noised[row - 1][col + 1]) + int(r_Noised[row][col - 1]) + int(r_Noised[row][col]) + int(
            r_Noised[row][col + 1]) + int(r_Noised[row + 1][col - 1]) + int(r_Noised[row + 1][col]) + int(
            r_Noised[row + 1][col + 1])) / 9
        # 通过数组索引访问该元素，并作出处理

# cv2.imshow("processed img",img) #将处理后的图像显示出来

plt.subplot(222), plt.imshow(b_result, cmap='gray')
plt.title('b_result'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(g_result, cmap='gray')
plt.title('g_result'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(r_result, cmap='gray')
plt.title('r_result'), plt.xticks([]), plt.yticks([])

img2 = cv2.merge([b_result, g_result, r_result])
cv2.imwrite('C:/Users/15103/Desktop/Lena_de.jpg', img2)

import math


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


img_original = cv2.imread('C:/Users/15103/Desktop/OriginalImages/Lena.jpg', 1)
print(PSNR(img2, img_original))

# 均值滤波
# cv2.blur(image, (ksize, ksize))
blur33 = cv2.blur(img, (3, 3))  # 模板大小3*3
blur35 = cv2.blur(img, (3, 5))  # 模板大小3*5
blur55 = cv2.blur(img, (5, 5))  # 模板大小5*5
blur77 = cv2.blur(img, (7, 7))  # 模板大小7*7

print("经过模板大小为3*3的均值滤波降噪处理后的PSNR值：")
print(PSNR(blur33, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/blur33_Lena_de.jpg',blur33)

print("经过模板大小为3*5的均值滤波降噪处理后的PSNR值：")
print(PSNR(blur35, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/blur35_Lena_de.jpg',blur35)

print("经过模板大小为5*5的均值滤波降噪处理后的PSNR值：")
print(PSNR(blur55, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/blur55_Lena_de.jpg',blur55)

print("经过模板大小为7*7的均值滤波降噪处理后的PSNR值：")
print(PSNR(blur77, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/blur77_Lena_de.jpg',blur77)
print("\n")

# 高斯滤波
""""
cv2.GaussianBlur(image, (size, size), sigmaX, sigmaY) 
sigmaX和sigmaY代表了横轴与纵轴权重的标准差，
若为0就是让opencv帮你自动推算方差大小。
"""""
GaussianBlur33 = cv2.GaussianBlur(img, (3, 3), 0, 0)
GaussianBlur55 = cv2.GaussianBlur(img, (5, 5), 0, 0)
GaussianBlur77 = cv2.GaussianBlur(img, (7, 7), 0, 0)

print("经过模板大小为3*3的高斯滤波降噪处理后的PSNR值：")
print(PSNR(GaussianBlur33, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/GaussianBlur33_Lena_de.jpg',GaussianBlur33)

print("经过模板大小为5*5的高斯滤波降噪处理后的PSNR值：")
print(PSNR(GaussianBlur55, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/GaussianBlur55_Lena_de.jpg',GaussianBlur55)

print("经过模板大小为7*7的高斯滤波降噪处理后的PSNR值：")
print(PSNR(GaussianBlur77, img_original))
cv2.imwrite('C:/Users/15103/Desktop/GaussianBlur77_Lena_de.jpg', GaussianBlur77)
print("\n")

# 中值滤波
# cv2.medianBlur(image, Ksize)
medianBlur33 = cv2.medianBlur(img, 3)
medianBlur55 = cv2.medianBlur(img, 5)
medianBlur77 = cv2.medianBlur(img, 7)

print("经过模板大小为3*3的中值滤波降噪处理后的PSNR值：")
print(PSNR(medianBlur33, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/medianBlur33_Lena_de.jpg',medianBlur33)

print("经过模板大小为5*5的中值滤波降噪处理后的PSNR值：")
print(PSNR(medianBlur55, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/medianBlur55_Lena_de.jpg',medianBlur55)

print("经过模板大小为7*7的中值滤波降噪处理后的PSNR值：")
print(PSNR(medianBlur77, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/medianBlur77_Lena_de.jpg',medianBlur77)
print("\n")

# 双边滤波
"""
cv2.bilaterFilter(image, Ksize, sigmaColor, sigmaSpace) 
Ksize是一个整数，代表了卷积核的大小，
sigmaColor是灰度差值权重的标准差，
sigmaSpace是位置权重的标准差，和前面的高斯滤波的权重是一致的
这两个标准差越大，滤波能力越强，同时还能较好的保留边缘信息。
"""
bilateralFilter3_25 = cv2.bilateralFilter(img, 3, 25, 25)
bilateralFilter3_50 = cv2.bilateralFilter(img, 3, 50, 50)
bilateralFilter3_75 = cv2.bilateralFilter(img, 3, 75, 75)
bilateralFilter5_25 = cv2.bilateralFilter(img, 5, 25, 25)
bilateralFilter5_50 = cv2.bilateralFilter(img, 5, 50, 50)
bilateralFilter5_75 = cv2.bilateralFilter(img, 5, 75, 75)
bilateralFilter7_25 = cv2.bilateralFilter(img, 5, 25, 25)
bilateralFilter7_50 = cv2.bilateralFilter(img, 5, 50, 50)
bilateralFilter7_75 = cv2.bilateralFilter(img, 5, 75, 75)
bilateralFilter9_25 = cv2.bilateralFilter(img, 5, 25, 25)
bilateralFilter9_50 = cv2.bilateralFilter(img, 5, 50, 50)
bilateralFilter9_75 = cv2.bilateralFilter(img, 5, 75, 75)

print("经过卷积核大小为3的、两个标准差为25的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter3_25, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter3_25_Lena_de.jpg',bilateralFilter3_25)

print("经过卷积核大小为3的、两个标准差为50的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter3_50, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter3_50_Lena_de.jpg',bilateralFilter3_50)

print("经过卷积核大小为3的、两个标准差为75的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter3_75, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter3_75_Lena_de.jpg',bilateralFilter3_75)

print("经过卷积核大小为5的、两个标准差为25的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter5_25, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter5_25_Lena_de.jpg',bilateralFilter5_25)

print("经过卷积核大小为5的、两个标准差为50的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter5_50, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter5_50_Lena_de.jpg',bilateralFilter5_50)

print("经过卷积核大小为5的、两个标准差为75的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter5_75, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter5_75_Lena_de.jpg',bilateralFilter5_75)

print("经过卷积核大小为7的、两个标准差为25的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter7_25, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter7_25_Lena_de.jpg',bilateralFilter7_25)

print("经过卷积核大小为7的、两个标准差为50的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter7_50, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter7_50_Lena_de.jpg',bilateralFilter7_50)

print("经过卷积核大小为7的、两个标准差为75的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter7_75, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter7_75_Lena_de.jpg',bilateralFilter7_75)

print("经过卷积核大小为9的、两个标准差为25的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter9_25, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter9_25_Lena_de.jpg',bilateralFilter9_25)

print("经过卷积核大小为9的、两个标准差为50的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter9_50, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter9_50_Lena_de.jpg',bilateralFilter9_50)

print("经过卷积核大小为9的、两个标准差为75的双边滤波降噪处理后的PSNR值：")
print(PSNR(bilateralFilter9_75, img_original))
# cv2.imwrite('C:/Users/15103/Desktop/bilateralFilter9_75_Lena_de.jpg',bilateralFilter9_75)