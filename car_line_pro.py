# # 导入os模块，用于执行操作系统命令
# import os
# # 使用pip命令安装opencv-python库，这是cv2的Python接口
# os.system('pip install opencv-python')

import numpy as np
import cv2

img = cv2.imread('vr.jpg')
# cv2.imshow('img', img)
# cv2.waitKey(0)

# 转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg', gray)
# cv2.waitKey(0)

"""
中间可以使用滤波做处理
高斯滤波：cv2.GaussianBlur(图像:gray, 高斯核:[5, 5], sigmaX:1)
2D滤波：cv2.filter2D(图像:gray, deep:-1, kernel)
归一化卷积框：cv2.blur(图像:gray, ksize:(5, 5))
中值模糊：cv2.medianBlur(图像:gray, ksize:5)
"""

# Canny算子获得边缘
edge_img = cv2.Canny(gray, 210, 300)
cv2.imwrite('edge_img.jpg', edge_img)
# cv2.waitKey(0)

# 取车道线的区域
poly_pts = np.array([[[0, 460], [1150, 470], [780, 0], [650, 0]]]) # 需要自己标定
mask = np.zeros_like(edge_img)

cv2.fillPoly(mask, poly_pts, color=255)
img_mask = cv2.bitwise_and(edge_img, mask)
cv2.imwrite('img_mask.jpg', img_mask)
# cv2.waitKey(0)

# hough变换获得直线
lines = cv2.HoughLinesP(img_mask, 1, np.pi/180, 15)

# 计算斜率的函数
def calculate_slope(line):
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2-y_1)/(x_2-x_1)

left_line = [x for x in lines if calculate_slope(x) > 0]
right_line = [x for x in lines if calculate_slope(x) < 0]


def reject_abnormal_lines(lines, threshold=0.2):
    """
    剔除斜率不一致的线段
    :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
    """
    slopes = [calculate_slope(line) for line in lines]
    while len(lines) > 0:
        mean = np.mean(slopes)
        diff = [abs(s - mean) for s in slopes]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines

reject_abnormal_lines(left_line)
reject_abnormal_lines(right_line)

# numpy用最小二乘法拟合直线
def least_fit(lines):
    """
            将lines中的线段拟合成一条线段
            :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
            :return: 线段上的两点,np.array([[xmin, ymin], [xmax, ymax]])
            """
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    poly = np.polyfit(x_coords, y_coords, deg=1)
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int64)

# 获得左右两条车道线
left_line = least_fit(left_line)
right_line = least_fit(right_line)

# 在图上画线
cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 0, 255), thickness=2)
cv2.line(img, tuple(right_line[0]), tuple(right_line[1]), color=(0, 0, 255), thickness=2)

cv2.imwrite('img.jpg', img)