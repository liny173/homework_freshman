import numpy as np
import cv2


img = cv2.imread('682.bmp')
# cv2.imshow('img', img)
# cv2.waitKey(0)

gauss = cv2.GaussianBlur(img, [5, 5], 1)
# cv2.imwrite('G_img.jpg', gauss)
# cv2.imshow('img', gauss)
# cv2.waitKey(0)

edge = cv2.Canny(img, 200, 245)
# cv2.imwrite('E_img.jpg', edge)
# cv2.imshow('img', edge)
# cv2.waitKey(0)

print(img.shape[0], img.shape[1])
mask = np.zeros_like(edge)
poly_pts = np.array([[[50, 130], [0, 200], [300, 320], [300, 120]]])

cv2.fillPoly(mask, poly_pts, color=255)
img_mask = cv2.bitwise_and(edge, mask)
# cv2.imwrite('M_img.jpg', img_mask)
# cv2.imshow('img', img_mask)
# cv2.waitKey(0)
# #
lines = cv2.HoughLinesP(img_mask, 1, np.pi/180, 15)

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

left_line = least_fit(left_line)
right_line = least_fit(right_line)

cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 0, 255), thickness=2)
cv2.line(img, tuple(right_line[0]), tuple(right_line[1]), color=(0, 0, 255), thickness=2)

# cv2.imwrite('car_img.jpg', img)
cv2.imshow('img', img)
cv2.waitKey(0)