import io
import cv2
import numpy as np
import math
import scipy.stats as ss
from scipy.optimize import least_squares


RECT_SCALE = 1000
bg_R = np.array([150.0 / 255.0, 150.0 / 255.0, 150.0 / 255.0])


def get_average_rgb(image_data):
    return np.average(image_data, axis=(0, 1))


def crop_image_by_position_and_rect(cv_image, position, rect):
    # img[y: y + h, x: x + w]
    height = cv_image.shape[0]
    width = cv_image.shape[1]
    position_x = position.x * width
    position_y = position.y * height
    rect_x = width * rect.x / RECT_SCALE
    rect_y = height * rect.y / RECT_SCALE
    return cv_image[int(position_y):int(position_y) + int(rect_y),
                    int(position_x):int(position_x) + int(rect_x)]


def read_matrix(path, n_params):
    H = None
    line_arr = np.array([])
    count = 0
    with open(path) as f:
        f.readline()
        for line in f:
            if "=" in line:
                count += 1
                if H is None:
                    H = line_arr
                else:
                    H = np.vstack((H, line_arr))
                line_arr = np.array([])
                continue
            if count >= n_params:
                break
            line_arr = np.hstack((line_arr, line.split()))
    return H.astype(np.float64)


def predict(H, params):
    return np.sort(np.abs(H[1, :] +
                          np.matmul(params.transpose(), H[2:, :])))


def SNIC_EMOR(background, target, params):
    target = gaussian_blur_background(target, 2, 5)
    background = gaussian_blur_background(background, 2, 5)

    H = read_matrix("invemor.txt", 5)
    crf = predict(H, params)

    def correct_intensity(colors, crf):
        x = np.linspace(0, 1, crf.size)
        return np.interp(colors, x, crf)
    x, y, c = target.shape
    b_x, b_y, c = background.shape
    image_norm = np.reshape(target, (x * y, 3)) / 255.0
    bg_image_norm = np.reshape(background, (b_x * b_y, 3)) / 255.0
    image_ic = correct_intensity(image_norm, crf)
    bg_image_ic = correct_intensity(bg_image_norm, crf)
    image = np.reshape(image_ic, (x, y, 3))
    bg_image = np.reshape(bg_image_ic, (b_x, b_y, 3))
    background[background == 0] = 1E-4
    R = np.multiply(np.divide(image, bg_image), bg_R)
    image = np.clip(R, 0.0, 1.0) * 255.0
    image = image.astype(np.uint8)
    return image


def gaussian_blur_background(cv_image, repeat=1, size=31):
    for x in range(5):
        cv_image = cv2.GaussianBlur(cv_image, (size, size), 0)
    return cv_image


def morphology_close_background(cv_image, repeat=1, size=31):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    for x in range(repeat):
        cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)
        cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_OPEN, kernel)
    return cv_image
