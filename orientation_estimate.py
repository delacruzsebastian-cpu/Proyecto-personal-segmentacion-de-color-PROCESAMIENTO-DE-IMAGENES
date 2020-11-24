import numpy as np
import cv2


def gradient_map(gray):
    # Image derivatives
    SCALE = 1
    DELTA = 0
    DDEPTH = cv2.CV_16S  ## to avoid overflow

    grad_x = cv2.Sobel(gray, DDEPTH, 1, 0, ksize=3, scale=SCALE, delta=DELTA)
    grad_y = cv2.Sobel(gray, DDEPTH, 0, 1, ksize=3, scale=SCALE, delta=DELTA)

    grad_x = np.float32(grad_x)
    grad_x = grad_x * (1 / 512)
    grad_y = np.float32(grad_y)
    grad_y = grad_y * (1 / 512)

    # Gradient and smoothing
    grad_x2 = cv2.multiply(grad_x, grad_x)
    grad_y2 = cv2.multiply(grad_y, grad_y)

    # Magnitude of the gradient
    Mag = np.sqrt(grad_x2 + grad_y2)

    # Orientation of the gradient
    theta = np.arctan(cv2.divide(grad_y, grad_x + np.finfo(float).eps))

    return theta, Mag


def orientation_map(gray, n):
    # Image derivatives
    SCALE = 1
    DELTA = 0
    DDEPTH = cv2.CV_16S  ## to avoid overflow

    grad_x = cv2.Sobel(gray, DDEPTH, 1, 0, ksize=3, scale=SCALE, delta=DELTA)
    grad_y = cv2.Sobel(gray, DDEPTH, 0, 1, ksize=3, scale=SCALE, delta=DELTA)

    grad_x = np.float32(grad_x)
    grad_x = grad_x * (1 / 512)
    grad_y = np.float32(grad_y)
    grad_y = grad_y * (1 / 512)

    # Gradient and smoothing
    grad_x2 = cv2.multiply(grad_x, grad_x)
    grad_y2 = cv2.multiply(grad_y, grad_y)
    grad_xy = cv2.multiply(grad_x, grad_y)
    g_x2 = cv2.blur(grad_x2, (n, n))
    g_y2 = cv2.blur(grad_y2, (n, n))
    g_xy = cv2.blur(grad_xy, (n, n))

    # Magnitude of the gradient
    Mag = np.sqrt(grad_x2 + grad_y2)
    M = cv2.blur(Mag, (n, n))

    # Gradient local aggregation
    vx = 2 * g_xy
    vy = g_x2 - g_y2
    fi = cv2.divide(vx, vy + np.finfo(float).eps)

    case1 = vy >= 0
    case2 = np.logical_and(vy < 0, vx >= 0)
    values1 = 0.5 * np.arctan(fi)
    values2 = 0.5 * (np.arctan(fi) + np.pi)
    values3 = 0.5 * (np.arctan(fi) - np.pi)
    theta = np.copy(values3)
    theta[case1] = values1[case1]
    theta[case2] = values2[case2]

    return theta, M
