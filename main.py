import sys
from matplotlib.image import imread
import numpy as np

from skimage.transform import rotate, resize
from skimage import img_as_ubyte

import cv2


def lines_to_vec(lines):
    return np.array([np.array(line[1])-np.array(line[0]) for line in lines])


def get_normalized_image(image):

    image_cv = img_as_ubyte(image)

    ret, thresh = cv2.threshold(image_cv, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]

    rect = cv2.minAreaRect(cnt)
    center, (w, h), angle = rect

    image = rotate(image, rect[-1], resize=True)

    if h > w:
        image = rotate(image, 90, resize=True)

    image_cv = img_as_ubyte(image)
    ret, thresh = cv2.threshold(image_cv, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)

    points = cv2.boxPoints(rect)

    rounded = np.round(points).astype(np.int)
    thresh = thresh[rounded[2, 1]: rounded[0, 1], rounded[0, 0]: rounded[2, 0]]
    thresh = resize(thresh, (200, 200))

    if np.mean(thresh[:100]) > np.mean(thresh[100:]):
        thresh = rotate(thresh, 180)

    # plt.imshow(thresh, cmap=plt.cm.gray)
    # plt.show()

    return thresh


def vectorize_image(image):
    get_normalized_image(image)


def run_alg(data_dir, set_no):
    for img_no in range(set_no):
        image = imread("{}/{}.png".format(data_dir, img_no))
        vectorize_image(image)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise TypeError("missing parameters")
    run_alg(sys.argv[1], int(sys.argv[2]))
