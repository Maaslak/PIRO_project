import sys
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import rotate, resize
from skimage import img_as_ubyte
from skimage.feature import canny

import cv2

def lines_to_vec(lines):
    return np.array([np.array(line[1])-np.array(line[0]) for line in lines])


def image_normalization(image):

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

    return thresh


def points_vector(edges):
    points = [np.argwhere(column) for column in edges.T]
    return [np.median(set) if set.size != 0 else 0. for set in points]


class VectorizedImage(object):

    # TODO
    def _hu_moments(self):
        pass

    # TODO
    def _polynomial(self):
        pass

    def __init__(self, image) -> None:
        super().__init__()
        normalized = image_normalization(image)[5:-5, 5:-5]
        edges = canny(normalized)
        self.points = points_vector(edges)
        # plt.imshow(edges, cmap=plt.cm.gray)
        # plt.show()

    # TODO works only on sums representation, should be extended by hu moments and polynomial values
    def distance(self, image):
        reversed_points = image.points[::-1]
        sums = [a + b for a, b in zip(self.points, reversed_points)]
        return np.std(sums)


def run_alg(data_dir, set_no):
    vectorized_images = []
    for img_no in range(set_no):
        image = imread("{}/{}.png".format(data_dir, img_no))
        vectorized_images.append(VectorizedImage(image))




if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise TypeError("missing parameters")
    run_alg(sys.argv[1], int(sys.argv[2]))
