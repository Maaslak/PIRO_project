import sys
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import closing, square

from skimage.transform import rotate, resize
from skimage import img_as_ubyte
from skimage.feature import canny

import cv2

SHOW_PLT = False

# Weights
K_DISTANCE = 1
K_HU = 1
K_POLYNOMIAL = 1

# How many numbers in every output line?
N = 5


def lines_to_vec(lines):
    return np.array([np.array(line[1])-np.array(line[0]) for line in lines])


def image_normalization(image):

    image_cv = img_as_ubyte(image)

    ret, thresh = cv2.threshold(image_cv, 127, 255, 0)
    thresh = closing(thresh, square(2))
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
    cnt = contours[np.argmax([contour.size for contour in contours])]
    rect = cv2.minAreaRect(cnt)

    points = cv2.boxPoints(rect)

    rounded = np.round(points).astype(np.int)

    thresh = thresh[np.min(rounded[..., -1]): np.max(rounded[..., -1]), np.min(rounded[..., 0]): np.max(rounded[..., 0])]
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
        plt.imshow(edges, cmap=plt.cm.gray)
        show_plt()

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

    similarities = np.array([[image_a.distance(image_b) for image_a in vectorized_images] for image_b in vectorized_images])

    # normalize
    distance_result = [pow(l/min(l), -1) for l in similarities]

    # TODO substitute for real results (this is mock of hu_result and polynomial_result)
    hu_result = np.ones_like(distance_result)
    polynomial_result = np.ones_like(distance_result)

    weights = [K_DISTANCE, K_HU, K_POLYNOMIAL]
    methods_results = [distance_result, hu_result, polynomial_result]

    return get_final_rank(list(zip(weights, methods_results)))


# The final ranking
def get_final_rank(components):
    results = []
    for component in components:
        results.append([component[0] * l for l in component[1]])

    _result = np.sum(np.dstack((results[0], results[1])), axis=2)
    _result = [np.argsort(l)[::-1] for l in _result]
    return _result


def show_plt():
    if SHOW_PLT:
        plt.show()


def print_results(_results):
    for r in _results:
        print(*r[:N], sep=", ")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        SHOW_PLT = int(sys.argv[3])
    elif len(sys.argv) < 3:
        raise TypeError("missing parameters")
    result = run_alg(sys.argv[1], int(sys.argv[2]))

    print_results(result)
