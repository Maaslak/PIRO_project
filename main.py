import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from skimage import img_as_ubyte
from skimage.feature import canny
from skimage.morphology import closing, square, convex_hull_image
from skimage.transform import rotate, resize, probabilistic_hough_line, ProjectiveTransform, warp
from skimage.util import pad

SHOW_PLT = False

# Weights
K_DISTANCE = 1
K_HU = 1
K_POLYNOMIAL = 1

# How many numbers in every output line?
N = 5

# min hum moment similarity
HU_TH = 0.8


def lines_to_vec(lines):
    return np.array([np.array(line[1]) - np.array(line[0]) for line in lines])


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = int(det(d, xdiff) / div)
    y = int(det(d, ydiff) / div)
    return x, y


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

    thresh = thresh[np.min(rounded[..., -1]): np.max(rounded[..., -1]),
             np.min(rounded[..., 0]): np.max(rounded[..., 0])]
    thresh = resize(thresh, (200, 200))

    chull = convex_hull_image(thresh)
    diff = chull - thresh
    angles = [0, 90, 180, 270]
    parts = [diff[:100], diff[:, 100:], diff[100:], diff[:, :100]]
    angle = angles[np.argmax([np.mean(part) for part in parts])]
    padding = 5

    thresh = pad(rotate(thresh, angle), padding, mode='constant')

    lines = np.array(probabilistic_hough_line(canny(thresh), line_length=50))

    left_line = lines[np.argmin(np.linalg.norm(lines[..., 0], axis=1))]
    right_line = lines[np.argmax(np.linalg.norm(lines[..., 0], axis=1))]

    bottom_line = lines[np.argmax(np.linalg.norm(lines[..., 1], axis=1))]
    up_line = [[0., 0.], [199., 0.]]

    line_pairs = [(left_line, bottom_line), (bottom_line, right_line),
                  (right_line, up_line), (up_line, left_line)]

    try:
        dst = np.array([line_intersection(*pair) for pair in line_pairs])
        src = np.array([[0, 200], [200, 200], [200, 0], [0, 0]])

        tform3 = ProjectiveTransform()
        tform3.estimate(src, dst)
        thresh = warp(thresh, tform3)
    except:
        pass
    thresh = thresh[padding:-3 * padding, padding:- 3 * padding]

    plt.imshow(thresh, cmap=plt.cm.gray)

    # for line in [left_line, right_line, bottom_line]:
    #     p0, p1 = line
    #     plt.plot((p0[0], p1[0]), (p0[1], p1[1]))

    show_plt()

    return thresh


def points_vector(edges):
    points = [np.argwhere(column) for column in edges.T]
    return [np.min(set) if set.size != 0 else 0. for set in points]

class VectorizedImage(object):

    def hu_moments(self, image):
        # m00 m20 m30 mu20
        similars = [1 if HU_TH < self.hu_flipped.get(key) / image.hu_original[key] < 1 + (1-HU_TH) else 0 for key in self.hu_original]
        return sum(similars)

    def get_hu_moments(self, normalized):
        flipped_image = cv2.flip(normalized, 0)
        flipped_image = cv2.flip(flipped_image, 1)
        flipped_edges = canny(flipped_image)
        np.array(self.edges, dtype=np.uint8)

        im1 = np.array(self.edges, dtype=np.uint8)
        im2 = np.array(flipped_edges, dtype=np.uint8)

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(im1)
        axarr[1].imshow(im2)
        # plt.show()
        image_hu = cv2.moments(im1)
        flipped_hu = cv2.moments(im2)
        # sum_hu = {k: int(image_hu.get(k, 0) + flipped_hu.get(k, 0)) for k in set(image_hu)}
        print(image_hu)
        print(flipped_hu)
        # print(sum_hu)
        print()

        return image_hu, flipped_hu

    # TODO
    def _polynomial(self):
        pass

    def __init__(self, image) -> None:
        super().__init__()
        normalized = image_normalization(image)
        edges = canny(normalized)
        self.edges = edges
        self.points = points_vector(edges)
        # plt.imshow(edges, cmap=plt.cm.gray)
        # show_plt()

        self.hu_original, self.hu_flipped = self.get_hu_moments(normalized)

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

    similarities = np.array(
        [[image_a.distance(image_b) for image_a in vectorized_images] for image_b in vectorized_images])
    hu = np.array([[image_a.hu_moments(image_b) for image_a in vectorized_images] for image_b in vectorized_images])

    # normalize
    distance_result = [pow(l / min(l), -1) for l in similarities]

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
