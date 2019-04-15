import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

from skimage.transform import probabilistic_hough_line
from skimage import feature
from skimage import transform
from skimage import measure
from skimage.morphology import disk, dilation
from skimage.draw import polygon

CONTOUR_PRECISION = 2
IMAGE_SIZE = 100


class Image:
    def __init__(self, image):
        self.image = transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        self.contour_points = self.detect_contour()

    def detect_contour(self):
        edges = feature.canny(self.image, 1)
        edges = dilation(edges, disk(2))

        # Zamiast edges chyba można dać po prostu image
        contours = measure.find_contours(edges, 0)
        # Najbardziej odpowiedni kontur znajduje się na pozycji 0
        contour = contours[0]

        coords = measure.approximate_polygon(contour, CONTOUR_PRECISION)
        contour_image = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
        plt.plot(coords[:, 1], coords[:, 0], '-w', linewidth=2)
        # contour_image[coords[:, 1]][coords[:, 0]] = 1
        # print(coords[:, 1], coords[:, 0])
        # rr, cc = polygon(coords[:, 1], coords[:, 0])
        #
        # contour_image[rr, cc] = 1
        plt.imshow(contour_image)
        plt.show()
        return coords


def lines_to_vec(lines):
    return np.array([np.array(line[1])-np.array(line[0]) for line in lines])


def find_base(image):
    edges = feature.canny(image, 1)
    edges = dilation(edges, disk(2))

    base_lines = np.array(probabilistic_hough_line(edges, threshold=20, line_length=70, line_gap=6))
    side_lines = probabilistic_hough_line(edges, threshold=20, line_length=15, line_gap=2)

    base_vectors = lines_to_vec(base_lines)
    side_vectors = lines_to_vec(side_lines)

    dot_product = base_vectors @ side_vectors.T  # if 0 then orthogonal

    base_dists = np.array([np.linalg.norm(base_vectors, axis=1)])
    side_dists = np.array([np.linalg.norm(side_vectors, axis=1)])

    norm_dot = base_dists.T @ side_dists
    cos_theta = dot_product / norm_dot
    cos_theta_thr = 0.02

    orthogonal = cos_theta < cos_theta_thr

    bases = np.argsort(base_dists)[::-1]

    bases = list(filter(lambda i: np.count_nonzero(orthogonal[i]) > 1, bases))[0]

    p0, p1 = base_lines[bases[0]]
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]), linewidth=3)

    for base_id in bases[:3]:
         p0, p1 = base_lines[base_id]
         plt.plot((p0[0], p1[0]), (p0[1], p1[1]), linewidth=3)

    plt.imshow(edges, cmap=plt.cm.gray)
    # plt.show()


def vectorize_image(image):
    find_base(image)


def run_alg(data_dir, set_no):
    for img_no in range(set_no):
        image = imread("{}/{}.png".format(data_dir, img_no))
        # vectorize_image(image)

        _image = Image(image)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise TypeError("missing parameters")
    run_alg(sys.argv[1], int(sys.argv[2]))
