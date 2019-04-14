import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

from skimage.transform import probabilistic_hough_line
from skimage import feature
from skimage.morphology import disk, dilation


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
    plt.show()


def vectorize_image(image):
    find_base(image)


def run_alg(data_dir, set_no):
    for img_no in range(set_no):
        image = imread("{}/{}.png".format(data_dir, img_no))
        vectorize_image(image)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise TypeError("missing parameters")
    run_alg(sys.argv[1], int(sys.argv[2]))
