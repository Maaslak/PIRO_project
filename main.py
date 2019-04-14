import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread

from skimage.transform import probabilistic_hough_line
from skimage import feature


def find_base(image):
    edges = feature.canny(image, 1)
    lines = probabilistic_hough_line(edges, threshold=50, line_length=30)

    for line in lines:
        p0, p1 = line
        # plt.plot((p0[0], p1[0]), (p0[1], p1[1]))

    # plt.imshow(edges, cmap=plt.cm.gray)
    # plt.show()


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
