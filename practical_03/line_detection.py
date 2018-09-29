
import matplotlib
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
import cv_utils as u


def test():
    img_paths = [
        'images/prac03ex01img01.png',
        'prac03ex03img01.png',
        'prac03ex03img02.jpg'
    ]

    images = []
    for path in img_paths:
        images.append(u.read_color(path))

    for img in images:
        plot = u.create_mpl_histogram_color(img, ('b', 'g', 'r'))
        plot.show()

# https://alyssaq.github.io/2014/understanding-hough-transform/


def attempt_hough():

    # get the images
    # img_paths = u.get_images_in_dir('images')
    img_paths = [
        'images/prac03ex01img01.png',
        'images/prac03ex03img01.png',
        'images/prac03ex03img02.jpg'
    ]

    for path in img_paths:
        # read in the image as color
        image_col = u.read_color(path)

        # make a grayscale copy
        image = u.bgr_to_gray(image_col)

        # apply a canny edge detector
        canny = u.apply_auto_canny(image, 0.33)

        # apply a hough transform to get the lines
        lines = cv2.HoughLinesP(canny, 1, np.pi/180, 30,
                                minLineLength=50, maxLineGap=250)

        # draw the hough lines on the image
        hough_image = u.draw_hough_lines(image_col, lines)

        # put everything into a plot
        plot = u.create_mpl_subplot(
            [canny, image, u.bgr_to_rgb(image_col), hough_image],
            False
        )

        # show the plot
        # plot.show()

        u.save_mpl_subplot(
            plot, 'line_output/{}.png'.format(path.split('/')[-1]))


def apply_hough_tranform(image):
    # create array of all of the theta's to check
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    # get the width and the height of the image
    w, h = image.shape
    # calculate the max length that rho could be (diagonal of the image)
    max_len = np.ceil(np.hypot(w, h))
    # create array of the rho values to check
    rhos = np.linspace(-max_len, max_len, max_len * 2.0)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    num_rhos = len(rhos)

    # create a blank hough accumulator
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.uint64)

    # get the indexes of the edges, all others will be zero because the image
    # has already been edge detected
    y_idxs, x_idxs = np.nonzero(image)  # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. max_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + max_len)
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


if __name__ == '__main__':
    attempt_hough()
