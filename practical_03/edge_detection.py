import matplotlib
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
import cv_utils as utils


if __name__ == '__main__':

    images = [
        'images/prac03ex01img01.png',
        'images/prac03ex01img02.png',
        'images/prac03ex01img03.png',
        '/Users/kevinglasson/Google Drive/Uni/2018/Semester 2/machine_perception/practical_03/images/blackandwhite.jpg',
        'images/P1380445.JPG',
        'images/P1380491.JPG'
    ]

    # loop through the images and apply the sobel kernel
    for i in images:
        # read the image in as grayscale and make it float 32
        img = utils.read_color(i)

        # sobel is improved by removing high frequency noise with a small
        # gaussian blur
        # blurred = utils.apply_gaussian_blur(img, 3)
        blurred = img

        # apply the sobel kernel for both ayis
        x = utils.apply_sobel_filter(blurred, 'x')
        y = utils.apply_sobel_filter(blurred, 'y')

        # combine the sobel x and y
        mag = np.hypot(x, y)
        # mag = utils.normalise(mag)

        # utils.show('mag', mag)

        # place all images in one plot
        plot = utils.create_mpl_subplot([
            utils.bgr_to_rgb(img),
            x,
            y,
            mag
        ])

        plot.set_cmap('gray')

        # save the figure
        plot.savefig(
            'edge_output/sobel_{}.png'.format(i.split('/')[-1]), dpi=300)

    # apply canny edge detection using manual thresholding
    for i in images:
        # read the image in as grayscale
        img = utils.read_gray(i)

        edges = cv2.Canny(img, 100, 210)

        plot = utils.create_mpl_subplot([img, edges])

        plot.set_cmap('gray')

        plot.savefig('edge_output/canny_{}.png'.format(i.split('/')[-1]), dpi=300)

    # apply canny edge detection using automatic thresholding
    for i in images:
        # read the image in as grayscale
        img = utils.read_gray(i)

        edges = cv2.Canny(img, 100, 210)

        plot = utils.create_mpl_subplot([img, edges])

        plot.set_cmap('gray')

        plot.savefig('edge_output/auto_canny_{}.png'.format(i.split('/')[-1]), dpi=300)