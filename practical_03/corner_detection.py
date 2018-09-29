from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv_utils as utils

if __name__ == '__main__':

    # the list of the images
    images = [
        'images/prac03ex01img01.png',
        'images/prac03ex01img02.png',
        'images/prac03ex01img03.png',
        ]

    # process all the images the same
    for i in images:
        img_col = utils.read_color(i)
        
        harris = utils.apply_harris_corners(img_col)
        shi = utils.apply_shi_tomasi_corners(img_col)
        
        plot = utils.create_mpl_subplot([
            utils.bgr_to_rgb(img_col),
            utils.bgr_to_rgb(harris),
            utils.bgr_to_rgb(shi)
            ])

        plot.savefig('corner_output/corners_{}'.format(i.split('/')[-1]))
        # plt.show()