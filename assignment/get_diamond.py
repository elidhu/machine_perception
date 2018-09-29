from mp_utils import features, filters, images, utils
from mp_utils import visualisations as vis
from cv2 import cv2
import numpy as np

from math import hypot

@utils.timing
def main():
    image_paths = []

    # get teh paths to the image
    image_paths =  \
        utils.get_images_in_dir('./sample_data/SetA') + \
        utils.get_images_in_dir('./sample_data/SetB') + \
        utils.get_images_in_dir('./sample_data/SetC') + \
        utils.get_images_in_dir('./sample_data/SetD')

    # image_paths =  \
    #     utils.get_images_in_dir('./contour_problems')

    # print(image_paths)

    # not important, for grouping images and debugging...
    i = 0

    # process each image
    for path in image_paths:
        image = images.read_color(path)
        gray = images.bgr_to_gray(image)

        # threshold the image
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 47, 1)
        # blur the image
        blurred = filters.apply_gaussian_blur(thresh, 9)
        bil = cv2.bilateralFilter(blurred, 11, 60, 60)

        # detect the edges
        canny = features.apply_canny(bil, 175, 190)

        # TODO: work on this, it tries to dilate int the direction of the diamond lines
        # this one does 45deg
        # kernel = np.array([
        #     1, 0, 1,
        #     0, 1, 0,
        #     1, 0, 1
        # ], np.uint8).reshape(3, 3)

        # find the contours
        cnts = features.apply_find_contours(canny, 4, 80000, 5)

        # attempt to de-noise contours
        # (highest x, highest y), (highest x, lowest y), (lowest x, highest y) and (lowest x, lowest y)

        # cnts = remove_closest(cnts, 0)

        cnt_image = features.draw_contours(image, cnts)

        # save the iamges for debugging
        name, _ = utils.get_file_and_ext(path)
        cv2.imwrite('contour_out/0{}{}.png'.format(i, name), cnt_image)
        cv2.imwrite('contour_out/0{}_{}.png'.format(i, name), canny)
        # again not important
        i += 1

def remove_closest(points, thresh):
    distances = []

    for p in np.nditer(points):
        print(p)
        
        for q in np.nditer(points):
            d = distance(p, q), (p, q)

            if not d == 0:
                distances.append(d)
        
    distances.sort(key=lambda tup: tup[0], reverse=True)

    print(distances)

    return distances
        

def distance(p, q):
    """compute the euclidean distance between 2 points
    
    Parameters
    ----------
    p : point
        first point
    q : point
        second point
    
    Returns
    -------
    float
        distance
    """
    return hypot(p[0] - q[0], p[1] - q[1])


if __name__ == '__main__':
    main()
