from mp_utils import features, filters, images, utils
from mp_utils import visualisations as vis
from cv2 import cv2
import numpy as np
import time


def main():
    image_paths = []

    image_paths =  \
        utils.get_images_in_dir('sample_data/SetC') + \
        utils.get_images_in_dir('sample_data/SetD')

        # utils.get_images_in_dir('sample_data/SetA') + \
        # utils.get_images_in_dir('sample_data/SetB') + \
        # utils.get_images_in_dir('sample_data/SetC') + \
        # utils.get_images_in_dir('sample_data/SetD')

    # image_paths = utils.get_images_in_dir('contour_problems')

    print(image_paths)

    # not important, for grouping images and debugging...
    i = 0

    for path in image_paths:

        # read in the image
        image = images.read_color(path)

        gray = images.bgr_to_gray(image)

        lab = images.bgr_to_lab(image)

        vis.show_cv_image([lab[0], lab[1], lab[2]])

        r = 200.0 / image.shape[1]
        dim = (200, int(image.shape[0] * r))
        # perform the actual resizing of the image and show it
        resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

        vis.show_cv_image(resized)

        blurred = filters.apply_gaussian_blur(resized, 13)
        bil = cv2.bilateralFilter(blurred, 11, 27, 27)


        canny = features.apply_canny(bil, 100, 180)
        vis.show_cv_image(canny)

        # kernel_e = np.ones((17, 17), np.uint8)
        # opening = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel_e)

        # vis.show_cv_image(opening)

        thresh = cv2.threshold((canny), 127, 255, cv2.THRESH_BINARY)[1]
        vis.show_cv_image(thresh)

        # apply a hough transform to get the lines
        lines = cv2.HoughLinesP(canny, 1, np.pi/180, 30,
                                minLineLength=50, maxLineGap=250)

        # draw the hough lines on the image
        hough_image = features.draw_hough_lines(resized, lines)

        vis.show_cv_image(hough_image)

        retval, labels = cv2.connectedComponents(bil)

        ##################################################
        ts = time.time()
        num = labels.max()

        N = 50
        for i in range(1, num+1):
            pts =  np.where(labels == i)
            if len(pts[0]) < N:
                labels[pts] = 0

        print("Time passed: {:.3f} ms".format(1000*(time.time()-ts)))
        # Time passed: 4.607 ms

        ##################################################

        # Map component labels to hue val
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue==0] = 0

        cv2.imshow('labeled.png', labeled_img)
        cv2.waitKey()
        ##################################################

        # # RETR_EXTERNAL returns the contours from the top level heirarchy
        # _, cnts, _ = cv2.findContours(
        #     labeled_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # cnts = cnts[1]
        # # asort based on the contour area
        # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        # # list to store the four pointed contours
        # four_pointed_countours = []

        # # loop over the contours
        # for c in cnts:
        #     # approximate the contour
        #     peri = cv2.arcLength(c, True)
        #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        #     area = cv2.contourArea(approx)

        #     # if our approximated contour has four points, then we
        #     # can assume that we have found our screen
        #     # TODO: experiment with the area value a little more! but it's working
        #     if len(approx) == 4 and area > 100000:
        #         four_pointed_countours.append(approx)
        #         # break

        # for cont in four_pointed_countours:
        #     cv2.drawContours(image, [cont], -1, (0, 255, 0), 10)

        # name, _ = utils.get_file_and_ext(path)
        # # vis.save_mpl_subplot(plot, 'contour_out/{}.png'.format(name))

        # cv2.imwrite('contour_out/0{}{}.png'.format(i, name), image)
        # # cv2.imwrite('contour_out/0{}_{}.png'.format(i, name), canny)
        # # cv2.imwrite('contour_out/0{}__{}..png'.format(i, name), hough_image)

        # # again not important
        # i += 1


if __name__ == '__main__':
    main()
