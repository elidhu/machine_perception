# Exercise 2 - Cropping
# In this exercise, you are given an image file (prac01ex02img01.png) and a text file (prac01ex02crop.txt)
# that contains the coordinates of the top-left and bottom-right corners of a rectangular region to crop
# (x l y l x r y r). Write a program that reads this image and extracts the rectangular region. Save
# the extracted region as an image file and observe its content.

from cv2 import cv2
import sys
import numpy as np

def crop(infile, x1, y1, x2, y2):
    """crop the given image file to the rectangle (x1, y1) (x2, y2)
    
    Arguments:
        infile {str} -- the path to the image file
        x1 {int} -- upper left x coord of the bounding box
        y1 {int} -- upper left y coord of the bounding box
        x2 {int} -- lower right x coord of the bouding box
        y2 {int} -- lower right y coord of the bounding box
    """
    # open the image
    im = cv2.imread(infile)
    # crop using array slicing
    crop_im = im[y1:y2, x1:x2]
    # write to file in the current dir
    cv2.imwrite('cropped_{}'.format(infile), crop_im)

def draw_bb(infile, x1, y1, x2, y2, color=(0, 255, 0)):
    """draw a bounding box given 2 points on a rectangle
    
    Arguments:
        infile {str} -- the path to the image file
        x1 {int} -- upper left x coord of the bounding box
        y1 {int} -- upper left y coord of the bounding box
        x2 {int} -- lower right x coord of the bouding box
        y2 {int} -- lower right y coord of the bounding box
    
    Keyword Arguments:
        color {tuple} -- rbg color for the bounding box (default: {(0, 255, 0)})
    """
    # open the image
    im = cv2.imread(infile)
    # draw the rectangular bounding box
    bb_im = cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
    # write to file in the current dir
    cv2.imwrite('bb_{}'.format(infile), bb_im)

def read_bb_array(infile):
    """read the bounding box from file in the form 'x1 y1 x2 y2'
    
    Arguments:
        infile {str} -- the path to the bounding box file
    """
    # load the single line text file into a np array
    bb = np.loadtxt(infile, dtype='int')
    # extract values from the array, name them, and return them
    x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
    return(x1, y1, x2, y2)

def show(im, name="image"):
    """show the cv2 image object
    
    Arguments:
        im {cv2 image} -- the cv2 image
    
    Keyword Arguments:
        name {str} -- the name for the image display window (default: {"image"})
    """
    # show the image with the given name
    cv2.imshow(name, im)
    # notify the user and wait for a key press to continue
    print('Press any key with the image focused to close')
    cv2.waitKey(0)

def rotation_delta(im1, im2):
    """calculate the angle and image has been rotated
    
    Arguments:
        im1 {cv2 image} -- the first image
        im2 {cv2 image} -- the second image
    """
    # calculate how much im1 has been roated by to get im2
    pass


if __name__ == '__main__':
    # get the sys arguments
    argv = sys.argv
    # get the input file
    infile = argv[1]
    # get the bounding box details
    bbfile = argv[2]
    # read in the bounding box file
    box = read_bb_array(bbfile)
    # crop the image to bb
    crop(infile, *box)
    # draw the bb on the image
    draw_bb(infile, *box)