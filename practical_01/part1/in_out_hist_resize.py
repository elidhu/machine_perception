# Exercise 1 - Basic Input/Output, Histogram, and Resizing
# Write a program that reads all colour images in a given directory, one by one. For each image
# - Print out the image file name and its dimensions (width and height).
# - Compute and print out the histogram (with 10 uniform bins) of each colour channel (R, G, B).
# Make an observation of the histograms.
# - Reduce the size of the input image by 50% and output this as an image file of the same format.
# Test the program with the images provided (i.e. prac01ex01imgXX.png).

import sys
import os
import glob
from PIL import Image
from shutil import copyfile, copy
import matplotlib.pyplot as plt

EXTENSIONS = ['jpg', 'jpeg', 'png']
OUTPUT = 'output'


def main(argv):
    """Performs the requested tasks as in the comments above for practical 1

    Arguments:
        argv {List} -- The system argv List
    """

    # check there was only 1 command line argument
    if not len(argv) == 2:
        print('Incorrect amount of args, please provide just the relative path to the image dir')
        sys.exit()

    # get the image dir and extract the image names
    image_dir = argv[1]
    image_paths = ['{}/{}'.format(image_dir, f) for f in os.listdir(image_dir) if f.split('.')[-1] in EXTENSIONS]

    # list of images in the directory
    ims = []

    # get the image information
    for image in image_paths:
        # open the image
        im = Image.open(image)
        # log to console
        print('  name: {}\n  format: {}\n  size: {}\n  mode: {}\n'.format(image, im.format, im.size, im.mode))
        ims.append((image, im))

    for loc, im in ims:
        # split the name
        name = loc.split('/')[-1].split('.')[:-1][0]
        fmt = loc.split('.')[-1]

        # make adirectory to store the generated things for each image
        if not os.path.isdir(name):
            os.mkdir(name)
        
        # make a copy of the image in the output directory
        copy(loc, name)

        # get the color histogram of the image it is concatenated RGB and each index
        # contains the number of points of that color
        histogram = im.histogram()

        # take only the Red counts
        r = histogram[0:256]  
        # take only the Green counts
        g = histogram[256:512]
        # take only the Blue counts
        b = histogram[512:768]

        plt.figure()

        # create histogram, they are all on the same figure
        plt.subplot(311)
        for i in range(1, 256):
            plt.bar(i, r[i], color=(i / 255.0, 0, 0))
        plt.subplot(312)
        for i in range(1, 256):
            plt.bar(i, g[i], color=(0, i / 255.0, 0))
        plt.subplot(313)
        for i in range(1, 256):
            plt.bar(i, b[i], color=(0, 0, i / 255.0))

        # save the figure
        plt.savefig('{}/{}_rgb_histogram.{}'.format(name, name, fmt), bbox_inches='tight')

        # resize the image to 50% and save it
        size = (im.width / 2, im.height / 2)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save('{}/{}_resized.{}'.format(name, name, fmt))


if __name__ == '__main__':
    argv = sys.argv
    main(argv)
