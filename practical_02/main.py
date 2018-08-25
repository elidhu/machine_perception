from cv2 import cv2
import os
import numpy as np

EXTENSIONS = ['jpg', 'jpeg', 'png']
IMAGE_DIR = 'images'
OUTPUT_DIR = 'output'


def main():
    # list images in the image directory, provided they have one of the specified extensions
    images = ['{}/{}'.format(IMAGE_DIR, f)
              for f in os.listdir(IMAGE_DIR) if f.split('.')[-1] in EXTENSIONS]

    # run exercise 1 on all of the images in the directory
    for img in images:
        # ex1(img)
        ex2(img)


def ex2(img):
    """preforms linear filtering as per prac2

    :param img: the path to an image
    :type img: str

    That is, the kernel is not mirrored around the anchor point. If you need a
    real convolution, flip the kernel using flip() and set the new anchor to
    (kernel.cols - anchor.x - 1, kernel.rows - anchor.y - 1)
    """
    # load the image and convert to gray
    image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
    # store the converted images
    cvt = [image]

    # apply prewit kernel
    prewit = get_prewit_kernel()
    # flip the kernal about both axis before convolution
    prewit = cv2.flip(prewit, -1)
    # append the convolution output
    cvt.append(cv2.filter2D(image, -1, prewit))

    # apply sobel x kernel
    sobel = get_sobel_kernel()
    s_x = sobel['x']
    # flip the kernal about both axis before convolution
    s_x = cv2.flip(s_x, -1)
    # append the convolution output
    cvt.append(cv2.filter2D(image, -1, s_x))

    # apply sobel y kernel
    s_y = sobel['y']
    # flip the kernal about both axis before convolution
    s_y = cv2.flip(s_y, -1)
    # append the convolution output
    cvt.append(cv2.filter2D(image, -1, s_y))

    # apply laplacian kernel
    laplacian = get_laplacian_kernel()
    # flip the kernal about both axis before convolution
    laplacian = cv2.flip(laplacian, -1)
    # append the convolution output
    cvt.append(cv2.filter2D(image, -1, laplacian))

    # apply gaussian kernel
    gaussian = get_gaussian_kernel()
    # flip the kernal about both axis before convolution
    gaussian = cv2.flip(gaussian, -1)
    # append the convolution output
    cvt.append(cv2.filter2D(image, -1, gaussian))

    # concatenate the images into a single array
    array = np.concatenate(tuple(cvt), axis=1)

    # create the output dir if it doesn't exist
    if not os.path.isdir('ex_2_' + OUTPUT_DIR):
        os.mkdir('ex_2_' + OUTPUT_DIR)

    # write the created image to the output dir
    cv2.imwrite(os.path.join('ex_2_' + OUTPUT_DIR,
                             os.path.basename(img)), array)


def ex1(img):
    # read the image in
    image = cv2.imread(img)

    # store the converted images
    cvt = []

    # convert the image to GRAY
    # note it is converted back to bgr to make it 3 channel for the purposes of
    # displaying ti next to the others (arrays need to be the same size)
    # it's still GRAY though!
    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cvt.append({
        'name': 'GRAY',
        'img': cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
    })
    # convert the image to HSV
    cvt.append({
        'name': 'HSV',
        'img': cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    })
    # convert the image to LUV
    cvt.append({
        'name': 'LUV',
        'img': cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    })
    # convert the image to LAB
    cvt.append({
        'name': 'LAB',
        'img': cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    })

    # Extract all of the images from the dict into a list
    img_arrays = [x['img'] for x in cvt]

    # insert the original image at the start of the list
    img_arrays.insert(0, image)

    # concatenate the images into a single array
    array = np.concatenate(tuple(img_arrays), axis=1)

    # cv2.imshow('Images', array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # create the output dir if it doesn't exist
    if not os.path.isdir('ex_1_' + OUTPUT_DIR):
        os.mkdir('ex_1_' + OUTPUT_DIR)

    # write the created image to the output dir
    cv2.imwrite(os.path.join('ex_1_' + OUTPUT_DIR,
                             os.path.basename(img)), array)


def get_prewit_kernel():
    return(np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1]).reshape(3, 3))


def get_sobel_kernel():
    sobel = {
        'x': np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(3, 3),
        'y': np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape(3, 3)
    }
    return(sobel)


def get_laplacian_kernel():
    return(np.array([0, 1, 0, 1, -4, 1, 0, 1, 0]).reshape(3, 3))


def get_gaussian_kernel():
    return((1 / 273.0) * np.array([1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1]))


if __name__ == '__main__':
    main()
