"""

Simple Focus Stacker

    Author:     Charles McGuinness (charles@mcguinness.us)
    Copyright:  Copyright 2015 Charles McGuinness
    License:    Apache License 2.0


This code will take a series of images and merge them so that each
pixel is taken from the image with the sharpest focus at that location.

The logic is roughly the following:

1.  Align the images.  Changing the focus on a lens, even
    if the camera remains fixed, causes a mild zooming on the images.
    We need to correct the images so they line up perfectly on top
    of each other.

2.  Perform a gaussian blur on all images

3.  Compute the laplacian on the blurred image to generate a gradient map

4.  Create a blank output image with the same size as the original input
    images

4.  For each pixel [x,y] in the output image, copy the pixel [x,y] from
    the input image which has the largest gradient [x,y]
    

This algorithm was inspired by the high-level description given at

http://stackoverflow.com/questions/15911783/what-are-some-common-focus-stacking-algorithms

"""

import numpy as np
import cv2
import logging


class FocusStack:
    def __init__(self, image_stack, debug=False, gaussian_blur_kernel_size=None, laplacian_kernel_size=None,
                 detector=None):
        """
        To use, instantiate with an image_stack, call "focus()" and then read the "focused_image" instance variable.
        
        Notes:
         * Image order matters for image_stack. The first entry is assumed to be the "base" image and is used to align
            all other images against
         * Stacking works best when gaussian and laplacian kernal sizes are the same or similar.
         * If ghosting occurs, it may be because one or more of the image_stack entries is misaligned. The aligned 
            image representations are available as the instance variable "aligned_image_stack" after calling the 
            "focus()" method. 

        :param image_stack: last of numpy arrays, each numpy array representing an image
        :param detector: the OpenCV feature detector to use, defaults to SIFT if not specified
            Note, SIFT generally produces better results, but it is not FOSS, ORB does OK.
            See: https://docs.opencv.org/ref/2.4/d0/d13/classcv_1_1Feature2D.html
        :param gaussian_blur_kernel_size: pixel height and width of Gaussian kernel used to blur images, must be odd,
            defaults to 5 if not specified
        :param laplacian_kernel_size: pixel height and width of Gaussian kernel used to blur images, must be odd,
            defaults to 5 if not specified
        :param debug: set to true to generate debug output 
        """
        assert isinstance(image_stack, list) == True
        assert isinstance(debug, bool) == True
        if gaussian_blur_kernel_size is not None:
            assert gaussian_blur_kernel_size % 2 == 1
        if laplacian_kernel_size is not None:
            assert laplacian_kernel_size % 2 == 1

        self.image_stack = image_stack
        self.detector = detector = cv2.SIFT() if detector is None else detector
        self.debug = debug
        logging.basicConfig(level=(logging.DEBUG if logging.DEBUG else logging.INFO),
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        self.gaussian_blur_kernel_size = 5 if gaussian_blur_kernel_size is None else gaussian_blur_kernel_size
        self.laplacian_kernel_size = 5 if laplacian_kernel_size is None else laplacian_kernel_size
        self.focused_image = None
        self.aligned_image_stack = []

    def focus(self):
        """
        Find the overlapping focus points and align the images accordingly to prdouce a final result.
        :return:
        """
        self._align_images()

        self.logger.debug("Computing the gradient_maps of the blurred images")
        self.logger.debug("Use SIFT {}".format(type(self.detector) == type(cv2.SIFT())))

        gradient_maps = []
        for i in range(len(self.aligned_image_stack)):
            self.logger.debug("Lap {}".format(i))
            gradient_maps.append(self._compute_gradient_map(cv2.cvtColor(self.aligned_image_stack[i], cv2.COLOR_BGR2GRAY)))

        gradient_maps = np.asarray(gradient_maps)
        self.logger.debug("Shape of array of laplacians = {}".format(gradient_maps.shape))
        base_aligned_image = self.aligned_image_stack[0]
        focused_image = np.zeros(shape=base_aligned_image.shape, dtype=base_aligned_image.dtype)

        for y in range(0, self.aligned_image_stack[0].shape[0]):
            for x in range(0, self.aligned_image_stack[0].shape[1]):
                focused_image[y, x] = self._compute_focused_image_pixel(x, y, gradient_maps)

        self.focused_image = focused_image

    def _compute_focused_image_pixel(self, x, y, gradient_maps):
        yxlaps = abs(gradient_maps[:, y, x])
        index = (np.where(yxlaps == max(yxlaps)))[0][0]
        return self.aligned_image_stack[index][y, x]

    def _align_images(self):
        """
        Align image_stack entries to overlap. Assume the first entry in image_stack is the base and align all others
        to it.
        """
        aligned_image_stack = []

        #   We assume that image 0 is the "base" image and align everything to it
        self.logger.debug("Detecting features of base image: {}".format(self.image_stack[0]))
        aligned_image_stack.append(self.image_stack[0])
        base_image_gray_scale = cv2.cvtColor(self.image_stack[0], cv2.COLOR_BGR2GRAY)
        self.base_image_kp, self.base_image_desc = self.detector.detectAndCompute(base_image_gray_scale, None)

        self.aligned_image_stack = [self._align_image_to_base(i, image) for i, image in enumerate(self.image_stack[1:])]

    def _align_image_to_base(self, i, image):
        self.logger.debug("Aligning image {}".format(i))

        image_kp, image_desc = self.detector.detectAndCompute(image, None)
        raw_matches = self._find_base_image_matches(image_desc)
        sorted_raw_matches = sorted(raw_matches, key=lambda x: x.distance)
        matches = sorted_raw_matches[0:128]  # TODO: understand why 0-128 range
        homography = self.find_homography(image_kp, self.base_image_kp, matches)
        return cv2.warpPerspective(image, homography, (image.shape[1], image.shape[0]),
                                       flags=cv2.INTER_LINEAR)

    def _find_base_image_matches(self, image_desc):
        self.logger.debug("Image Description {}".format(image_desc))
        if type(self.detector) == type(cv2.SIFT()):
            bf = cv2.BFMatcher()
            # This returns the top two matches for each feature point (list of list)
            # TODO: understand why k=2
            pairMatches = bf.knnMatch(image_desc, self.base_image_desc, k=2)
            # TODO: understand why 0.7 distance multiplier
            rawMatches = [m for m, n in pairMatches if m.distance < 0.7 * n.distance]
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            rawMatches = bf.match(image_desc, self.base_image_desc)
        return rawMatches

    def _compute_gradient_map(self, image):
        """
        Compute the gradient map of the image
        :return: gradient map computed by applying a Gaussian blur and Laplacian derivative
        """
        blurred = cv2.GaussianBlur(image, (self.gaussian_blur_kernel_size, self.gaussian_blur_kernel_size), 0)
        return cv2.Laplacian(blurred, cv2.CV_64F, ksize=self.laplacian_kernel_size)

    @staticmethod
    def find_homography(image_1_kp, image_2_kp, matches):
        image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

        for i in range(0, len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

        homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

        return homography


def findHomography(image_1_kp, image_2_kp, matches):
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    for i in range(0,len(matches)):
        image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
        image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

    homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

    return homography