import os
import cv2
import FocusStack
import sys
import argparse

"""
    Focus stack driver program. 

    This program looks for a series of files of type .jpg, .jpeg, or .png
    in a specified subdirectory, by default "input", and then merges them together using the
    FocusStack module.  The output is written to a specified file, by default "merged.png"

    Author:     Charles McGuinness (charles@mcguinness.us)
    Copyright:  Copyright 2015 Charles McGuinness
    License:    Apache License 2.0

"""

def parse_args(args):
    """
    Parse command line arguments
    :param args: array of arguments to parse, typically sys.argv[1:]
    :return: Namespace of arguments
    >>> args = parse_args([])
    >>> args.input == 'input'
    True
    >>> args.output == 'merged'
    True
    >>> args = parse_args(['--input', 'mypath', '--output', 'result'])
    >>> args.input == 'mypath'
    True
    >>> args.output == 'result'
    True
    """
    parser = argparse.ArgumentParser(
        description=
            "Looks for .png, .jpg or .jpeg files in the 'input' directory" \
             " and merges them together as the specified 'output' file."
    )
    parser.add_argument('--input', metavar='input',  help='input directory containing image files to merge', default='input')
    parser.add_argument('--output', metavar='output', help='output file name', default='merged')
    return parser.parse_args(args)


def gather_image_files(input_dir):
    """
    Generate an array of .png, .jpg and/or .jpeg files from the specified input directory
    :param input_dir: directory to read file names from
    :return: array of file names in sorted order
    >>> gather_image_files("input")
    ['step0.jpg', 'step1.jpg', 'step2.jpg', 'step3.jpg', 'step4.jpg', 'step5.jpg']
    """
    image_files = sorted(os.listdir(input_dir))
    for img in image_files:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)

    return image_files


def merge_and_focus_images(image_files, input_dir):
    """
    Read image files into memory and use to generate a singular focused image object represented as a numpy array
    :param image_files: array of image file names to read
    :param input_dir: directory containing image files
    :return: an X x Y numpy array suitable for writing out to a file via OpenVC
    >>> merge_and_focus_images(['step0.jpg', 'step1.jpg'], 'input').shape == (1141, 1521, 3)
    True
    """
    focusimages = []
    for img in image_files:
        print "Reading in file {}".format(img)
        focusimages.append(cv2.imread("{}/{}".format(input_dir, img)))

    return FocusStack.focus_stack(focusimages)


def main():
    args = parse_args(sys.argv[1:])
    merged_and_focused_image = merge_and_focus_images(gather_image_files(args.input), args.input)
    cv2.imwrite(args.output + ".png", merged_and_focused_image)
    print "That's All Folks!"

if __name__ == "__main__":
    main()

