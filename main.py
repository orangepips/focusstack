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
    parser.add_argument('--debug', dest='debug', help='Generate debug output and show on screen', action="store_true")
    parser.set_defaults(debug=False)
    return parser.parse_args(args)


def gather_image_file_names(input_dir):
    """
    Generate an array of .png, .jpg and/or .jpeg files from the specified input directory
    :param input_dir: directory to read file names from
    :return: array of image file names in sorted order
    >>> gather_image_file_names("input")
    ['step0.jpg', 'step1.jpg', 'step2.jpg', 'step3.jpg', 'step4.jpg', 'step5.jpg']
    """
    image_file_names = sorted(os.listdir(input_dir))
    for img in image_file_names:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_file_names.remove(img)

    return image_file_names


def read_image_files(image_file_names, input_dir):
    """
    Read image files into memory and use to generate a singular focused image object represented as a numpy array
    :param image_file_names: array of image file names to read
    :param input_dir: directory containing image files
    :return: array of numpy arrays representing each image in same order as passed image_file_names
    an X x Y numpy array suitable for writing out to a file via OpenVC
    >>> read_image_files(['step0.jpg', 'step1.jpg'], 'input').shape == (1141, 1521, 3)
    True
    """
    image_files = []
    for image_file_name in image_file_names:
        print "Reading in file {}".format(image_file_name)
        image_files.append(cv2.imread("{}/{}".format(input_dir, image_file_name)))
    return image_files


def main():
    args = parse_args(sys.argv[1:])
    image_file_names = gather_image_file_names(args.input)

    image_files = read_image_files(image_file_names, args.input)

    output_file_name = args.output + ".png"
    fs = FocusStack.FocusStack(image_files, debug=args.debug)
    fs.focus()
    cv2.imwrite(output_file_name, fs.focused_image)
    print "Done. File '{}' written.".format(output_file_name)


if __name__ == "__main__":
    main()

