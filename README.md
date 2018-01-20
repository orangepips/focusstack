# focusstack

Simple Focus Stacking in Python

Uses Python 2.7 and [OpenCV2](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html).

This project implements a simple [focus stacking](https://en.wikipedia.org/wiki/Focus_stacking) algorithm in Python.

Focus stacking is useful when your depth of field is shallower than
all the objects you wish to capture (in macro photography, this is
very common).  Instead, you take a series of pictures where you focus
progressively throughout the range of images, so that you have a series
of images that, in aggregate, have everything in focus.  Focus stacking,
then, merges these images by taking the best focused region from all
of them to build up the final composite image.

The focus stacking logic is contained in [`FocusStack.py`](FocusStack.py).
There is a sample driver program in [`main.py`](main.py).  By default,
it assumes that there is a subdirectory called "input" containing source
images and generates an output called "merged.png".

You can specify alternate an alternate input directory and/or output file
name by using the command line parameters `--input` and `--output`:

`$ python main.py --input my_input_directory --output my_merged_file`

Input files are expected in either `.png`, `.jpg` or `.jpeg` format.

The output file is always in `.png` format.

The the "input" directory in the repository includes sample images
to experiment with the code without having to shoot your own set of images.

This project is Copyright 2015 Charles McGuinness, and is released under the
Apache 2.0 license (see license file for precise details). 