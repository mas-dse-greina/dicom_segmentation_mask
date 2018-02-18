#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Processes a directory of DICOM files and creates
both the 2D slices with their associated image masks.

usage: process_dicom_to_hdf5.py [-h] [--print_random_image]
                                [--data_directory DATA_DIRECTORY]
                                [--output_filename OUTPUT_FILENAME]

optional arguments:
  -h, --help            show this help message and exit
  --print_random_image  unit test: print random image and mask
  --data_directory DATA_DIRECTORY
                        base directory for data
  --output_filename OUTPUT_FILENAME
                        Name of the hdf5 to create for data

Unit test:
1. To print a random DICOM image and its associated mask:
	`python process_dicom_to_hdf5.py --print_random_image`


"""

import argparse
import shutil
import atexit
import os
from tqdm import trange
from configparser import ConfigParser

#### Read from the configuration file config.ini ####
config = ConfigParser()
config.read("config.ini")

DICOMS_DIR_BASE = config.get("local", "DATA_DIR_BASE") + r"dicoms/"         # Top-level directory for dicoms
CONTOURS_DIR_BASE = config.get("local", "DATA_DIR_BASE") + r"contourfiles/" # Top-level directory for contour files

CONTOURS_SUB_DIR = config.get("local", "CONTOURS_SUB_DIR")
LINK_FILE_NAME = config.get("local", "LINK_FILE_NAME")

class readable_dir(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		prospective_dir=values
		if not os.path.isdir(prospective_dir):
			raise argparse.ArgumentTypeError("{0} is not a valid path".format(prospective_dir))
		if os.access(prospective_dir, os.R_OK):
			setattr(namespace,self.dest,prospective_dir)
		else:
			raise argparse.ArgumentTypeError("{0} is not a readable directory".format(prospective_dir))


parser = argparse.ArgumentParser(description="Process the DICOM files and masks")
parser.add_argument("--print_random_image", action="store_true", default=False,
					help="unit test: print random image and mask")

parser.add_argument("--data_directory", action=readable_dir,
					default=config.get("local", "DATA_DIR_BASE"),
					help="base directory for data")

parser.add_argument("--output_filename", default=config.get("local", "HDF5_FILENAME"), help="Name of the hdf5 to create for data")
args = parser.parse_args()

DATA_DIR_BASE = args.data_directory
HDF5_FILENAME = args.output_filename

from parsing import parse_contour_file, parse_dicom_file, poly_to_mask
import glob
import pandas as pd
import numpy as np
import h5py

def getFiles(dfLink, idx):
	'''
	Get the list of DICOM files and contour files associated with this patient idx
	'''
	dicomDirname = DICOMS_DIR_BASE + dfLink["patient_id"].iloc[idx] + "/"                    # DICOM Directory name
	contourDirname = CONTOURS_DIR_BASE + dfLink["original_id"].iloc[idx] + CONTOURS_SUB_DIR  # Contour Directory name

	dicomFiles   = glob.glob(dicomDirname + "*.dcm")   # Get the DICOM files within this directory
	contourFiles = glob.glob(contourDirname + "*.txt") # Get the contour files within this directory

	return dicomFiles, contourFiles

import fnmatch  # Filter file names
import re  # Import regular expressions to extract slice #
import os

def get_matching_slice(contourFilename, dicomFiles):
	'''
	Associates the DICOM slice with the contour file.
	The assumption here is that the last 4 digits in the contour filename are the
	slice number from the DICOM. Verified this in the EDA python notebook
	by plotting the masks over the DICOM images.
	'''
	sliceName = os.path.basename(os.path.splitext(contourFilename)[0])  # The mask name

	# Use regex to find the pattern xxxx-yyyy in the file name. Extract the yyyy and convert to int.
	# This will be the slice number
	sliceIdx = int(re.findall(r'\d{4}-\d{4}', sliceName)[0][-4:])

	dicomFilename = fnmatch.filter(dicomFiles, "*{}.dcm".format(sliceIdx))[0] # Find associated dicom image for slice

	return dicomFilename


def getMask(contourFilename, imgWidth, imgHeight, maskThreshold=0.5):
	'''
	contourFilename = absolute path to the contour file
	imgWidth = desired width
	imgHeight = desired height
	maskThreshold = [0,1] Sanity check. If mask is larger than this percentage, then contour might be bad.
	TODO: Add a Hough ellipse detector to validate one and only one round mask.
	'''

	# Extract the polygon contour points
	polygonPoints = parse_contour_file(contourFilename)
	# Fill the polygon
	imgMask = poly_to_mask(polygonPoints, imgWidth, imgHeight)

	# Sanity check - What if the polygon is malformed? Let's check to make sure the mask isn't
	# more than a certain percentage of the entire image
	percentMask = imgMask.sum() / float(imgMask.shape[0] * imgMask.shape[1])
	if percentMask > maskThreshold:
		print("The mask is more than {} of the image. Please check if polygon is correct. {} {}".format(maskThreshold,
																									   dicomFilename,                                                                                     sliceName))
	return imgMask


def get_imgs_and_masks(contourFilename, dicomFiles):
	'''
	Returns the image and mask for a given contour filename.
	'''
	dicomFilename = get_matching_slice(contourFilename, dicomFiles)

	imgDict = parse_dicom_file(dicomFilename)

	# Get the original DICOM image
	img = imgDict["pixel_data"]
	(imgHeight, imgWidth) = img.shape  # Get the image shape

	# Test:  The width and height should be the same that is listed in the DICOM header
	if (imgDict["dicom"].Rows!= imgHeight) | (imgDict["dicom"].Columns != imgWidth):
		print("Image size does not correspond to header {} {}".format(contourFilename, dicomFilename))

	# Get the associated mask for the image
	imgMask = getMask(contourFilename, imgWidth, imgHeight, maskThreshold=0.5)

	return img, imgMask, imgDict

import matplotlib.pyplot as plt

def plot_imgs_and_masks(img, img_mask, imgDict):

	plt.figure(figsize=(15,15))
	plt.subplot(1,2,1)
	plt.imshow(img, cmap="bone");
	plt.title("Original MRI of heart\nPatient #{}".format(imgDict["dicom"].PatientID));

	plt.subplot(1,2,2)
	plt.imshow(img, cmap="bone");
	plt.imshow(img_mask, alpha=0.3);

	plt.title("With inner diameter mask (yellow)");

	print("Pixel dimensions are {:.3f} x {:.3f} mm".format(imgDict["dicom"].PixelSpacing[0],
														   imgDict["dicom"].PixelSpacing[1]))
	print("Slice thickness is {:.3f} mm".format(imgDict["dicom"].SliceThickness))

	plt.show()

def main():
	dfLink = pd.read_csv(DATA_DIR_BASE + LINK_FILE_NAME)

	if args.print_random_image:  # Test code by plotting random image and mask
		patientIdx = np.random.randint(0, dfLink.shape[0])
		dicomFiles, contourFiles = getFiles(dfLink, patientIdx)
		contourIdx = np.random.randint(0, np.shape(contourFiles)[0])
		img, imgMask, imgDict = get_imgs_and_masks(contourFiles[contourIdx], dicomFiles)
		plot_imgs_and_masks(img, imgMask, imgDict)

	else:  # Run the main code


		print("Reading from {} file".format(LINK_FILE_NAME))
		print("Base data directory is {}".format(DATA_DIR_BASE))

		bFirstTensor = True

		# The images and masks will be saved into a single HDF5 file.
		# HDF5 can handle unlimited file sizes and only loads
		# the data from the file needed. Very useful for a data loader
		# when the data is too large for the RAM.
		with h5py.File(HDF5_FILENAME, "w") as HDF5:

			tProgressBar = trange(dfLink.shape[0], desc='Patient', leave=True)
			for patientIdx in tProgressBar:

				dicomFiles, contourFiles = getFiles(dfLink, patientIdx)
				for contourIdx in trange(np.shape(contourFiles)[0]):

					tProgressBar.set_description("Patient {} (mask {})".format(patientIdx+1,
												 os.path.splitext(os.path.basename(contourFiles[contourIdx]))[0]))
					img, imgMask, imgDict = get_imgs_and_masks(contourFiles[contourIdx], dicomFiles)

					# We need to flatten the image and mask to put in a HDF5 dataframe
					imgTensor = img.ravel().reshape(1,-1)
					mskTensor = imgMask.ravel().reshape(1,-1)

					# HDF5 expects all of the tensors to be of equal size
					# So an error will be thrown if any of the masks or images is different size.
					# TODO: Check explicitly for different sized images/masks and handle gracefully.
					if bFirstTensor:

						bFirstTensor = False
						imgSet = HDF5.create_dataset("input", data=imgTensor, maxshape=[None, imgTensor.shape[1]])
						mskSet = HDF5.create_dataset("output", data=mskTensor, maxshape=[None, mskTensor.shape[1]])

					else:

						row = imgSet.shape[0] # Count current dataset rows
						imgSet.resize(row+1, axis=0) # Add new row
						imgSet[row, :] = imgTensor # Insert data into new row

						row = mskSet.shape[0] # Count current dataset rows
						mskSet.resize(row+1, axis=0) # Add new row
						mskSet[row, :] = mskTensor # Insert data into new row

			HDF5["input"].attrs["lshape"] = (img.shape[0], img.shape[1], 1)
			HDF5["output"].attrs["lshape"] = (imgMask.shape[0], imgMask.shape[1], 1)


		print("\n\nFinished.")

if __name__ == "__main__":
	main()
