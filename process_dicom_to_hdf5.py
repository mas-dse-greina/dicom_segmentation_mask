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
"""

DATA_DIR_BASE = r"../data/final_data/"               # Top-level directory for data
DICOMS_DIR_BASE = DATA_DIR_BASE + r"dicoms/"         # Top-level directory for dicoms
CONTOURS_DIR_BASE = DATA_DIR_BASE + r"contourfiles/" # Top-level directory for contour files
CONTOURS_SUB_DIR = r"/i-contours/"                    # Sub-directory ("i-contours" or "o-contours")

import argparse

parser = argparse.ArgumentParser(description='Process the DICOM files and masks')
parser.add_argument("--print_random_image", action="store_true", help="unit test: print image")
args = parser.parse_args()



from parsing import parse_contour_file, parse_dicom_file, poly_to_mask
import glob
import pandas as pd
import numpy as np

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
    dfLink = pd.read_csv(DATA_DIR_BASE + "link.csv")

    if args.print_random_image:  # Test code by plotting random image and mask
        patientIdx = np.random.randint(0, dfLink.shape[0])
        dicomFiles, contourFiles = getFiles(dfLink, patientIdx)
        contourIdx = np.random.randint(0, np.shape(contourFiles)[0])
        img, imgMask, imgDict = get_imgs_and_masks(contourFiles[contourIdx], dicomFiles)
        plot_imgs_and_masks(img, imgMask, imgDict)
    else:
        print("Main code")

if __name__ == "__main__":
    main()
