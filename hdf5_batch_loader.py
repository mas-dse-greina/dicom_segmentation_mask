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
Loads an HDF5 file that was created by process_dicom_to_hdf5.py
Creates batches that can be loaded into a Keras fit_generator method.
Handles simple data augmentation (flip and 90/180/270 rotations). These can
be extended to 3D tensors if needed.
"""

import argparse
from configparser import ConfigParser
import h5py
import numpy as np
from tqdm import trange


#### Read from the configuration file config.ini ####
config = ConfigParser()
config.read("config.ini")

parser = argparse.ArgumentParser(description="Load batches of images and masks from the HDF5 datafile")

parser.add_argument("--input_filename", default=config.get("local", "HDF5_FILENAME"), help="Name of the hdf5 to load for data")
parser.add_argument("--batchsize", type=int, default=config.get("local", "BATCH_SIZE"), help="batch size for data loader")
parser.add_argument("--epochs", type=int, default=config.get("local", "NUM_EPOCHS"), help="number of epochs to train")
parser.add_argument("--print_random_image", action="store_true", default=False,
					help="unit test: print random image and mask")
parser.add_argument("--print_batch_indices", action="store_true", default=False,
					help="unit test: print the indices for each batch")
args = parser.parse_args()

HDF5_FILENAME = args.input_filename
BATCH_SIZE = args.batchsize
NUM_EPOCHS = args.epochs

def img_flip(img, msk):
	'''
	Performs a random flip on the tensor. Should work for any N-D matrix.
	'''
	shape = img.shape
	# This will flip along n-1 axes. (If we flipped all n axes then we'd get the same result every time)
	ax = np.random.choice(len(shape)-1,len(shape)-2, replace=False) + 1 # Choose randomly which axes to flip
	for i in ax:
		img = np.flip(img, i) # Randomly flip along all but one axis
		msk = np.flip(msk, i)

	return img, msk

def img_rotate(img, msk):
	'''
	Perform a random rotation on the tensor - Should work for any N-D matrix
	'''
	shape = img.shape
	# This will flip along n-1 axes. (If we flipped all n axes then we'd get the same result every time)
	ax = np.random.choice(len(shape)-1,2, replace=False) # Choose randomly which axes to flip
	img = np.flip(img.swapaxes(ax[0], ax[1]), ax[0]) # Random +90 or -90 rotation
	msk = np.flip(msk.swapaxes(ax[0], ax[1]), ax[0]) # Random +90 or -90 rotation

	return img, msk

def augment_data(imgs, msks):
	'''
	Performs random flips, rotations, and other operations on the image tensors.
	'''

	imgs_length = imgs.shape[0]

	for idx in range(imgs_length):
		img = imgs[idx, :]
		msk = msks[idx, :]

		if (np.random.rand() > 0.5):
			img, msk = img_rotate(img, msk)

		if (np.random.rand() > 0.5):
			img, msk = img_flip(img, msk)

		imgs[idx,:] = img
		msks[idx, :] = msk

	return imgs, msks

def get_batch(hdf5_file, idx, batch_size=16):

	img_shape = tuple([batch_size] + list(hdf5_file['input'].attrs['lshape']))
	msk_shape = tuple([batch_size] + list(hdf5_file['output'].attrs['lshape']))

	imgs = hdf5_file["input"][idx,:]
	imgs = imgs.reshape(img_shape)

	msks = hdf5_file["output"][idx,:]
	msks = msks.reshape(msk_shape)

	return imgs, msks


def get_random_batch(hdf5_file, maxIdx, batch_size=16):

	# Randomly shuffle indices. Take first batch_size. Sort.
	# This will give us a completely random set of indices for each batch
	indicies = np.arange(0,maxIdx)
	np.random.shuffle(indicies) # Shuffle the indicies
	random_idx = np.sort(indicies[0:batch_size])

	return get_batch(hdf5_file, random_idx, batch_size)


import matplotlib.pyplot as plt
def plot_imgs_and_masks(img_original, img, img_mask):
	'''
	Plot the images and mask overlay
	'''

	plt.figure(figsize=(15,15))
	plt.subplot(1,2,1)
	plt.imshow(img_original, cmap="bone");
	plt.title("Original MRI of heart");

	plt.subplot(1,2,2)
	plt.imshow(img, cmap="bone");
	plt.imshow(img_mask, alpha=0.3);

	plt.title("With inner diameter mask (yellow)\nRandom rotation/flip");
	plt.savefig("test_figure.png", dpi=300)


def main() :
	with h5py.File(HDF5_FILENAME) as HDF5:

		numImgs = HDF5["input"].shape[0]
		numMsks = HDF5["output"].shape[0]

		print("Number of images: {}".format(numImgs))
		print("Number of masks: {}".format(numMsks))
		print("Image size = {} pixels".format(tuple(HDF5["input"].attrs["lshape"])))
		print("Mask size = {} pixels".format(tuple(HDF5["output"].attrs["lshape"])))
		print("Batch size = {}".format(BATCH_SIZE))
		print("Number of epochs = {}\n\n".format(NUM_EPOCHS))

		if args.print_random_image:

			imgs_original, msks = get_random_batch(HDF5, numImgs, BATCH_SIZE)

			# Create random rotations and flips on image and masks
			imgs, msks = augment_data(imgs_original.copy(), msks)

			for i in range(3):
				idx = np.random.randint(BATCH_SIZE)
				plot_imgs_and_masks(np.squeeze(imgs_original[idx,:,:,:]),
									np.squeeze(imgs[idx,:,:,:]),
									np.squeeze(msks[idx,:,:,:]))

			print("Plotting 3 random images and masks.")
			plt.show()

		else:  # Go through for some epochs

			indicies = np.arange(0,numImgs)

			tProgressBar = trange(NUM_EPOCHS, desc='Epoch', leave=True)
			for epoch in range(NUM_EPOCHS):

				if args.print_batch_indices:
					print("********** EPOCH {} **************".format(epoch+1))

				np.random.shuffle(indicies) # Shuffle the indicies

				for batchIdx in range(0, numImgs, BATCH_SIZE):

					idx = np.sort(indicies[batchIdx:(batchIdx+BATCH_SIZE)])

					if args.print_batch_indices:
						print(idx)
					else:
						tProgressBar.set_description("Epoch {} (batch {})".format(epoch+1,
												 	idx))

					imgs_original, msks = get_batch(HDF5, idx, idx.shape[0])

					# Create random rotations and flips on image and masks
					imgs, msks = augment_data(imgs_original.copy(), msks)

					# Use the batch in to train the model


if __name__ == "__main__":
	main()
