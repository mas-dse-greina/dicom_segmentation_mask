# dicom_segmentation_mask
A pipeline to generate HDF5 datsets for DICOM image and a correpsonding segmentation mask.

Files:

1. *Dicom segmentation EDA.ipynb* - This is the initial exploratory analysis to test the modules within the pipeline and sanity check that correct images and masks were pulled from the DICOM directory.

2. *python process_dicom_to_hdf5.py* - This is the standalone code for processing the DICOM and masks into an HDF5 file. The HDF5 method is preferable for deep learning frameworks because it can hold the data in a single, extensible file format that can be accessed out of system memory. The HDF5 can be arbitrarily large and the datasets within the file can be randomly sliced as if they were numpy arrays. 

```
usage: process_dicom_to_hdf5.py [-h] [--print_random_image]
                                [--data_directory DATA_DIRECTORY]
                                [--output_filename OUTPUT_FILENAME]

Process the DICOM files and masks

optional arguments:
  -h, --help            show this help message and exit
  --print_random_image  unit test: print random image and mask
  --data_directory DATA_DIRECTORY
                        base directory for data
  --output_filename OUTPUT_FILENAME
                        Name of the hdf5 to create for data

```
  

3. *hdf5_batch_loader.py* - This is the standalone code for accessing the saved HDF5 data file (please run python_process_dicom_to_hdf5.py first so that you'll have a .h5 file to work with). The code loads in a random batch of images and masks. I've also added image augmentation which randomly flips and/or rotates the images and masks. This is essential in training models with limited training samples. A sample image from the HDF5 looks like this:

![image with mask](https://github.com/mas-dse-greina/dicom_segmentation_mask/blob/master/test_figure.png)

Note the original image has been randomly flipped and rotated.


```
usage: process_dicom_to_hdf5.py [-h] [--print_random_image]
                                [--data_directory DATA_DIRECTORY]
                                [--output_filename OUTPUT_FILENAME]

Process the DICOM files and masks

optional arguments:
  -h, --help            show this help message and exit
  --print_random_image  unit test: print random image and mask
  --data_directory DATA_DIRECTORY
                        base directory for data
  --output_filename OUTPUT_FILENAME
                        Name of the hdf5 to create for data

```

Note that there is a config.ini file which can be modified to point to custom directories.

## After building the pipeline, please answer the following questions:

### How did you verify that you are parsing the contours correctly?

I've plotted the original image next to the combined image/mask to give a visual sanity check of the pipeline. The extracted masks look correct, but we note that subset 2 (patient SCD0000301) has a slightly different image resolution (1.289 mm per pixel) than the other 5 (which are 1.376). They all seem to be 256 x 256 image. I'd suggest using Simple ITK or OpenCV to normalize the images to the same pixel resolution and then crop as needed to get the 256 x 256 pixels. Obviously the pixel masks for those will also need to be modified to compensate for the change in resolution.

Further, I added a test in the getMask() function to test whether the mask is taking up too much of the image. This might not be needed, but I thought there might be a corner case where the polygon points are corrupted and Pillow would end up making a mask that was far too large. If I had more time, I'd consider adding a Hough transform to detect ellipses in the image and verify that there is just one ellipse.

Note that the 4-digit number in the contour filename seems to point to the slice from the DICOM. Also, there are slices that do not have associated masks. In the final output, I can simply fill these with empty masks and store the array as a sparse matrix to save on disk size.

### What changes did you make to the code, if any, in order to integrate it into our production code base?

I added the entire dicom dictionary to return because I thought it important to check the resolution of the images. Ideally, the model should be working from images of uniform pixels/mm.

In the process_dicom_to_hdf5.py script, I create an extensible end-to-end script that iteratively goes through the data directory, extracts the DICOM image and mask, and saves these to an HDF5 file. We can use the HDF5 file for training our convolutional neural network by using the HDF5 batch loader.

### Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?

I added a data augmentator which randomly rotated and flipped the tensors. It might be useful to add more data augmentation (random crops, zooms, contrast adjustments) in future iterations.

### How do you/did you verify that the pipeline was working correctly?

There's a simple unit test that can be accessed from the command line switch (--print_random_image). This prints out 3 random images from the batch. One of these images is saved to disk. I could create a function that randomly saved images to disk to keep track of the algorithm.

### Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?

I'd like to use SimpleITK to normalize the pixels/voxels to a uniform size. I'd also like to add the Hough transform to test for the presence of just a single elliptical mask in case the mask polygon points are corrupted.



