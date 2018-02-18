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
usage: hdf5_batch_loader.py [-h] [--input_filename INPUT_FILENAME]
                            [--batchsize BATCHSIZE] [--epochs EPOCHS]
                            [--print_random_image] [--print_batch_indices]

Load batches of images and masks from the HDF5 datafile

optional arguments:
  -h, --help            show this help message and exit
  --input_filename INPUT_FILENAME
                        Name of the hdf5 to load for data
  --batchsize BATCHSIZE
                        batch size for data loader
  --epochs EPOCHS       number of epochs to train
  --print_random_image  unit test: print random image and mask
  --print_batch_indices
                        unit test: print the indices for each batch

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

For the random batch iteration, I've added a unit test with the command line switch (--print_batch_indices) which will print out the indices per batch for each epoch of training.

```
$ python hdf5_batch_loader.py --print_batch_indices
Number of images: 96
Number of masks: 96
Image size = (256, 256, 1) pixels
Mask size = (256, 256, 1) pixels
Batch size = 16
Number of epochs = 3


Epoch:   0%|                                                                          | 0/3 [00:00<?, ?it/s]********** EPOCH 1 **************
[ 8 10 23 24 26 31 34 45 48 51 58 61 69 70 78 80]
[ 1  3 15 19 25 33 38 56 64 68 71 76 81 82 88 90]
[ 6 20 21 27 36 37 40 41 44 46 49 53 66 74 77 94]
[ 5  7 13 16 17 29 32 35 43 52 54 65 79 87 89 93]
[ 0  4 12 22 42 55 57 59 60 62 63 67 75 83 84 95]
[ 2  9 11 14 18 28 30 39 47 50 72 73 85 86 91 92]
********** EPOCH 2 **************
[ 4  8 16 17 22 25 44 47 54 56 68 76 77 84 86 90]
[ 2 13 20 23 33 40 41 46 58 62 63 64 78 80 81 88]
[ 3  5 14 19 28 29 30 32 37 45 48 53 71 73 79 92]
[ 7  9 11 15 27 34 50 52 57 59 61 70 74 82 83 89]
[ 0 12 21 26 35 36 38 39 42 51 65 67 75 85 87 94]
[ 1  6 10 18 24 31 43 49 55 60 66 69 72 91 93 95]
********** EPOCH 3 **************
[16 20 28 29 42 43 44 57 59 63 66 67 69 71 92 94]
[ 0  7 12 18 19 22 24 27 35 45 48 49 76 80 85 93]
[ 1  2 11 13 21 23 37 46 52 53 54 56 62 70 73 87]
[ 4  6  8 17 34 36 55 58 61 74 78 81 84 86 89 91]
[ 5  9 10 14 15 25 26 30 31 32 33 40 41 65 79 82]
[ 3 38 39 47 50 51 60 64 68 72 75 77 83 88 90 95]

```

### Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?

I'd like to use SimpleITK to normalize the pixels/voxels to a uniform size. I'd also like to add the Hough transform to test for the presence of just a single elliptical mask in case the mask polygon points are corrupted.

To use in Keras' fit_generator I just need to call the batch loader through an iterator. So an infinite loop that just keeps yielding a batch.




