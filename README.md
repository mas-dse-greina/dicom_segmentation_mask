# dicom_segmentation_mask
A pipeline to generate HDF5 datsets for DICOM image and a correpsonding segmentation mask.

1. Dicom segmentation EDA - This is the initial exploratory analysis to test the modules within the pipeline and sanity check that correct images and masks were pulled from the DICOM directory.

2. python process_dicom_to_hdf5.py - This is the standalone code for processing the DICOM and masks into an HDF5 file.

Unit test:
  `python process_dicom_to_hdf5.py --print_random_image` will print a random DICOM image and mask
  

