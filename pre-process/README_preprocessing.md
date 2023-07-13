## Pre-process your data
**Step 1: nnU-Net segmentation** (lung, lobes, trachea, aorta)

For the trachea and aorta, use pre-trained nnU-Net Task_055_SegTHOR.

**Step 2: Setup your paths

Set all the paths to the data: 
- pre-process/paths.py

**Step 3: Register Exp to Insp images**

$ python registration.py 

**Step 4: Intensity normalization**

(i) Intensity normalization of each 3D volume, using aorta and trachea mean intensities for normalization

(ii) Re-sampling of all 3D images and corresponding labels to a fixed voxel size 0.5 using linear for images and nearest-neighbour interpolation for labels

(iii) Patchify images

$ python process_images_patches.py