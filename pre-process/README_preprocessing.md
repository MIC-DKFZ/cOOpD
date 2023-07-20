## Pre-process your data
**Step 1: nnU-Net segmentation** (lung, lobes, trachea, aorta)

For the trachea and aorta, use pre-trained nnU-Net Task_055_SegTHOR.

Please also cite the following work if you use this pipeline for training:
```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). 
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.
```

**Step 2: Setup your paths**

Set all the paths to the data: 
- pre-process/paths.py

**Step 3: Save metadata from dicom files to pickle**

```
$ python save_metadata_pickle.py 
```


**Step 3: Register Exp to Insp images**
```
$ python registration.py 
```

**Step 4: Intensity normalization, Resampling & Patchify**

(i) Intensity normalization of each 3D volume, using aorta and trachea mean intensities for normalization

(ii) Re-sampling of all 3D images and corresponding labels to a fixed voxel size 0.5 using linear for images and nearest-neighbour interpolation for labels

(iii) Patchify images

```
$ python process_images_patches.py
```

**Step 5: Save info of all patches**

(i) Set your paths in /datamodules/analyse_patchdataset.py

(ii) Save file with info

```
$ python /datamodules/analyse_patchdataset.py
```

**Step 6: Select patients for training, validation and test sets**

As the generative model is only fit on healthy patches (%Emph<1%) from class 0 (healthy), this has to be set in the patients for the task.

(i) Set your paths in /datamodules/select_patients.py

(ii) Save files with list of patches per task

```
$ python /datamodules/select_patients.py
```


Now you're ready for training :)