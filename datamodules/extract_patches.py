import numbers
import numpy as np
import os
import json
from sklearn.utils import check_array, check_random_state
from numpy.lib.stride_tricks import as_strided
from itertools import product
import SimpleITK as sitk

def extract_patches(arr, patch_shape=8, extraction_step=1):
    """Extracts patches of any n-dimensional array in place using strides.
    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted
    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.
    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = [slice(None, None, st) for st in extraction_step]
    #indexing_strides_deprecated = arr[slices].strides
    indexing_strides = arr[tuple(slices)].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches
def _compute_n_patches_3d(i_x, i_y, i_z, p_x, p_y, p_z, max_patches=None):
    """Compute the number of patches that will be extracted in a volume.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    i_x : int
        The number of voxels in x dimension
    i_y : int
        The number of voxels in y dimension
    i_z : int
        The number of voxels in z dimension
    p_x : int
        The number of voxels in x dimension of a patch
    p_y : int
        The number of voxels in y dimension of a patch
    p_z : int
        The number of voxels in z dimension of a patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    """
    n_x = i_x - p_x + 1
    n_y = i_y - p_y + 1
    n_z = i_z - p_z + 1
    all_patches = n_x * n_y * n_z
    #max_patches = 50
    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Real))
              and 0 < max_patches <= 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def extract_patches_3d_fromMask(volume, mask, patch_size, max_patches = None, random_state=None):
    """Reshape a 3D volume into a collection of patches
    The resulting patches are allocated in a dedicated array.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    volume : array, shape = (volume_x, volume_y, volume_z)
        No channels are allowed
    patch_size : tuple of ints (patch_x, patch_y, patch_z)
        the dimensions of one patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    random_state : int or RandomState
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.
    Returns
    -------
    patches : array, shape = (n_patches, patch_x, patch_y, patch_z)
         The collection of patches extracted from the volume, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.
    Examples
    --------
    TBD
    """
    channel, v_x, v_y, v_z = volume.shape[:4]
    channel_patch, p_x, p_y, p_z = patch_size


    if p_x > v_x:
        raise ValueError("Height of the patch should be less than the height"
                         " of the volume.")

    if p_y > v_y:
        raise ValueError("Width of the patch should be less than the width"
                         " of the volume.")

    if p_z > v_z:
        raise ValueError("z of the patch should be less than the z"
                         " of the volume.")

    volume = check_array(volume, allow_nd=True)
    #print('shape',volume.shape)
    volume = np.moveaxis(volume, 0, -1)
    volume = volume.reshape((v_x, v_y, v_z, -1))
    #volume = volume.reshape((-1,v_x, v_y, v_z))
    n_colors = volume.shape[-1]

    extracted_patches = extract_patches(volume, patch_shape=(p_x, p_y, p_z, n_colors), extraction_step=1)
    #print(extracted_patches.shape)

    n_patches = _compute_n_patches_3d(v_x, v_y, v_z, p_x, p_y, p_z, max_patches)
    #print(n_patches)
    # check the indexes where mask is True

    M = np.array(np.where(mask[:,int(p_x / 2): int(v_x - p_x / 2),
                          int(p_y / 2):int(v_y - p_y / 2),
                          int(p_z / 2):int(v_z - p_z / 2)] == True)).T

    print('here4')

    #max_patches = 50
    if max_patches:
        #print('max_patches_if')
        rng = check_random_state(random_state)
        #print('len', len(M), n_patches)

        indx = rng.randint(len(M), size=n_patches)
        i_s = M[indx][:, 1] #0
        j_s = M[indx][:, 2] #1
        k_s = M[indx][:, 3] #2
        #print('extracted indexes', i_s, j_s, k_s, extracted_patches.shape)
        patches = extracted_patches[i_s, j_s, k_s, 0]



    else:
        patches = extracted_patches
    patches = patches.reshape(-1, p_x, p_y, p_z, n_colors)
    patches = np.transpose(patches, axes=[0,4, 1, 2, 3])
    #patches = patches.reshape(n_colors, p_x, p_y, p_z)



    # # remove the color dimension if useless
    # if patches.shape[-1] == 1:
    #     return patches.reshape((n_patches, p_x, p_y, p_z))
    # else:
    #     return patches

    print('patch_final', patches[0,0,0,0,0], patches.shape)

    sitk.WriteImage(sitk.GetImageFromArray(patches[0,0,:,:,:]), '/home/silvia/Downloads/try.nii.gz')


    return patches





## try to extract all patches in vol
def getN_allpatches_3d_fromMask(volume, mask, patch_size, overlap = (0,0,0,0)):
    """Reshape a 3D volume into a collection of patches
    The resulting patches are allocated in a dedicated array.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    volume : array, shape = (volume_x, volume_y, volume_z)
        No channels are allowed
    patch_size : tuple of ints (patch_x, patch_y, patch_z)
        the dimensions of one patch
    overlap : tuple of overlap in each channel

    Returns
    -------
    patches : list, shape = [patches_extracted]
         The collection of patches extracted from the volume
    Examples
    --------
    TBD
    """
    channel, v_x, v_y, v_z = volume.shape[:4]
    channel_patch, p_x, p_y, p_z = patch_size


    if p_x > v_x:
        raise ValueError("Height of the patch should be less than the height"
                         " of the volume.")

    if p_y > v_y:
        raise ValueError("Width of the patch should be less than the width"
                         " of the volume.")

    if p_z > v_z:
        raise ValueError("z of the patch should be less than the z"
                         " of the volume.")

    volume = check_array(volume, allow_nd=True)
    volume = np.moveaxis(volume, 0, -1)
    volume = volume.reshape((v_x, v_y, v_z, -1))
    n_colors = volume.shape[-1]

    # check the indexes where mask is True

    M = np.array(np.where(mask[:,int(p_x / 2): int(v_x - p_x / 2),
                          int(p_y / 2):int(v_y - p_y / 2),
                          int(p_z / 2):int(v_z - p_z / 2)] == True)).T

    import ndpatch

    volume_helper = volume.reshape(n_colors, v_x, v_y, v_z)


    #indices that comply with overlapping
    indices = get_patches_indx(array_shape=volume_helper.shape, patch_shape= patch_size, overlap= overlap, start=(M[0][0], M[0][1], M[0][2], M[0][3]))

    #indices where Mask is present
    M_helper = M.tolist()
    indices_set = set(map(tuple, indices))
    M_set = set(map(tuple, M_helper))


    print('whre to extract patches:', indices_set.intersection((M_set)))
    print('Number of patches available:', len(indices_set.intersection((M_set))))

    return len(indices_set.intersection((M_set)))
def extract_allpatches_3d_fromMask(volume, mask, patch_size, overlap = (0,0,0,0)):
    """Reshape a 3D volume into a collection of patches
    The resulting patches are allocated in a dedicated array.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    volume : array, shape = (volume_x, volume_y, volume_z)
        No channels are allowed
    patch_size : tuple of ints (patch_x, patch_y, patch_z)
        the dimensions of one patch
    overlap : tuple of overlap in each channel

    Returns
    -------
    patches : list, shape = [patches_extracted]
         The collection of patches extracted from the volume
    Examples
    --------
    TBD
    """
    channel, v_x, v_y, v_z = volume.shape[:4]
    channel_patch, p_x, p_y, p_z = patch_size


    if p_x > v_x:
        raise ValueError("Height of the patch should be less than the height"
                         " of the volume.")

    if p_y > v_y:
        raise ValueError("Width of the patch should be less than the width"
                         " of the volume.")

    if p_z > v_z:
        raise ValueError("z of the patch should be less than the z"
                         " of the volume.")

    volume = check_array(volume, allow_nd=True)
    volume = np.moveaxis(volume, 0, -1)
    volume = volume.reshape((v_x, v_y, v_z, -1))
    n_colors = volume.shape[-1]

    extracted_patches = extract_patches(volume, patch_shape=(p_x, p_y, p_z, n_colors), extraction_step=1)
    #print(extracted_patches.shape)

    # check the indexes where mask is True

    M = np.array(np.where(mask[:,int(p_x / 2): int(v_x - p_x / 2),
                          int(p_y / 2):int(v_y - p_y / 2),
                          int(p_z / 2):int(v_z - p_z / 2)] == True)).T

    print('here4')
    print((M[0][0], M[0][1], M[0][2], M[0][3]))
    print((M[-1][0], M[-1][1], M[-1][2], M[-1][3]))
    import ndpatch

    volume_helper = volume.reshape(n_colors, v_x, v_y, v_z)


    #indices that comply with overlapping
    indices = get_patches_indx(array_shape=volume_helper.shape, mask= mask, patch_shape= patch_size, overlap= overlap, start=(M[0][0], M[0][1], M[0][2], M[0][3]))

    #indices where Mask is present
    M_helper = M.tolist()
    indices_set = set(map(tuple, indices))
    M_set = set(map(tuple, M_helper))


    print('whre to extract patches:', indices_set.intersection((M_set)))
    print('Number of patches available:', len(indices_set.intersection((M_set))))

    patches = []
    index_list = []
    for indx in indices_set.intersection((M_set)):
        print(indx)

        patch_single = extracted_patches[indx[1], indx[2], indx[3], 0]
        patch_single = patch_single.reshape(n_colors, p_x, p_y, p_z)
        # patch_single = np.transpose(patch_single, axes=[0,4, 1, 2, 3])
        patches.append(patch_single)
        index_list.append(indx)


    print('patch_final', patches[0][0,0,0,0])



    return patches, index_list




def new_fromMask(volume, mask, patch_size, overlap = (0,0,0,0)):
    """Reshape a 3D volume into a collection of patches
    The resulting patches are allocated in a dedicated array.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    volume : array, shape = (volume_x, volume_y, volume_z)
        No channels are allowed
    patch_size : tuple of ints (patch_x, patch_y, patch_z)
        the dimensions of one patch
    overlap : tuple of overlap in each channel

    Returns
    -------
    patches : list, shape = [patches_extracted]
         The collection of patches extracted from the volume
    Examples
    --------
    TBD
    """
    channel, v_x, v_y, v_z = volume.shape[:4]
    channel_patch, p_x, p_y, p_z = patch_size


    if p_x > v_x:
        raise ValueError("Height of the patch should be less than the height"
                         " of the volume.")

    if p_y > v_y:
        raise ValueError("Width of the patch should be less than the width"
                         " of the volume.")

    if p_z > v_z:
        raise ValueError("z of the patch should be less than the z"
                         " of the volume.")

    volume = check_array(volume, allow_nd=True)
    volume = np.moveaxis(volume, 0, -1)
    volume = volume.reshape((v_x, v_y, v_z, -1))
    n_colors = volume.shape[-1]

    mask = check_array(mask, allow_nd=True)
    print(np.where(mask>0))
    mask_transf = np.moveaxis(mask, 0, -1)
    mask_transf = mask_transf.reshape((v_x, v_y, v_z, -1))

    extracted_patches = extract_patches(volume, patch_shape=(p_x, p_y, p_z, n_colors), extraction_step=1)
    extracted_patches_label = extract_patches(mask_transf, patch_shape=(p_x, p_y, p_z, n_colors), extraction_step=1)

    #print(extracted_patches.shape)

    # check the indexes where mask is True

    M = np.array(np.where(mask[:,int(p_x / 2): int(v_x - p_x / 2),
                          int(p_y / 2):int(v_y - p_y / 2),
                          int(p_z / 2):int(v_z - p_z / 2)] == True)).T

    print('here4')
    print((M[0][0], M[0][1], M[0][2], M[0][3]))
    print((M[-1][0], M[-1][1], M[-1][2], M[-1][3]))
    import ndpatch

    volume_helper = volume.reshape(n_colors, v_x, v_y, v_z)


    #indices that comply with overlapping
    indices = get_patches_indx(array_shape=volume_helper.shape, mask= mask, patch_shape= patch_size, overlap= overlap, start=(M[0][0], M[0][1], M[0][2], M[0][3]))

    #indices where Mask is present

    patches = []
    all = extracted_patches_label
    index_list = []
    for indx in indices:
        print(indx)
        #print(extracted_patches_label.shape)
        if indx[1]+patch_size[1]< extracted_patches_label.shape[0] and indx[2]+patch_size[2]< extracted_patches_label.shape[1] and indx[3]+patch_size[3]< extracted_patches_label.shape[2]:

            patch_label_single = extracted_patches_label[indx[1], indx[2], indx[3], 0]
            patch_label_single = patch_label_single.reshape(n_colors, p_x, p_y, p_z)
            uniques, counts = np.unique(patch_label_single, return_counts=True)
            percentages = dict(zip(uniques, counts * 100 / len(patch_label_single)))

            print((len(uniques), uniques))
            if ((len(uniques)==1 and uniques == 1) or (len(uniques)==2 and percentages[1]/(percentages[1]+percentages[0]) > 0.3)):

                # patch_single = np.transpose(patch_single, axes=[0,4, 1, 2, 3])
                patch_single = extracted_patches[indx[1], indx[2], indx[3], 0]
                patch_single = patch_single.reshape(n_colors, p_x, p_y, p_z)
                patches.append(patch_single)
                index_list.append(indx)


    print('patch_final', patches[0][0,0,0,0])



    return patches, index_list

def get_patches_indx(array_shape, mask, patch_shape, overlap, start):
    array_shape = np.asarray(mask.shape)
    patch_shape = np.asarray(patch_shape)
    overlap = np.asarray(overlap)
    stop = start + array_shape
    step = patch_shape - overlap
    slices = [slice(_start, _stop, _step) for _start, _stop, _step in zip(start, stop, step)]
    all = np.array(np.mgrid[slices].reshape(len(slices), -1).T, dtype=np.int).tolist()
    return all





    # array_shape = np.asarray(array_shape)
    # patch_shape = np.asarray(patch_shape)
    # overlap = np.asarray(overlap)
    # stop = start + array_shape
    # step = patch_shape - overlap
    # slices = [slice(_start, _stop, _step) for _start, _stop, _step in zip(start, stop, step)]

    #return np.array(np.mgrid[slices].reshape(len(slices), -1).T, dtype=np.int).tolist()

if __name__ == "__main__":

    #trial
    numpy_array = np.load('/home/silvia/Documents/CRADL/pre-processed/all/028969756.npz', mmap_mode="r")
    insp = numpy_array['insp']
    label = numpy_array['label']
    patches, index = new_fromMask(insp, label, (1, 50, 50, 50), overlap=(0, 0, 0, 0))
    sitk.WriteImage(sitk.GetImageFromArray(patches[0][0, :, :, :]), '/home/silvia/Downloads/try_0.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(patches[1][0, :, :, :]), '/home/silvia/Downloads/try_1.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(patches[2][0, :, :, :]), '/home/silvia/Downloads/try_2.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(patches[3][0, :, :, :]), '/home/silvia/Downloads/try_3.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(patches[4][0, :, :, :]), '/home/silvia/Downloads/try_4.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(patches[5][0, :, :, :]), '/home/silvia/Downloads/try_5.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(patches[6][0, :, :, :]), '/home/silvia/Downloads/try_6.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(patches[7][0, :, :, :]), '/home/silvia/Downloads/try_7.nii.gz')




###
    pattern = '*.npz'
    base_dir = '/home/silvia/Documents/CRADL/pre-processed/no_resample'
    with open(os.path.join(base_dir, 'filenames_for_eval.txt'), "r") as fp:
        filenames = json.load(fp)
    directories_read = [base_dir + '/' + sub + pattern.replace('*', '') for sub in filenames]

    for dir in directories_read:
        print(dir)
        numpy_array = np.load(dir, mmap_mode="r")
        insp = numpy_array['insp']
        label = numpy_array['label']

        patches, index = extract_allpatches_3d_fromMask(insp, label, (1, 50, 50, 50), overlap=(0,0,0,0))

        for i in range(0, len(patches)):
            patch = patches[i]
            np.savez_compressed(os.path.join(base_dir, 'patches_insp_overlap0', os.path.basename(dir).split('.')[0] +
                                             '_' + str(i) + '_' +str(index[i][1])
                                             + '_' +str(index[i][2]) + '_' +str(index[i][3])), patch)

            sitk.WriteImage(sitk.GetImageFromArray(patch[0, :, :, :]), '/home/silvia/Downloads/try_1.nii.gz')
