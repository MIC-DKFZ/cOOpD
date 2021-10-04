import numbers
import numpy as np
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
              and 0 < max_patches < 1):
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