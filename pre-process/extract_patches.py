import numbers
import numpy as np
from sklearn.utils import check_array, check_random_state
from numpy.lib.stride_tricks import as_strided
from itertools import product
import SimpleITK as sitk


def _compute_n_patches(i_h, i_w, p_h, p_w, max_patches=None):
    """Compute the number of patches that will be extracted in an image.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image with
    p_h : int
        The height of a patch
    p_w : int
        The width of a patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    """
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    all_patches = n_h * n_w

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
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches


def extract_patches_2d(image, patch_size, max_patches=None, random_state=None):
    """Reshape a 2D image into a collection of patches
    The resulting patches are allocated in a dedicated array.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    image : array, shape = (image_height, image_width) or
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.
    patch_size : tuple of ints (patch_height, patch_width)
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
    patches : array, shape = (n_patches, patch_height, patch_width) or
         (n_patches, patch_height, patch_width, n_channels)
         The collection of patches extracted from the image, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.
    Examples
    --------
    >>> from sklearn.feature_extraction import image
    >>> one_image = np.arange(16).reshape((4, 4))
    >>> one_image
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> print(patches.shape)
    (9, 2, 2)
    >>> patches[0]
    array([[0, 1],
           [4, 5]])
    >>> patches[1]
    array([[1, 2],
           [5, 6]])
    >>> patches[8]
    array([[10, 11],
           [14, 15]])
    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = extract_patches(image, patch_shape=(p_h, p_w, n_colors), extraction_step=1)

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches


def extract_patches_3d(volume, patch_size, max_patches=None, random_state=None):
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
    v_x, v_y, v_z = volume.shape[:3]
    p_x, p_y, p_z = patch_size

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
    volume = volume.reshape((v_x, v_y, v_z, -1))
    n_colors = volume.shape[-1]

    extracted_patches = extract_patches(volume, patch_shape=(p_x, p_y, p_z, n_colors), extraction_step=1)

    n_patches = _compute_n_patches_3d(v_x, v_y, v_z, p_x, p_y, p_z, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(v_x - p_x + 1, size=n_patches)
        j_s = rng.randint(v_y - p_y + 1, size=n_patches)
        k_s = rng.randint(v_z - p_z + 1, size=n_patches)

        patches = extracted_patches[i_s, j_s, k_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_x, p_y, p_z, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_x, p_y, p_z))
    else:
        return patches

#max_patches = 50
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
    print(volume.shape)
    volume = np.moveaxis(volume, 0, -1)
    volume = volume.reshape((v_x, v_y, v_z, -1))
    #volume = volume.reshape((-1,v_x, v_y, v_z))
    n_colors = volume.shape[-1]
    extracted_patches = extract_patches(volume, patch_shape=(p_x, p_y, p_z, n_colors), extraction_step=1)

    n_patches = _compute_n_patches_3d(v_x, v_y, v_z, p_x, p_y, p_z, max_patches)
    # check the indexes where mask is True

    M = np.array(np.where(mask[:,int(p_x / 2): int(v_x - p_x / 2),
                          int(p_y / 2):int(v_y - p_y / 2),
                          int(p_z / 2):int(v_z - p_z / 2)] == True)).T


    #max_patches = 50
    if max_patches:
        rng = check_random_state(random_state)

        indx = rng.randint(len(M), size=n_patches)
        i_s = M[indx][:, 1] #0
        j_s = M[indx][:, 2] #1
        k_s = M[indx][:, 3] #2
        print('extracted indexes', i_s, j_s, k_s)
        patches = extracted_patches[i_s, j_s, k_s, 0]
    else:
        patches = extracted_patches
    print('aqui')
    patches = patches.reshape(-1, p_x, p_y, p_z, n_colors)
    patches = np.transpose(patches, axes=[0,4, 1, 2, 3])
    #patches = patches.reshape(n_colors, p_x, p_y, p_z)



    # # remove the color dimension if useless
    # if patches.shape[-1] == 1:
    #     return patches.reshape((n_patches, p_x, p_y, p_z))
    # else:
    #     return patches

    return patches


def reconstruct_from_patches_2d(patches, image_size):
    """Reconstruct the image from all of its patches.
    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    patches : array, shape = (n_patches, patch_height, patch_width) or
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.
    image_size : tuple of ints (image_height, image_width) or
        (image_height, image_width, n_channels)
        the size of the image that will be reconstructed
    Returns
    -------
    image : array, shape = image_size
        the reconstructed image
    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i:i + p_h, j:j + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            # divide by the amount of overlap
            # XXX: is this the most efficient way? memory-wise yes, cpu wise?
            img[i, j] /= float(min(i + 1, p_h, i_h - i) *
                               min(j + 1, p_w, i_w - j))
    return img


def reconstruct_from_patches_3d(patches, volume_size):
    """Reconstruct the volume from all of its patches.
    Patches are assumed to overlap and the volume is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    patches : array, shape = (n_patches, patch_x, patch_y, patch_z)
    volume_size : tuple of ints (volume_x, volume_y, volume_z)
        the size of the image that will be reconstructed
    Returns
    -------
    volume : array, shape = volume_size
        the reconstructed volume
    """
    v_x, v_y, v_z = volume_size[:3]
    p_num, p_x, p_y, p_z = patches.shape[1:5]
    vol = np.zeros(volume_size)
    # compute the dimensions of the patches array
    n_x = v_x - p_x + 1
    n_y = v_y - p_y + 1
    n_z = v_z - p_z + 1
    for p, (num,i, j, k) in zip(patches, product(range(p_num),range(n_x), range(n_y), range(n_z))):
        vol[i:i + p_x, j:j + p_y, k:k + p_z] += p[num]

    for i in range(v_x):
        for j in range(v_y):
            for k in range(v_z):
                # divide by the amount of overlap
                # XXX: is this the most efficient way? memory-wise yes, cpu wise?
                vol[i, j, k] /= float(min(i + 1, p_x, v_x - i) *
                                      min(j + 1, p_y, v_y - j) *
                                      min(k + 1, p_z, v_z - k))
    return vol

def extract_allpatches_intersection(volume, mask, patch_size, overlap = (0,0,0,0)):
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

    #extracted_patches = extract_patches(volume, patch_shape=(p_x, p_y, p_z, n_colors), extraction_step=1)
    #print(extracted_patches.shape)

    # check the indexes where mask is True

    M = np.array(np.where(mask[:,int(p_x / 2): int(v_x - p_x / 2),
                          int(p_y / 2):int(v_y - p_y / 2),
                          int(p_z / 2):int(v_z - p_z / 2)] == True)).T

    print('here4')
    print((M[0][0], M[0][1], M[0][2], M[0][3]))
    import ndpatch

    volume_helper = volume.reshape(n_colors, v_x, v_y, v_z)


    #indices that comply with overlapping
    indices = get_patches_indx(volume_helper.shape, patch_size, overlap, start=(M[0][0], M[0][1], M[0][2], M[0][3]))

    #indices where Mask is present
    M_helper = M.tolist()
    indices_set = set(map(tuple, indices))
    M_set = set(map(tuple, M_helper))


    print('whre to extract patches:', indices_set.intersection((M_set)))
    print('Number of patches available:', len(indices_set.intersection((M_set))))
    aux_intersection = indices_set.intersection((M_set))

    return aux_intersection

def extract_allpatches_3d_fromMask(volume, patch_size, aux_intersection):
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
    patches = []
    index_list = []

    for indx in aux_intersection:
        print(indx)

        patch_single = extracted_patches[indx[1], indx[2], indx[3], 0]
        patch_single = patch_single.reshape(n_colors, p_x, p_y, p_z)
        # patch_single = np.transpose(patch_single, axes=[0,4, 1, 2, 3])
        patches.append(patch_single)
        index_list.append(indx)


    print('patch_final', patches[0][0,0,0,0])

    return patches, index_list

def extract_allpatches_3d_fromMask_new(volume, mask, patch_size, overlap = (0,0,0,0)):
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
    #indices = get_patches_indx_new(mask= mask, patch_shape= patch_size, overlap= overlap, start=(M[0][0], M[0][1], M[0][2], M[0][3]))
    indices = get_patches_indx_new(mask= mask, patch_shape= patch_size, overlap= overlap, start=(0, 0, 0, 0))

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
    print(index_list)



    return patches, index_list





def get_patches_indx(array_shape, patch_shape, overlap, start):
    array_shape = np.asarray(array_shape)
    patch_shape = np.asarray(patch_shape)
    overlap = np.asarray(overlap)
    stop = start + array_shape
    step = patch_shape - overlap
    slices = [slice(_start, _stop, _step) for _start, _stop, _step in zip(start, stop, step)]
    return np.array(np.mgrid[slices].reshape(len(slices), -1).T, dtype=np.int).tolist()

def get_patches_indx_new(mask, patch_shape, overlap, start):
    array_shape = np.asarray(mask.shape)
    patch_shape = np.asarray(patch_shape)
    overlap = np.asarray(overlap)
    stop = start + array_shape
    step = patch_shape - overlap
    slices = [slice(_start, _stop, _step) for _start, _stop, _step in zip(start, stop, step)]
    all = np.array(np.mgrid[slices].reshape(len(slices), -1).T, dtype=np.int).tolist()
    return all