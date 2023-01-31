from copy import deepcopy

import numpy as np

from batchgenerators.transforms import BrightnessMultiplicativeTransform, BrightnessTransform, GaussianNoiseTransform, \
    MirrorTransform, SpatialTransform, GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform, AbstractTransform
from batchgenerators.transforms.color_transforms import ClipValueRange
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, PadTransform #, FillupPadTransform
from batchgenerators.transforms.noise_transforms import BlankSquareNoiseTransform, GaussianBlurTransform 
from batchgenerators.transforms.spatial_transforms import ResizeTransform, ZoomTransform, Rot90Transform, MirrorTransform
from batchgenerators.transforms.utility_transforms import AddToDictTransform, CopyTransform, NumpyToTensor, \
    ReshapeTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform

import random
import copy
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, **data_dict):
        #print('transf', data_dict['data'].shape)
        xi = self.transform(**data_dict)
        xj = self.transform(**data_dict)
        xi['data'] = (xi['data'], xj['data'])
        return xi

class DoubleHeadedTransform(AbstractTransform):
    def __init__(self, transform_i, transform_j):
        super().__init__()
        self.transform_i = transform_i 
        self.transform_j = transform_j

    def __call__(self, **data_dict):
        out_dict_1 = self.transform_i(**data_dict)
        out_dict_2 = self.transform_j(**data_dict)
        out_dict_1['data'] = (out_dict_1['data'], out_dict_2['data'])
        # out_dict_1['data_2'] = out_dict_2['data']
        return out_dict_1
    
    def __repr__(self):
        return str(type(self).__name__) + " \t ( " + repr(self.transform_i) + "\n\t" + repr(self.transform_i)+ " )"

class SplitHeadTransformation(AbstractTransform):
    def __init__(self, transform_base, transform_split):
        super().__init__()
        self.transform_base = transform_base
        self.transform_split = transform_split
        self.final_transform = NumpyToTensor()

    def __call__(self, **data_dict):
        out_dict_base = self.transform_base(**data_dict)
        out_dict_split = self.transform_split(**{'data':deepcopy(out_dict_base['data'])})
        # out_dict_split = self.transform_base(**{'data':data_dict['data']})
        out_dict_base['data'] = (self.final_transform(**out_dict_base)['data'], self.final_transform(**out_dict_split)['data'])
        return out_dict_base

    def __repr__(self):
        return str(type(self).__name__) + " \t ( " + repr(self.transform_base) + "\n\t" + repr(self.transform_base)+repr(self.transform_split)+ " )"

class RandomIntensityScale(AbstractTransform):

    def __init__(self, prob:float=0.5, factors: tuple=(-1, -1)):
        """Multiplies the intensity of an input with random values drawn in the range factors.
        

        Args:
            prob ([type]): [Probability for the transformation]
            factors (tuple, optional): [Range of values from which factors are drawn]. Defaults to (-1, -1).
        """
        super().__init__()
        self.prob= prob 
        self.factors = factors
        

    def __call__(self, **data_dict):
        if np.random.uniform()<= self.prob:
            data_dict['data'] = data_dict['data']*np.random.uniform(self.factors[0], self.factors[1])
        return data_dict

def get_transforms(mode="train", target_size=32, add_noise=False, mask_type="", base_train='default',
                   rotate=True, elastic_deform=True, rnd_crop=False, color_augment=True,
                   add_transforms=(), double_headed=False, transform_type='single'):
    transform_list = []
    noise_list = []

    if isinstance(target_size, int):
        target_size1 = target_size2 = target_size
    elif isinstance(target_size, (list, tuple)):
        target_size1 = target_size[0]
        target_size2 = target_size[1]
    else:
        raise NotImplementedError

    if mode == "train":
        if base_train == 'default':
            transform_list = [  # FillupPadTransform(min_size=(target_size1 + 5, target_size2 + 5)),
                ResizeTransform(target_size=(target_size1 + 1, target_size2 + 1),
                                order=1, concatenate_list=True),

                # RandomCropTransform(crop_size=(target_size + 5, target_size + 5)),
                MirrorTransform(axes=(2,)),
                # ReshapeTransform(new_shape=(1, -1, "h", "w")),
                SpatialTransform(patch_size=(target_size1, target_size2), random_crop=rnd_crop,
                                 patch_center_dist_from_border=(target_size1 // 2, target_size2 // 2),
                                 do_elastic_deform=elastic_deform, alpha=(0., 100.), sigma=(10., 13.),
                                 do_rotation=rotate,
                                 angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                 scale=(0.9, 1.2),
                                 border_mode_data="nearest", border_mode_seg="nearest"),
                # ReshapeTransform(new_shape=(batch_size, -1, "h", "w"))
            ]
        elif base_train == 'random_crop':
            rnd_crop = True
            transform_list = [  # FillupPadTransform(min_size=(target_size1 + 5, target_size2 + 5)),
                ResizeTransform(target_size=(target_size1 + 1, target_size2 + 1),
                                order=1, concatenate_list=True),

                # RandomCropTransform(crop_size=(target_size + 5, target_size + 5)),
                MirrorTransform(axes=(2,)),
                # ReshapeTransform(new_shape=(1, -1, "h", "w")),
                SpatialTransform(patch_size=(target_size1, target_size2), random_crop=rnd_crop,
                                 patch_center_dist_from_border=(target_size1 // 3, target_size2 // 3),
                                 do_elastic_deform=elastic_deform, alpha=(0., 100.), sigma=(10., 13.),
                                 do_rotation=rotate,
                                 angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                 scale=(0.9, 1.2),
                                 border_mode_data="nearest", border_mode_seg="nearest"),
                # ReshapeTransform(new_shape=(batch_size, -1, "h", "w"))
            ]
        else:
            raise NotImplementedError
        if color_augment:
            transform_list += [  # BrightnessTransform(mu=0, sigma=0.2),
                BrightnessMultiplicativeTransform(multiplier_range=(0.95, 1.1))]

        transform_list += [
            GaussianNoiseTransform(noise_variance=(0., 0.05)),
            ClipValueRange(min=-1.5, max=1.5),
            # CopyTransform({"data": ("data_clean", "data_blur")}, copy=True),
            # GaussianBlurTransform(blur_sigma=3.5, data_key="data_blur")
        ]

        noise_list = []  # [GaussianNoiseTransform(noise_variance=(0., 0.2))]

        if mask_type == "blank":
            noise_list += [BlankSquareNoiseTransform(squre_size=(target_size // 2),
                                                     noise_val=0,
                                                     n_squres=1,
                                                     square_pos=[(0, 0), (target_size // 2, 0), (0, target_size // 2),
                                                                 (target_size // 2, target_size // 2)])]
        elif mask_type == "context":
            noise_list += [BlankSquareNoiseTransform(squre_size=(0, target_size // 2),
                                                     noise_val=(-1.5, +1.5),
                                                     n_squres=1,)]
        elif mask_type == 'context_david':
            noise_list += [BlankSquareNoiseTransform(squre_size=(0, target_size // 2),
                                                     noise_val=(-1.5, +1.5),
                                                     n_squres=1,)]
        elif mask_type == "noise":
            noise_list += [BlankSquareNoiseTransform(squre_size=(0, target_size // 2),
                                                     noise_val=(-1.5, +1.5),
                                                     n_squres=(0, 3),
                                                     channel_wise_n_val=True)]
        elif mask_type == "test":
            pass
            # noise_list += [SquareMaskTransform(squre_size=(0, np.max(target_size) // 2),
            #                                    noise_val=(-1.5, +1.5),
            #                                    n_squres=(0, 3),
            #                                    channel_wise_n_val=True)]

        elif mask_type == 'noise_rnd_scale':
            noise_list += [RandomIntensityScale(), BlankSquareNoiseTransform(squre_size=(0, target_size // 2),
                                                                             noise_val=(-1.5, +1.5),
                                                                             n_squres=(0, 3),
                                                                             channel_wise_n_val=True)]

        elif mask_type == "gaussian":
            noise_list += [GaussianNoiseTransform(noise_variance=(0., 0.2))]

        elif isinstance(mask_type, float):
            noise_list += [RndTransform(transform=Compose([BlankSquareNoiseTransform(squre_size=(0, target_size // 2),
                                                                                     noise_val=(-1.5, +1.5),
                                                                                     n_squres=(0, 3),
                                                                                     channel_wise_n_val=False),
                                                           AddToDictTransform(in_key="has_no_noise", in_val=0)]),
                                        prob=mask_type,
                                        alternative_transform=AddToDictTransform(in_key="has_no_noise", in_val=1))]

    elif mode == "val":

        transform_list = [  # FillupPadTransform(min_size=(target_size1 + 5, target_size2 + 5)),
            ResizeTransform(target_size=(target_size1 + 1, target_size2 + 1),
                            order=1, concatenate_list=True),
            CenterCropTransform(crop_size=(target_size1, target_size2)),
            ClipValueRange(min=-1.5, max=1.5),
            # BrightnessTransform(mu=0, sigma=0.2),
            # BrightnessMultiplicativeTransform(multiplier_range=(0.95, 1.1)),
            # CopyTransform({"data": "data_clean"}, copy=True)
        ]

        # noise_list = [CopyTransform({"data": ("data_clean", "data_blur")}, copy=True)]
        #
        # if mask_type == "blank":
        #     noise_list += [CopyTransform({"data": ("data_mask_1", "data_mask_2",
        #                                            "data_mask_3", "data_mask_4")}, copy=True),
        #                    BlankSquareNoiseTransform(squre_size=(target_size // 2), noise_val=0, n_squres=1,
        #                                              square_pos=[(0, 0)], data_key="data_mask_1"),
        #                    BlankSquareNoiseTransform(squre_size=(target_size // 2), noise_val=0, n_squres=1,
        #                                              square_pos=[(target_size // 2, 0)], data_key="data_mask_2"),
        #                    BlankSquareNoiseTransform(squre_size=(target_size // 2), noise_val=0, n_squres=1,
        #                                              square_pos=[(0, target_size // 2)], data_key="data_mask_3"),
        #                    BlankSquareNoiseTransform(squre_size=(target_size // 2), noise_val=0, n_squres=1,
        #                                              square_pos=[(target_size // 2, target_size // 2)],
        #                                              data_key="data_mask_4")]
        #
        # noise_list += [  # GaussianNoiseTransform(noise_variance=(0.1, 0.2)),
        #     GaussianBlurTransform(blur_sigma=1.5, data_key="data_blur")]

    if transform_type == 'split':
        final_transform = SplitHeadTransformation(Compose(transform_list), Compose(noise_list))
        return final_transform
    if add_noise:
        transform_list = transform_list + noise_list

    # if add_resize:
    #     resize_list = [
    #         CopyTransform({"seg": "label"}, copy=True),
    #
    #         CopyTransform({"data": "data_1"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1),
    #         CopyTransform({"data": "data_2"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1),
    #         CopyTransform({"data": "data_3"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1),
    #         CopyTransform({"data": "data_4"}, copy=True),
    #         CopyTransform({"data_1": "data"}, copy=True),
    #
    #         CopyTransform({"data_clean": "data_clean_1"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1, data_key="data_clean"),
    #         CopyTransform({"data_clean": "data_clean_2"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1, data_key="data_clean"),
    #         CopyTransform({"data_clean": "data_clean_3"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1, data_key="data_clean"),
    #         CopyTransform({"data_clean": "data_clean_4"}, copy=True),
    #         CopyTransform({"data_clean_1": "data_clean"}, copy=True),
    #     ]
    #
    #     tranform_list = tranform_list + resize_list

    if transform_type == 'split':
        final_transform = SplitHeadTransformation(Compose(transform_list), Compose(noise_list))
        return final_transform
    if add_noise:
        transform_list = transform_list + noise_list

    transform_list += list(add_transforms)

    transform_list.append(NumpyToTensor())

    final_transform = Compose(transform_list)
    if double_headed:
        final_transform = DoubleHeadedTransform(final_transform, final_transform)
    return final_transform


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])


    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)


    return xvals, yvals

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def nonlinear_transformation(x):
    # x = np.clip(x, -1000, 1000)
    # x = NormalizeData(x)
    points = [[np.min(x), np.min(x)], [np.random.uniform(np.min(x), np.max(x)), np.random.uniform(np.min(x), np.max(x))], [np.random.uniform(np.min(x), np.max(x)), np.random.uniform(np.min(x), np.max(x))], [np.max(x), np.max(x)]]
    #points = [[0,0], [random.random(), random.random()], [random.random(), random.random()], [1,1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=50)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)

    return nonlinear_x


def local_pixel_shuffling(x, prob=0.5):

    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, _, img_rows, img_cols, img_deps = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[0, 0, noise_x:noise_x+block_noise_size_x,
                               noise_y:noise_y+block_noise_size_y,
                               noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x,
                                 block_noise_size_y,
                                 block_noise_size_z))
        image_temp[0, 0, noise_x:noise_x+block_noise_size_x,
                      noise_y:noise_y+block_noise_size_y,
                      noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x):
    _, _, img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:,:,
          noise_x:noise_x+block_noise_size_x,
          noise_y:noise_y+block_noise_size_y,
          noise_z:noise_z+block_noise_size_z] = np.random.rand(_, block_noise_size_x,
                                                               block_noise_size_y,
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x

def image_out_painting(x):
    _, _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4], ) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    noise_z = random.randint(3, img_deps-block_noise_size_z-3)
    x[:,:,
      noise_x:noise_x+block_noise_size_x,
      noise_y:noise_y+block_noise_size_y,
      noise_z:noise_z+block_noise_size_z] = image_temp[:,:, noise_x:noise_x+block_noise_size_x,
                                                       noise_y:noise_y+block_noise_size_y,
                                                       noise_z:noise_z+block_noise_size_z]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, :,
          noise_x:noise_x+block_noise_size_x,
          noise_y:noise_y+block_noise_size_y,
          noise_z:noise_z+block_noise_size_z] = image_temp[:,:, noise_x:noise_x+block_noise_size_x,
                                                           noise_y:noise_y+block_noise_size_y,
                                                           noise_z:noise_z+block_noise_size_z]
        cnt -= 1
    return x

class NonLinearTransform(AbstractTransform):
    """Non linear Transform of data based on bezier_curve"""

    def __init__(self, data_key="data",  p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key

    def __call__(self, **data_dict):
        # for b in range(len(data_dict[self.data_key])):
        #     if np.random.uniform() < self.p_per_sample:
        #         data_dict[self.data_key][b] = nonlinear_transformation(data_dict[self.data_key][b])
        # return data_dict

        if np.random.uniform() < self.p_per_sample:
            data_dict[self.data_key] = nonlinear_transformation(data_dict[self.data_key])
        return data_dict

class LocalPixelShuffling(AbstractTransform):
    """Local Shuffle Pixel"""

    def __init__(self, data_key="data",  p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key

    def __call__(self, **data_dict):
        # for b in range(len(data_dict[self.data_key])):
        #     if np.random.uniform() < self.p_per_sample:
        #         data_dict[self.data_key][b] = local_pixel_shuffling(data_dict[self.data_key][b], self.p_per_sample)
        # return data_dict
        if np.random.uniform() < self.p_per_sample:
            data_dict[self.data_key] = local_pixel_shuffling(data_dict[self.data_key], self.p_per_sample)
        return data_dict


class InOutPainting(AbstractTransform):
    """Inpainting and Outpainting"""

    def __init__(self, data_key="data",  p_per_sample=1, p_inpaint_rate =1):
        self.p_per_sample = p_per_sample
        self.p_inpaint_rate = p_inpaint_rate
        self.data_key = data_key

    def __call__(self, **data_dict):
        # for b in range(len(data_dict[self.data_key])):
        #     if random.random() < self.p_per_sample:
        #         if np.random.uniform() < self.p_inpaint_rate:
        #             #Inpainting
        #             data_dict[self.data_key][b] = image_in_painting(data_dict[self.data_key][b])
        #         else:
        #             #Outpainting
        #             data_dict[self.data_key][b] = image_out_painting(data_dict[self.data_key][b])
        # return data_dict
        if random.random() < self.p_per_sample:
            if np.random.uniform() < self.p_inpaint_rate:
                #Inpainting
                data_dict[self.data_key] = image_in_painting(data_dict[self.data_key])
            else:
                #Outpainting
                data_dict[self.data_key] = image_out_painting(data_dict[self.data_key])
        return data_dict




def get_simclr_pipeline_transform(mode='train', patch_size=(1,30,30,30), base_train = 'default', rnd_crop=False,
                                  elastic_deform=True, rotate=True, double_headed=False):

    tr_transforms = []

    if mode == "train":
        if base_train == 'default':
            # global transformations
            tr_transforms.append(MirrorTransform(axes=(2,)),) #axes (tuple of int): axes along which to mirror
            tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.95, 1.1), per_channel=True))
            tr_transforms.append(
                SpatialTransform(patch_size=(patch_size), random_crop=rnd_crop,
                                 patch_center_dist_from_border=[i // 2 for i in patch_size],
                                 do_elastic_deform=elastic_deform, alpha=(0., 100.), sigma=(10., 13.),
                                 do_rotation=rotate,
                                 angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                                 angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                                 angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                                 scale=(0.9, 1.2),
                                 border_mode_data="nearest", border_mode_seg="nearest"),
            )

            tr_transforms.append(GaussianNoiseTransform(noise_variance=(0., 0.05)))
            tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True))
        elif base_train == 'models_genesis':
            # global transformations

            tr_transforms.append(NonLinearTransform(p_per_sample=0.5))
            tr_transforms.append(LocalPixelShuffling(p_per_sample=0.5))
            tr_transforms.append(InOutPainting(p_per_sample=0.5, p_inpaint_rate=0.5))
            tr_transforms.append(MirrorTransform(axes=(2,)))
            tr_transforms.append(Rot90Transform(p_per_sample= 0.3))
            # tr_transforms.append(
            #     SpatialTransform_2(
            #         patch_size, [i // 2 for i in patch_size],
            #         do_rotation=True,
            #         angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            #         angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            #         angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            #         random_crop=True,
            #         p_rot_per_sample=0.3
            #     )
            # )
        else:
            raise NotImplementedError
    elif mode == "val":
        # tr_transforms.append(
        #     SpatialTransform(patch_size=(patch_size), random_crop=rnd_crop,
        #                      patch_center_dist_from_border=[i // 2 for i in patch_size]),)
        tr_transforms = tr_transforms

    elif mode == "fit":
        tr_transforms = tr_transforms
        # tr_transforms.append(SpatialTransform_2(patch_size=(patch_size), random_crop=rnd_crop,
        #                      patch_center_dist_from_border=[i // 2 for i in patch_size],
        #                      do_elastic_deform=elastic_deform, deformation_scale=(0, 0.25),
        #                      do_rotation=rotate,
        #                      angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
        #                      scale=(0.9, 1.2),
        #                      border_mode_data="nearest", border_mode_seg="nearest",
        #                     p_el_per_sample=0.2, p_rot_per_sample=0.2, p_scale_per_sample=0.2)
        # )
        # tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
        # tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
        # tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.20)) #0.15
        # tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.2, 0.5), different_sigma_per_channel=True,
        #                                            p_per_channel=0.5, p_per_sample=0.10)) #0.10

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    if double_headed:
        #tr_transforms = SimCLRDataTransform(tr_transforms, tr_transforms)
        #tr_transforms = SimCLRDataTransform(tr_transforms)
        tr_transforms = DoubleHeadedTransform(tr_transforms, tr_transforms)
    return tr_transforms